import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import argparse
from pathlib import Path
from collections import defaultdict
import queue
import threading
from sklearn.metrics.pairwise import cosine_similarity

# Configuration class
@dataclass
class TrainingConfig:
    model_name: str = 'sentence-transformers/all-mpnet-base-v2'
    mlp_hidden_size: int = 256
    dropout_rate: float = 0.3
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 2e-5
    temperature: float = 0.07
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    num_workers: int = max(1, multiprocessing.cpu_count() - 1)
    max_length: int = 128
    train_mode: str = 'mlp'  # 'mlp' or 'full'
    negative_ratio: float = 1.0  # ratio of negatives to positives
    similarity_threshold: float = 0.7  # for hard negative mining

def setup_logger(output_dir: str) -> logging.Logger:
    """Setup detailed logging with both file and console handlers."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'training_{timestamp}.log')
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Setup logger
    logger = logging.getLogger('DomainConceptTrainer')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class ContrastiveModel(nn.Module):
    """Model supporting both frozen and full fine-tuning approaches."""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.encoder = SentenceTransformer(config.model_name)
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, config.mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.mlp_hidden_size, config.mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.mlp_hidden_size // 2, config.mlp_hidden_size // 4)
        )
        
        # Freeze encoder if in MLP mode
        if config.train_mode == 'mlp':
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        # Get embeddings
        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Project through MLP
        outputs = self.mlp(embeddings)
        
        # Normalize outputs
        outputs = F.normalize(outputs, p=2, dim=1)
        return outputs

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with efficient pair generation."""
    def __init__(self, 
                anchors: List[str],
                anchor_labels: List[str],
                positives: Dict[str, List[str]],
                negatives: Dict[str, List[str]]):
        self.anchors = anchors
        self.anchor_labels = anchor_labels
        self.positives = positives
        self.negatives = negatives
        
        # Create pairs
        self.pairs = self._create_pairs()
        
    def _create_pairs(self) -> List[Tuple[str, str, int]]:
        """Create training pairs with parallel processing."""
        pairs = []
        for idx, (anchor, label) in enumerate(zip(self.anchors, self.anchor_labels)):
            # Add positive pairs
            for pos in self.positives[label]:
                pairs.append((anchor, pos, 1))
            
            # Add negative pairs
            for neg in self.negatives[label]:
                pairs.append((anchor, neg, 0))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        return self.pairs[idx]

class DataProcessor:
    """Handles data loading, preprocessing, and pair generation."""
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def load_and_split_data(self, 
                           attributes_path: str, 
                           concepts_path: str) -> Tuple[Dict, Dict, Dict]:
        """Load data and create train/val/test splits."""
        self.logger.info("Loading datasets...")
        
        # Load data
        attributes_df = pd.read_csv(attributes_path)
        concepts_df = pd.read_csv(concepts_path)
        
        # Create domain-concept labels
        attributes_df['label'] = attributes_df.apply(
            lambda x: f"{x['domain']}-{x['concept']}", axis=1
        )
        concepts_df['label'] = concepts_df.apply(
            lambda x: f"{x['domain']}-{x['concept']}", axis=1
        )
        
        # Split attributes data
        train_df, temp_df = train_test_split(
            attributes_df,
            train_size=self.config.train_ratio,
            stratify=attributes_df['label'],
            random_state=42
        )
        
        val_size = self.config.val_ratio / (1 - self.config.train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            stratify=temp_df['label'],
            random_state=42
        )
        
        self.logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Process splits
        train_data = self._process_split(train_df, concepts_df, 'train')
        val_data = self._process_split(val_df, concepts_df, 'val')
        test_data = self._process_split(test_df, concepts_df, 'test')
        
        return train_data, val_data, test_data
    
    def _process_split(self, 
                      attributes_df: pd.DataFrame,
                      concepts_df: pd.DataFrame,
                      split_name: str) -> Dict:
        """Process each data split with parallel negative mining."""
        self.logger.info(f"Processing {split_name} split...")
        
        # Prepare anchors from concepts
        anchors = []
        anchor_labels = []
        for _, row in concepts_df.iterrows():
            anchor_text = f"{row['domain']}-{row['concept']}: {row['concept_definition']}"
            anchors.append(anchor_text)
            anchor_labels.append(row['label'])
            
        # Prepare positives and negatives
        positives = defaultdict(list)
        negatives = defaultdict(list)
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for label in anchor_labels:
                futures.append(
                    executor.submit(
                        self._process_label_pairs,
                        label,
                        attributes_df,
                        concepts_df
                    )
                )
            
            # Collect results
            for future in tqdm(futures, desc=f"Processing {split_name} pairs"):
                label, pos, neg = future.result()
                positives[label].extend(pos)
                negatives[label].extend(neg)
        
        return {
            'anchors': anchors,
            'anchor_labels': anchor_labels,
            'positives': positives,
            'negatives': negatives
        }

def _process_label_pairs(self,
                          label: str,
                          attributes_df: pd.DataFrame,
                          concepts_df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
        """Process positive and negative pairs for a given label."""
        # Get positives
        pos_df = attributes_df[attributes_df['label'] == label]
        positives = [
            f"{row['attribute_name']} {row['description']}"
            for _, row in pos_df.iterrows()
        ]
        
        # Get negatives
        neg_df = attributes_df[attributes_df['label'] != label]
        
        # Calculate similarities for hard negative mining
        if len(neg_df) > 0:
            # Create embeddings for similarity calculation
            encoder = SentenceTransformer(self.config.model_name)
            pos_embeddings = encoder.encode(
                [f"{row['attribute_name']} {row['description']}" for _, row in pos_df.iterrows()],
                show_progress_bar=False
            )
            neg_embeddings = encoder.encode(
                [f"{row['attribute_name']} {row['description']}" for _, row in neg_df.iterrows()],
                show_progress_bar=False
            )
            
            # Calculate similarities
            similarities = cosine_similarity(pos_embeddings, neg_embeddings)
            
            # Get hard negatives (high similarity scores)
            hard_negative_indices = np.where(similarities.max(axis=0) > self.config.similarity_threshold)[0]
            
            # Select negatives
            num_negatives = int(len(positives) * self.config.negative_ratio)
            num_hard = min(len(hard_negative_indices), num_negatives // 2)
            num_random = num_negatives - num_hard
            
            # Combine hard and random negatives
            selected_indices = np.concatenate([
                hard_negative_indices[:num_hard],
                np.random.choice(
                    [i for i in range(len(neg_df)) if i not in hard_negative_indices],
                    size=num_random,
                    replace=False
                )
            ])
            
            negatives = [
                f"{row['attribute_name']} {row['description']}"
                for idx, row in neg_df.iloc[selected_indices].iterrows()
            ]
        else:
            negatives = []
        
        return label, positives, negatives

class ContrastiveLoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, anchors: torch.Tensor, positives: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        # Compute similarities
        pos_sim = torch.sum(anchors * positives, dim=1) / self.temperature
        neg_sim = torch.matmul(anchors, negatives.t()) / self.temperature
        
        # Construct labels and logits
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        return loss

class ModelTrainer:
    """Handles model training and evaluation."""
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = ContrastiveModel(config).to(self.device)
        self.criterion = ContrastiveLoss(temperature=config.temperature)
        
        # Setup optimizer
        if config.train_mode == 'mlp':
            params = self.model.mlp.parameters()
        else:
            params = self.model.parameters()
            
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
        
    def train(self, train_data: Dict, val_data: Dict, output_dir: str) -> None:
        """Train the model with logging and checkpointing."""
        self.logger.info("Starting training...")
        
        # Create datasets
        train_dataset = ContrastiveDataset(**train_data)
        val_dataset = ContrastiveDataset(**val_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self._validate(val_loader)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # Save checkpoint if improved
            if val_loss < best_loss:
                best_loss = val_loss
                self._save_checkpoint(output_dir, epoch, val_loss)
                
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as pbar:
            for batch_idx, (anchors, positives, labels) in enumerate(pbar):
                # Get embeddings
                anchor_embeds = self.model(anchors)
                positive_embeds = self.model(positives)
                
                # Sample in-batch negatives
                with torch.no_grad():
                    negative_embeds = positive_embeds.roll(1, dims=0)
                
                # Compute loss
                loss = self.criterion(anchor_embeds, positive_embeds, negative_embeds)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for anchors, positives, labels in val_loader:
                # Get embeddings
                anchor_embeds = self.model(anchors)
                positive_embeds = self.model(positives)
                negative_embeds = positive_embeds.roll(1, dims=0)
                
                # Compute loss
                loss = self.criterion(anchor_embeds, positive_embeds, negative_embeds)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def _save_checkpoint(self, output_dir: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        if self.config.train_mode == 'full':
            self.model.encoder.save(checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'model.pt'))
        
        # Save config
        with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.config), f, indent=2)
            
        self.logger.info(f"Saved checkpoint to {checkpoint_dir}")

class PredictionService:
    """Handles all prediction scenarios."""
    def __init__(self, model_path: str, config: Optional[TrainingConfig] = None):
        self.logger = logging.getLogger('DomainConceptPredictor')
        
        # Load config
        if config is None:
            with open(os.path.join(model_path, 'config.json')) as f:
                config_dict = json.load(f)
                self.config = TrainingConfig(**config_dict)
        else:
            self.config = config
            
        # Initialize model
        self.model = self._load_model(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def _load_model(self, model_path: str) -> ContrastiveModel:
        """Load trained model."""
        model = ContrastiveModel(self.config)
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
        return model
    
    def predict_single(self, 
                      text: str,
                      concept_texts: List[str],
                      top_k: int = 3) -> List[Dict[str, Any]]:
        """Predict for a single text."""
        self.model.eval()
        
        with torch.no_grad():
            # Get embeddings
            text_embed = self.model([text])
            concept_embeds = self.model(concept_texts)
            
            # Calculate similarities
            similarities = F.cosine_similarity(
                text_embed.unsqueeze(1),
                concept_embeds.unsqueeze(0),
                dim=2
            ).squeeze()
            
            # Get top-k
            top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(concept_texts)))
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(top_k_values, top_k_indices)):
                concept_text = concept_texts[idx]
                domain, concept = concept_text.split(':')[0].strip().split('-')
                
                results.append({
                    'rank': i + 1,
                    'domain': domain,
                    'concept': concept,
                    'confidence': float(score)
                })
                
            return results
    
    def predict_batch(self,
                     texts: List[str],
                     concept_texts: List[str],
                     batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predict for a batch of texts."""
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Get predictions for batch
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = []
                for text in batch_texts:
                    futures.append(
                        executor.submit(
                            self.predict_single,
                            text,
                            concept_texts
                        )
                    )
                
                # Collect results
                for future in futures:
                    results = future.result()
                    if results:
                        all_results.append(results[0])  # Take top prediction
                    
        return all_results
    
    def predict_csv(self,
                   input_path: str,
                   concept_texts: List[str],
                   output_path: str) -> None:
        """Predict for texts in CSV file."""
        # Read input
        df = pd.read_csv(input_path)
        
        # Prepare texts
        texts = [
            f"{row['attribute_name']} {row['description']}"
            for _, row in df.iterrows()
        ]
        
        # Get predictions
        predictions = self.predict_batch(texts, concept_texts)
        
        # Add predictions to DataFrame
        df['predicted_domain'] = [p['domain'] for p in predictions]
        df['predicted_concept'] = [p['concept'] for p in predictions]
        df['confidence'] = [p['confidence'] for p in predictions]
        
        # Save results
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved predictions to {output_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Domain-Concept Training and Prediction')
    
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Operation mode')
    parser.add_argument('--attributes', type=str, required=True,
                      help='Path to attributes CSV')
    parser.add_argument('--concepts', type=str, required=True,
                      help='Path to concepts CSV')
    parser.add_argument('--output-dir', type=str, default='./output',
                      help='Output directory')
    parser.add_argument('--train-mode', choices=['mlp', 'full'], default='mlp',
                      help='Training mode: MLP only or full fine-tuning')
    parser.add_argument('--model-path', type=str,
                      help='Path to trained model for prediction')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(args.output_dir)
    
    try:
        if args.mode == 'train':
            # Create config
            config = TrainingConfig(
                train_mode=args.train_mode,
                batch_size=args.batch_size
            )
            
            # Process data
            data_processor = DataProcessor(config, logger)
            train_data, val_data, test_data = data_processor.load_and_split_data(
                args.attributes,
                args.concepts
            )
            
            # Train model
            trainer = ModelTrainer(config, logger)
            trainer.train(train_data, val_data, args.output_dir)
            
        else:  # predict mode
            if not args.model_path:
                raise ValueError("--model-path required for predict mode")
                
            # Load concepts
            concepts_df = pd.read_csv(args.concepts)
            concept_texts = [
                f"{row['domain']}-{row['concept']}: {row['concept_definition']}"
                for _, row in concepts_df.iterrows()
            ]
            
            # Initialize predictor
            predictor = PredictionService(args.model_path)
            output_path = os.path.join(args.output_dir, 'predictions.csv')
            predictor.predict_csv(args.attributes, concept_texts, output_path)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
