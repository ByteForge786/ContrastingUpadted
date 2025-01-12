import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import os
import json
from datetime import datetime
from dataclasses import dataclass
import random
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Setup logging
def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

@dataclass
class ModelConfig:
    model_name: str = 'microsoft/mpnet-base'
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 2e-5
    max_length: int = 128
    temperature: float = 0.07
    train_ratio: float = 0.8
    num_workers: int = multiprocessing.cpu_count()
    hidden_size: int = 768  # Should match model's hidden size

class DualEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize encoders
        self.definition_encoder = AutoModel.from_pretrained(config.model_name)
        self.attribute_encoder = AutoModel.from_pretrained(config.model_name)
        
        # Projection heads
        self.definition_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        self.attribute_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
    def encode_definition(self, input_ids: torch.Tensor, 
                         attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.definition_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        projected = self.definition_proj(pooled)
        return F.normalize(projected, p=2, dim=1)
    
    def encode_attribute(self, input_ids: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.attribute_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        projected = self.attribute_proj(pooled)
        return F.normalize(projected, p=2, dim=1)

class ContrastiveDataset(Dataset):
    def __init__(self, 
                 definition_texts: List[str],
                 attribute_texts: List[str],
                 positive_pairs: List[Tuple[int, int]],
                 tokenizer: AutoTokenizer,
                 max_length: int = 128):
        self.definition_texts = definition_texts
        self.attribute_texts = attribute_texts
        self.positive_pairs = positive_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.positive_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        def_idx, attr_idx = self.positive_pairs[idx]
        
        # Tokenize definition
        definition_encoding = self.tokenizer(
            self.definition_texts[def_idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize attribute
        attribute_encoding = self.tokenizer(
            self.attribute_texts[attr_idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'def_input_ids': definition_encoding['input_ids'].squeeze(0),
            'def_attention_mask': definition_encoding['attention_mask'].squeeze(0),
            'attr_input_ids': attribute_encoding['input_ids'].squeeze(0),
            'attr_attention_mask': attribute_encoding['attention_mask'].squeeze(0),
            'def_idx': torch.tensor(def_idx),
            'attr_idx': torch.tensor(attr_idx)
        }

class DataProcessor:
    def __init__(self, 
                 attributes_path: str,
                 concepts_path: str,
                 config: ModelConfig):
        self.attributes_path = Path(attributes_path)
        self.concepts_path = Path(concepts_path)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.attributes_df = None
        self.concepts_df = None
        self.definition_texts = []
        self.attribute_texts = []
        self.positive_pairs = []
        
    def process_data(self) -> Tuple[ContrastiveDataset, ContrastiveDataset]:
        """Load and process data into train/test datasets."""
        self.logger.info("Loading data files...")
        self.attributes_df = pd.read_csv(self.attributes_path)
        self.concepts_df = pd.read_csv(self.concepts_path)
        
        # Process concepts/definitions
        self.definition_texts = [
            f"{row['domain']}-{row['concept']}: {row['concept_definition']}"
            for _, row in self.concepts_df.iterrows()
        ]
        
        # Create domain-concept to index mapping
        dc_to_idx = {
            f"{row['domain']}-{row['concept']}": idx 
            for idx, row in self.concepts_df.iterrows()
        }
        
        # Process attributes with parallel processing
        self.logger.info("Processing attributes in parallel...")
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            
            for _, row in self.attributes_df.iterrows():
                futures.append(
                    executor.submit(
                        self._process_attribute,
                        row['attribute_name'],
                        row['description'],
                        row['domain'],
                        row['concept']
                    )
                )
            
            # Collect results
            for idx, future in enumerate(tqdm(futures)):
                attr_text = future.result()
                self.attribute_texts.append(attr_text)
                
                # Add positive pair
                dc_key = f"{self.attributes_df.iloc[idx]['domain']}-{self.attributes_df.iloc[idx]['concept']}"
                def_idx = dc_to_idx[dc_key]
                self.positive_pairs.append((def_idx, idx))
        
        # Split data
        return self._create_datasets()
    
    @staticmethod
    def _process_attribute(name: str, description: str, 
                          domain: str, concept: str) -> str:
        """Process single attribute text."""
        return f"{name} {description}"
    
    def _create_datasets(self) -> Tuple[ContrastiveDataset, ContrastiveDataset]:
        """Create train and test datasets."""
        # Split positive pairs
        train_pairs, test_pairs = train_test_split(
            self.positive_pairs,
            train_size=self.config.train_ratio,
            random_state=42
        )
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Create datasets
        train_dataset = ContrastiveDataset(
            self.definition_texts,
            self.attribute_texts,
            train_pairs,
            tokenizer,
            self.config.max_length
        )
        
        test_dataset = ContrastiveDataset(
            self.definition_texts,
            self.attribute_texts,
            test_pairs,
            tokenizer,
            self.config.max_length
        )
        
        self.logger.info(f"Created datasets - Train: {len(train_pairs)}, Test: {len(test_pairs)}")
        return train_dataset, test_dataset

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, definition_embeds: torch.Tensor,
                attribute_embeds: torch.Tensor) -> torch.Tensor:
        # Compute similarity matrix
        sim_matrix = torch.matmul(definition_embeds, attribute_embeds.T) / self.temperature
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        # Compute loss from both directions
        loss_def = F.cross_entropy(sim_matrix, labels)
        loss_attr = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_def + loss_attr) / 2

class ModelTrainer:
    def __init__(self, config: ModelConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = DualEncoder(config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Training components
        self.criterion = InfoNCELoss(temperature=config.temperature)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.5
        )
        
        # Metrics tracking
        self.best_loss = float('inf')
        self.best_model_path = None
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_dataset: ContrastiveDataset,
              val_dataset: ContrastiveDataset) -> None:
        """Train the model."""
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
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self._validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}"
            )
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_path = self._save_model(f"best_model")
                
        # Save final metrics
        self._save_training_results()
        
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            def_embeds = self.model.encode_definition(
                batch['def_input_ids'],
                batch['def_attention_mask']
            )
            
            attr_embeds = self.model.encode_attribute(
                batch['attr_input_ids'],
                batch['attr_attention_mask']
            )
            
            # Compute loss
            loss = self.criterion(def_embeds, attr_embeds)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                def_embeds = self.model.encode_definition(
                    batch['def_input_ids'],
                    batch['def_attention_mask']
                )
                
                attr_embeds = self.model.encode_attribute(
                    batch['attr_input_ids'],
                    batch['attr_attention_mask']
                )
                
                loss = self.criterion(def_embeds, attr_embeds)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def _save_model(self, name: str) -> str:
        """Save model and return path."""
        save_path = self.output_dir / name
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), save_path / 'model.pt')
        
# Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
            
        return str(save_path)
    
    def _save_training_results(self) -> None:
        """Save training metrics and results."""
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'best_model_path': self.best_model_path,
            'config': vars(self.config),
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
class PredictionService:
    def __init__(self, model_path: str):
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        
        # Load config
        with open(self.model_path / 'config.json') as f:
            config_dict = json.load(f)
        self.config = ModelConfig(**config_dict)
        
        # Initialize model and tokenizer
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def _load_model(self) -> DualEncoder:
        """Load the trained model."""
        model = DualEncoder(self.config)
        model_path = self.model_path / 'model.pt'
        model.load_state_dict(torch.load(model_path))
        return model
    
    @torch.no_grad()
    def predict_single(self, 
                      attribute_text: str,
                      definition_texts: List[str],
                      top_k: int = 3) -> List[Dict]:
        """Predict for a single attribute."""
        self.model.eval()
        
        # Tokenize attribute
        attr_encoding = self.tokenizer(
            attribute_text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get attribute embedding
        attr_embed = self.model.encode_attribute(
            attr_encoding['input_ids'],
            attr_encoding['attention_mask']
        )
        
        # Process each definition
        def_embeddings = []
        for def_text in definition_texts:
            def_encoding = self.tokenizer(
                def_text,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            def_embed = self.model.encode_definition(
                def_encoding['input_ids'],
                def_encoding['attention_mask']
            )
            def_embeddings.append(def_embed)
        
        # Stack all definition embeddings
        def_embeddings = torch.cat(def_embeddings)
        
        # Calculate similarities
        similarities = torch.matmul(attr_embed, def_embeddings.T).squeeze()
        
        # Get top-k matches
        top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(definition_texts)))
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(top_k_values, top_k_indices)):
            def_text = definition_texts[idx]
            domain, concept = def_text.split(':')[0].strip().split('-')
            
            results.append({
                'rank': i + 1,
                'domain': domain,
                'concept': concept,
                'confidence': float(score),
                'definition': def_text.split(':', 1)[1].strip()
            })
            
        return results
    
    def predict_batch(self, 
                     attribute_texts: List[str],
                     definition_texts: List[str],
                     batch_size: int = 32) -> List[List[Dict]]:
        """Predict for a batch of attributes."""
        self.model.eval()
        
        results = []
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            
            for attr_text in attribute_texts:
                futures.append(
                    executor.submit(
                        self.predict_single,
                        attr_text,
                        definition_texts
                    )
                )
            
            for future in tqdm(futures, desc="Processing predictions"):
                results.append(future.result())
                
        return results
    
    def predict_csv(self, 
                   input_path: str,
                   concepts_path: str,
                   output_path: str) -> None:
        """Predict for attributes in CSV file."""
        # Load input data
        df = pd.read_csv(input_path)
        
        # Load concepts
        concepts_df = pd.read_csv(concepts_path)
        definition_texts = [
            f"{row['domain']}-{row['concept']}: {row['concept_definition']}"
            for _, row in concepts_df.iterrows()
        ]
        
        # Prepare attribute texts
        attribute_texts = [
            f"{row['attribute_name']} {row['description']}"
            for _, row in df.iterrows()
        ]
        
        # Get predictions
        predictions = self.predict_batch(attribute_texts, definition_texts)
        
        # Process results
        results_df = pd.DataFrame()
        results_df['attribute_name'] = df['attribute_name']
        results_df['description'] = df['description']
        
        # Add top prediction for each attribute
        results_df['predicted_domain'] = [pred[0]['domain'] for pred in predictions]
        results_df['predicted_concept'] = [pred[0]['concept'] for pred in predictions]
        results_df['confidence'] = [pred[0]['confidence'] for pred in predictions]
        
        # Save results
        results_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved predictions to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Dual Encoder for Domain-Concept Matching')
    
    # Common arguments
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Operation mode')
    parser.add_argument('--attributes', type=str, required=True,
                      help='Path to attributes CSV')
    parser.add_argument('--concepts', type=str, required=True,
                      help='Path to concepts CSV')
    parser.add_argument('--output-dir', type=str, default='./output',
                      help='Output directory')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                      help='Learning rate')
    
    # Prediction arguments
    parser.add_argument('--model-path', type=str,
                      help='Path to trained model')
    parser.add_argument('--input', type=str,
                      help='Input file for prediction')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(args.output_dir)
    
    try:
        if args.mode == 'train':
            # Create config
            config = ModelConfig(
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate
            )
            
            # Initialize components
            data_processor = DataProcessor(args.attributes, args.concepts, config)
            train_dataset, val_dataset = data_processor.process_data()
            
            trainer = ModelTrainer(config, args.output_dir)
            trainer.train(train_dataset, val_dataset)
            
        else:  # predict mode
            if not args.model_path or not args.input:
                raise ValueError("--model-path and --input required for predict mode")
                
            predictor = PredictionService(args.model_path)
            output_path = os.path.join(args.output_dir, 'predictions.csv')
            
            predictor.predict_csv(args.input, args.concepts, output_path)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()        # Save
