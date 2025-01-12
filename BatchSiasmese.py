import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict

# Setup logging with timestamp
def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

@dataclass
class TrainingConfig:
    model_name: str = 'sentence-transformers/all-mpnet-base-v2'
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 2e-5
    margin: float = 0.5
    hidden_size: int = 256
    dropout_rate: float = 0.3
    num_folds: int = 5
    patience: int = 5
    train_ratio: float = 0.8
    num_workers: int = multiprocessing.cpu_count()

class EnhancedSiameseNetwork(nn.Module):
    def __init__(self, model_name: str, hidden_size: int = 256, dropout_rate: float = 0.3):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        
        # Get encoder output dimension
        encoder_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Additional neural layers
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, text1: List[str], text2: List[str]) -> torch.Tensor:
        # Encode both texts
        embeddings1 = self.encoder.encode(text1, convert_to_tensor=True)
        embeddings2 = self.encoder.encode(text2, convert_to_tensor=True)
        
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Concatenate embeddings
        combined = torch.cat((embeddings1, embeddings2), dim=1)
        
        # Pass through MLP
        similarity = self.mlp(combined)
        return similarity.squeeze()

class SiameseDataset(Dataset):
    def __init__(self, pairs: np.ndarray, labels: np.ndarray):
        self.pairs = pairs
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Tuple[str, str, float]:
        return (self.pairs[idx][0], self.pairs[idx][1], self.labels[idx])

class DataProcessor:
    def __init__(self, attributes_path: str, concepts_path: str, config: TrainingConfig):
        self.attributes_path = Path(attributes_path)
        self.concepts_path = Path(concepts_path)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.attributes_df = None
        self.concepts_df = None
        self.train_dataset = None
        self.test_dataset = None
        
    def process_data(self) -> Tuple[SiameseDataset, SiameseDataset]:
        """Load, process and split data into train/test sets."""
        self.logger.info("Loading and processing data...")
        
        # Load data
        self.attributes_df = pd.read_csv(self.attributes_path)
        self.concepts_df = pd.read_csv(self.concepts_path)
        
        # Create pairs and labels
        pairs = []
        labels = []
        
        # Use ProcessPoolExecutor for parallel pair creation
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            
            for _, attr_row in self.attributes_df.iterrows():
                futures.append(
                    executor.submit(
                        self._create_pairs_for_attribute,
                        attr_row,
                        self.concepts_df
                    )
                )
            
            # Collect results
            for future in tqdm(futures, desc="Creating pairs"):
                pair_results, label_results = future.result()
                pairs.extend(pair_results)
                labels.extend(label_results)
        
        pairs = np.array(pairs)
        labels = np.array(labels)
        
        # Split data
        return self._split_data(pairs, labels)
    
    @staticmethod
    def _create_pairs_for_attribute(attr_row: pd.Series, concepts_df: pd.DataFrame) -> Tuple[List, List]:
        pairs = []
        labels = []
        attribute_text = f"{attr_row['attribute_name']} {attr_row['description']}"
        
        for _, concept_row in concepts_df.iterrows():
            concept_text = f"{concept_row['domain']}-{concept_row['concept']}: {concept_row['concept_definition']}"
            label = 1.0 if (attr_row['domain'] == concept_row['domain'] and 
                           attr_row['concept'] == concept_row['concept']) else 0.0
            pairs.append((attribute_text, concept_text))
            labels.append(label)
            
        return pairs, labels
    
    def _split_data(self, pairs: np.ndarray, labels: np.ndarray) -> Tuple[SiameseDataset, SiameseDataset]:
        """Split data into train and test sets."""
        total_size = len(pairs)
        train_size = int(total_size * self.config.train_ratio)
        test_size = total_size - train_size
        
        train_dataset, test_dataset = random_split(
            SiameseDataset(pairs, labels),
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.logger.info(f"Split data into {train_size} train and {test_size} test samples")
        return train_dataset, test_dataset

class ModelTrainer:
    def __init__(self, config: TrainingConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
    def train(self, train_dataset: Dataset, test_dataset: Dataset) -> None:
        """Train model with k-fold cross validation."""
        self.logger.info(f"Starting training with {self.config.num_folds}-fold cross validation")
        
        kfold = KFold(n_splits=self.config.num_folds, shuffle=True, random_seed=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            self.logger.info(f"Training fold {fold + 1}/{self.config.num_folds}")
            
            # Initialize model and optimizer
            model = EnhancedSiameseNetwork(
                self.config.model_name,
                self.config.hidden_size,
                self.config.dropout_rate
            ).to(self.device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
            criterion = nn.BCEWithLogitsLoss()
            
            # Create data loaders for this fold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                sampler=train_subsampler,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                sampler=val_subsampler,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            # Training loop
            patience_counter = 0
            best_fold_loss = float('inf')
            
            for epoch in range(self.config.num_epochs):
                train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = self._evaluate(model, val_loader, criterion)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Log metrics
                self.metrics['train_loss'].append(train_loss)
                self.metrics['train_acc'].append(train_acc)
                self.metrics['val_loss'].append(val_loss)
                self.metrics['val_acc'].append(val_acc)
                
                self.logger.info(
                    f"Fold {fold + 1}, Epoch {epoch + 1}: "
                    f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
                    f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}"
                )
                
                # Check for improvement
                if val_loss < best_fold_loss:
                    best_fold_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model for this fold
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_model_path = self._save_model(
                            model,
                            f"best_model_fold{fold + 1}"
                        )
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        self.logger.info(f"Early stopping on fold {fold + 1}")
                        break
            
            fold_results.append({
                'fold': fold + 1,
                'best_val_loss': best_fold_loss
            })
            
        # Final evaluation on test set
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Load best model
        best_model = EnhancedSiameseNetwork(
            self.config.model_name,
            self.config.hidden_size,
            self.config.dropout_rate
        ).to(self.device)
        best_model.load_state_dict(torch.load(self.best_model_path))
        
        test_loss, test_acc = self._evaluate(best_model, test_loader, criterion)
        self.logger.info(f"Final Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}")
        
        # Save training results
        self._save_training_results(fold_results, test_loss, test_acc)
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_text1, batch_text2, batch_labels in tqdm(train_loader, desc="Training"):
            batch_labels = batch_labels.float().to(self.device)
            
            # Forward pass
            outputs = model(batch_text1, batch_text2)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += len(batch_labels)
        
        return total_loss / len(train_loader), correct / total
    
    def _evaluate(self, model: nn.Module, data_loader: DataLoader,
                 criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_text1, batch_text2, batch_labels in data_loader:
                batch_labels = batch_labels.float().to(self.device)
                outputs = model(batch_text1, batch_text2)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
        
        return total_loss / len(data_loader), correct / total
    
    def _save_model(self, model: nn.Module, name: str) ->








def _save_model(self, model: nn.Module, name: str) -> str:
        """Save model and return path."""
        save_path = self.output_dir / name
        os.makedirs(save_path, exist_ok=True)
        
        # Save encoder
        model.encoder.save(save_path)
        
        # Save full model state
        model_path = save_path / 'model_state.pt'
        torch.save(model.state_dict(), model_path)
        
        # Save config
        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
            
        return str(save_path)
    
    def _save_training_results(self, fold_results: List[Dict], test_loss: float, test_acc: float) -> None:
        """Save training results and metrics."""
        results = {
            'fold_results': fold_results,
            'test_metrics': {
                'loss': test_loss,
                'accuracy': test_acc
            },
            'training_metrics': dict(self.metrics),
            'config': vars(self.config),
            'best_model_path': self.best_model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved training results to {results_path}")

class PredictionService:
    def __init__(self, model_path: str, config: Optional[TrainingConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        
        # Load config
        config_path = self.model_path / 'config.json'
        if config is None:
            with open(config_path) as f:
                config_dict = json.load(f)
                self.config = TrainingConfig(**config_dict)
        else:
            self.config = config
            
        # Initialize model
        self.model = self._load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def _load_model(self) -> EnhancedSiameseNetwork:
        """Load the trained model."""
        try:
            model = EnhancedSiameseNetwork(
                self.config.model_name,
                self.config.hidden_size,
                self.config.dropout_rate
            )
            
            state_dict_path = self.model_path / 'model_state.pt'
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict)
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    @torch.no_grad()
    def predict_single(self, attribute_name: str, description: str, 
                      concept_texts: List[str], top_k: int = 3) -> List[Dict]:
        """Predict for a single attribute."""
        self.model.eval()
        
        # Prepare input
        attribute_text = f"{attribute_name} {description}"
        attribute_texts = [attribute_text] * len(concept_texts)
        
        # Get predictions
        outputs = self.model(attribute_texts, concept_texts)
        probabilities = torch.sigmoid(outputs)
        
        # Get top-k predictions
        top_k_values, top_k_indices = torch.topk(probabilities, min(top_k, len(concept_texts)))
        
        # Prepare results
        results = []
        for i, (prob, idx) in enumerate(zip(top_k_values, top_k_indices)):
            concept_text = concept_texts[idx]
            domain, concept = concept_text.split(':')[0].strip().split('-')
            
            results.append({
                'rank': i + 1,
                'domain': domain,
                'concept': concept,
                'confidence': float(prob)
            })
            
        return results
    
    def predict_batch(self, df: pd.DataFrame, concept_texts: List[str], 
                     output_path: Optional[str] = None) -> pd.DataFrame:
        """Predict for a batch of attributes."""
        self.logger.info(f"Processing {len(df)} attributes...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            
            for _, row in df.iterrows():
                futures.append(
                    executor.submit(
                        self.predict_single,
                        row['attribute_name'],
                        row['description'],
                        concept_texts
                    )
                )
            
            for future in tqdm(futures, desc="Processing predictions"):
                predictions = future.result()
                if predictions:
                    top_pred = predictions[0]  # Get top prediction
                    results.append({
                        'predicted_domain': top_pred['domain'],
                        'predicted_concept': top_pred['concept'],
                        'confidence': top_pred['confidence']
                    })
                else:
                    results.append({
                        'predicted_domain': None,
                        'predicted_concept': None,
                        'confidence': 0.0
                    })
        
        # Create results DataFrame
        results_df = pd.concat([
            df,
            pd.DataFrame(results)
        ], axis=1)
        
        # Save if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved predictions to {output_path}")
            
        return results_df

def main():
    parser = argparse.ArgumentParser(description='Enhanced Siamese Network for Text Matching')
    
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
    parser.add_argument('--num-folds', type=int, default=5,
                      help='Number of CV folds')
    
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
            config = TrainingConfig(
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                num_folds=args.num_folds
            )
            
            # Initialize components
            data_processor = DataProcessor(args.attributes, args.concepts, config)
            train_dataset, test_dataset = data_processor.process_data()
            
            trainer = ModelTrainer(config, args.output_dir)
            trainer.train(train_dataset, test_dataset)
            
        else:  # predict mode
            if not args.model_path or not args.input:
                raise ValueError("--model-path and --input required for predict mode")
                
            # Load concepts for prediction
            concepts_df = pd.read_csv(args.concepts)
            concept_texts = [
                f"{row['domain']}-{row['concept']}: {row['concept_definition']}"
                for _, row in concepts_df.iterrows()
            ]
            
            # Initialize prediction service
            predictor = PredictionService(args.model_path)
            
            # Read input and make predictions
            input_df = pd.read_csv(args.input)
            output_path = os.path.join(args.output_dir, 'predictions.csv')
            predictor.predict_batch(input_df, concept_texts, output_path)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
