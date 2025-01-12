import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import json
import sys
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    domain: str
    concept: str
    confidence: float

@dataclass
class TrainingStats:
    """Container for training statistics."""
    epoch: int
    loss: float
    num_samples: int
    accuracy: float
    learning_rate: float
    batch_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'loss': self.loss,
            'num_samples': self.num_samples,
            'accuracy': self.accuracy,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'timestamp': datetime.now().isoformat()
        }

class MetricsTracker:
    """Track and log training/prediction metrics."""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics: Dict[str, List[float]] = {}
        self.epoch_metrics: Dict[str, float] = {}
        self.best_metric = float('-inf')
        
    def add_metric(self, name: str, value: float, epoch: Optional[int] = None) -> bool:
        """
        Add metric and return True if it's the best metric so far.
        Use accuracy as the primary metric for model selection.
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
        if epoch is not None:
            self.epoch_metrics[f"{name}_epoch_{epoch}"] = value
        
        # Check if this is the best accuracy
        is_best = False
        if name == 'accuracy' and value > self.best_metric:
            self.best_metric = value
            is_best = True
        
        return is_best
            
    def save_metrics(self, filename: str = "metrics.json") -> None:
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump({
                    'metrics': self.metrics,
                    'epoch_metrics': self.epoch_metrics,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Saved metrics to {output_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

class ParallelSiameseDataset(Dataset):
    """Optimized Dataset for Siamese Network training with parallel processing."""
    def __init__(self, 
                attributes_df: pd.DataFrame,
                concepts_df: pd.DataFrame,
                batch_size: int = 32,
                num_workers: int = 4):
        self.attributes_df = attributes_df
        self.concepts_df = concepts_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Parallel pair creation
        self.pairs = []
        self.labels = []
        self._create_pairs_parallel()
        
    def _create_pairs_parallel(self):
        """Create pairs using multiprocessing."""
        def create_pairs_chunk(attr_chunk, concepts_df):
            chunk_pairs = []
            chunk_labels = []
            for _, attr_row in attr_chunk.iterrows():
                attribute_text = f"{attr_row['attribute_name']} {attr_row['description']}"
                
                for _, concept_row in concepts_df.iterrows():
                    concept_text = f"{concept_row['domain']}-{concept_row['concept']}: {concept_row['concept_definition']}"
                    
                    label = 1.0 if (attr_row['domain'] == concept_row['domain'] and 
                                   attr_row['concept'] == concept_row['concept']) else 0.0
                    
                    chunk_pairs.append((attribute_text, concept_text))
                    chunk_labels.append(label)
            
            return chunk_pairs, chunk_labels

        # Split attributes into chunks
        chunks = np.array_split(self.attributes_df, self.num_workers)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(create_pairs_chunk, chunk, self.concepts_df) 
                for chunk in chunks
            ]
            
            for future in as_completed(futures):
                pairs, labels = future.result()
                self.pairs.extend(pairs)
                self.labels.extend(labels)
        
        # Convert to numpy arrays
        self.pairs = np.array(self.pairs)
        self.labels = np.array(self.labels)
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Tuple[str, str, float]:
        return (self.pairs[idx][0], self.pairs[idx][1], self.labels[idx])

class SiameseNetwork(nn.Module):
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.similarity = nn.CosineSimilarity(dim=1)
        
    def forward(self, text1: List[str], text2: List[str]) -> torch.Tensor:
        # Encode both texts with parallel processing
        embeddings1 = self.encoder.encode(text1, convert_to_tensor=True)
        embeddings2 = self.encoder.encode(text2, convert_to_tensor=True)
        
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Calculate similarity
        similarity = self.similarity(embeddings1, embeddings2)
        
        return similarity

class AdaptiveSiameseLoss(nn.Module):
    """Adaptive loss with margin and dynamic weight"""
    def __init__(self, initial_margin: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(initial_margin))
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Adaptive binary cross-entropy with dynamically adjusted margin
        loss = F.binary_cross_entropy_with_logits(
            predictions, 
            targets, 
            reduction=self.reduction
        )
        
        return loss

class DataProcessor:
    def __init__(self, attributes_path: str, concepts_path: str):
        self.attributes_path = Path(attributes_path)
        self.concepts_path = Path(concepts_path)
        self.logger = logging.getLogger(__name__)
        
        # Parallel data loading
        self.attributes_df = None
        self.concepts_df = None
        self.domain_concepts = {}
        self.load_data_parallel()

    def load_data_parallel(self) -> None:
        """Parallel data loading and validation"""
        try:
            # Use ThreadPoolExecutor for parallel loading
            with ThreadPoolExecutor(max_workers=2) as executor:
                attributes_future = executor.submit(pd.read_csv, self.attributes_path)
                concepts_future = executor.submit(pd.read_csv, self.concepts_path)
                
                self.attributes_df = attributes_future.result()
                self.concepts_df = concepts_future.result()
            
            # Validate columns
            required_attr_cols = ['attribute_name', 'description', 'domain', 'concept']
            required_concept_cols = ['domain', 'concept', 'concept_definition']
            
            self._validate_columns(self.attributes_df, required_attr_cols, 'attributes')
            self._validate_columns(self.concepts_df, required_concept_cols, 'concepts')

            # Parallel domain-concept cache building
            self._build_domain_concepts_cache()
            
            self.logger.info(f"Loaded {len(self.attributes_df)} attributes and {len(self.concepts_df)} concept definitions")
        
        except Exception as e:
            self.logger.error(f"Parallel data loading error: {str(e)}")
            raise

    def _validate_columns(self, df: pd.DataFrame, required_cols: List[str], df_name: str) -> None:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {df_name} CSV: {missing_cols}")

    def _build_domain_concepts_cache(self) -> None:
        # Parallel domain-concept cache building
        with ThreadPoolExecutor() as executor:
            futures = []
            for _, row in self.concepts_df.iterrows():
                futures.append(executor.submit(self._add_to_domain_concepts, row))
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()
        
        self.logger.info(f"Found {len(self.domain_concepts)} domains with unique concepts")

    def _add_to_domain_concepts(self, row):
        if row['domain'] not in self.domain_concepts:
            self.domain_concepts[row['domain']] = set()
        self.domain_concepts[row['domain']].add(row['concept'])

    def get_all_concept_texts(self) -> List[str]:
        """Get list of all concept texts."""
        return [
            f"{row['domain']}-{row['concept']}: {row['concept_definition']}"
            for _, row in self.concepts_df.iterrows()
        ]

class ModelTrainer:
    def __init__(self, 
                model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                batch_size: int = 64,
                num_epochs: int = 15,
                learning_rate: float = 1e-4,
                weight_decay: float = 1e-5,
                output_dir: str = './model_output'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = SiameseNetwork(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def train(self, data_processor: DataProcessor) -> None:
        try:
            self.logger.info("Preparing training data...")
            
            # Initialize dataset with parallel processing
            dataset = ParallelSiameseDataset(
                data_processor.attributes_df,
                data_processor.concepts_df,
                self.batch_size
            )
            
            # Use DistributedSampler for better data distribution
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True
            )
            
            # Initialize loss and optimizer with adaptive components
            criterion = AdaptiveSiameseLoss()
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                factor=0.5, 
                patience=3, 
                verbose=True
            )
            
            # Initialize metrics tracker
            metrics_tracker = MetricsTracker(self.output_dir)
            
            self.logger.info(f"Starting training for {self.num_epochs} epochs...")
            
            best_model_path = os.path.join(self.output_dir, 'best_model')
            
            for epoch in range(self.num_epochs):
                total_loss = 0
                correct_predictions = 0
                total_predictions = 0
                
                # Set model to training mode
                self.model.train()
                
                with tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}") as pbar:
                    for batch_text1, batch_text2, batch_labels in pbar:
                        # Move data to device
                        batch_labels = batch_labels.float().to(self.device)
                        
                        # Forward pass with parallel processing
                        similarities = self.model(batch_text1, batch_text2)
                        
                        # Calculate loss
                        loss = criterion(similarities, batch_labels)
                        
                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Update metrics
                        total_loss += loss.item()
                        predictions = (similarities > 0.5).float()
                        correct_predictions += (predictions == batch_labels).sum().item()
                        total_predictions += len(batch_labels)
                        
                        pbar.set_postfix({
                            'loss': total_loss / (pbar.n + 1),
                            'acc': correct_predictions / total_predictions
                        })
                
                # Calculate epoch metrics
                avg_loss = total_loss / len(dataloader)
                accuracy = correct_predictions / total_predictions
                
                # Add metrics and check for best model
                is_best = metrics_tracker.add_metric('loss', avg_loss, epoch)
                metrics_tracker.add_metric('accuracy', accuracy, epoch)
                
                # Learning rate scheduling
                scheduler.step(accuracy)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}"
                )
                
                # Save best model
                if is_best:
                    os.makedirs(best_model_path, exist_ok=True)
