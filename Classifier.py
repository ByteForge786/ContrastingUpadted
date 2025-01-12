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
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import yaml
import json
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from functools import lru_cache
import multiprocessing
from collections import defaultdict

# Setup logging with process info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Optimize torch operations
torch.backends.cudnn.benchmark = True

@dataclass
class PredictionResult:
    domain: str
    concept: str
    confidence: float

@dataclass
class SamplingResult:
    positive_pairs: List[Tuple[str, str]]
    negative_pairs: List[Tuple[str, str, float]]
    stats: Dict[str, int]

@dataclass
class TrainingBatch:
    anchors: List[str]
    positives: List[str]
    negatives: List[str]
    negative_weights: List[float]

    def to_device(self, device: torch.device):
        return {
            'anchors': [a.to(device) for a in self.anchors],
            'positives': [p.to(device) for p in self.positives],
            'negatives': [n.to(device) for n in self.negatives],
            'negative_weights': torch.tensor(self.negative_weights, device=device)
        }

@dataclass
class TrainingStats:
    epoch: int
    loss: float
    val_loss: float
    num_samples: int
    positive_pairs: int
    hard_negatives: int
    medium_negatives: int
    learning_rate: float
    batch_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'loss': self.loss,
            'val_loss': self.val_loss,
            'num_samples': self.num_samples,
            'positive_pairs': self.positive_pairs,
            'hard_negatives': self.hard_negatives,
            'medium_negatives': self.medium_negatives,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'timestamp': datetime.now().isoformat()
        }

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_path = None
        
    def __call__(self, val_loss: float, model: nn.Module, epoch: int, save_path: str) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, save_path, epoch)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, save_path, epoch)
            self.counter = 0
            
        return self.early_stop
        
    def save_checkpoint(self, model: nn.Module, save_path: str, epoch: int):
        path = f"{save_path}/best_model_epoch_{epoch}"
        model.save(path)
        self.best_model_path = path

class MetricsTracker:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.epoch_metrics: Dict[str, float] = {}
        
    def add_metric(self, name: str, value: float, epoch: Optional[int] = None) -> None:
        self.metrics[name].append(value)
        
        if epoch is not None:
            self.epoch_metrics[f"{name}_epoch_{epoch}"] = value
            
    def save_metrics(self, filename: str = "metrics.json") -> None:
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump({
                    'metrics': dict(self.metrics),
                    'epoch_metrics': self.epoch_metrics,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Saved metrics to {output_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

class DataProcessor:
    def __init__(self, attributes_path: str, concepts_path: str, test_size: float = 0.2, random_state: int = 42):
        self.attributes_path = Path(attributes_path)
        self.concepts_path = Path(concepts_path)
        self.test_size = test_size
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Use more efficient data structures
        self.attributes_df = None
        self.concepts_df = None
        self.domain_concepts: Dict[str, Set[str]] = defaultdict(set)
        
        # Cache for concept definitions
        self._concept_def_cache: Dict[Tuple[str, str], str] = {}
        
        # Split indices
        self.train_indices = None
        self.val_indices = None
        
        self.load_data()

    def load_data(self) -> None:
        try:
            # Load data in parallel
            with ThreadPoolExecutor() as executor:
                attr_future = executor.submit(pd.read_csv, self.attributes_path)
                concept_future = executor.submit(pd.read_csv, self.concepts_path)
                
                self.attributes_df = attr_future.result()
                self.concepts_df = concept_future.result()

            # Validate columns
            required_attr_cols = ['attribute_name', 'description', 'domain', 'concept']
            required_concept_cols = ['domain', 'concept', 'concept_definition']
            
            self._validate_columns(self.attributes_df, required_attr_cols, 'attributes')
            self._validate_columns(self.concepts_df, required_concept_cols, 'concepts')

            # Optimize dataframe
            self._optimize_dataframes()

            # Create train/validation split
            all_indices = np.arange(len(self.attributes_df))
            self.train_indices, self.val_indices = train_test_split(
                all_indices,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.attributes_df['domain'].values
            )

            # Build domain-concept cache efficiently
            self._build_domain_concepts_cache()
            
            self.logger.info(
                f"Loaded {len(self.attributes_df)} attributes and {len(self.concepts_df)} concept definitions. "
                f"Split into {len(self.train_indices)} train and {len(self.val_indices)} validation samples"
            )
        
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _optimize_dataframes(self):
        """Optimize memory usage of dataframes."""
        for df in [self.attributes_df, self.concepts_df]:
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = pd.Categorical(df[col])

    def _validate_columns(self, df: pd.DataFrame, required_cols: List[str], df_name: str) -> None:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {df_name} CSV: {missing_cols}")

    def _build_domain_concepts_cache(self) -> None:
        """Build domain-concepts cache using vectorized operations."""
        domain_concept_pairs = self.concepts_df[['domain', 'concept']].drop_duplicates()
        for _, row in domain_concept_pairs.iterrows():
            self.domain_concepts[row['domain']].add(row['concept'])
            
        # Pre-cache concept definitions
        for _, row in self.concepts_df.iterrows():
            self._concept_def_cache[(row['domain'], row['concept'])] = row['concept_definition']
        
        self.logger.info(f"Found {len(self.domain_concepts)} domains with unique concepts")

    @lru_cache(maxsize=1024)
    def get_attribute_text(self, attribute_name: str, description: str) -> str:
        """Cached attribute text generation."""
        return f"{attribute_name} {description}"

    def get_concept_definition(self, domain: str, concept: str) -> str:
        """Get concept definition from cache."""
        cache_key = (domain, concept)
        if cache_key not in self._concept_def_cache:
            raise ValueError(f"No definition found for domain={domain}, concept={concept}")
        return self._concept_def_cache[cache_key]

    def get_domains(self) -> List[str]:
        return list(self.domain_concepts.keys())

    def get_concepts_for_domain(self, domain: str) -> List[str]:
        return list(self.domain_concepts.get(domain, set()))

    def get_train_attributes(self) -> pd.DataFrame:
        return self.attributes_df.iloc[self.train_indices]

    def get_val_attributes(self) -> pd.DataFrame:
        return self.attributes_df.iloc[self.val_indices]

class BatchedPairSampler:
    def __init__(self, data_processor: DataProcessor, batch_size: int = 32):
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentence transformer for similarity computation
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compute_batch_similarities(self, texts1: List[str], texts2: List[str]) -> torch.Tensor:
        """Compute similarities for batches of texts."""
        with torch.no_grad():
            embeddings1 = self.model.encode(texts1, convert_to_tensor=True, batch_size=self.batch_size)
            embeddings2 = self.model.encode(texts2, convert_to_tensor=True, batch_size=self.batch_size)
            
            # Normalize embeddings
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)
            
            # Compute similarities
            similarities = torch.mm(embeddings1, embeddings2.t())
            
        return similarities

    def sample_pairs(self, domain: str, concept: str, is_training: bool = True) -> SamplingResult:
        try:
            # Get attributes based on split
            attributes_df = (
                self.data_processor.get_train_attributes() if is_training 
                else self.data_processor.get_val_attributes()
            )
            
            # Get all positive attributes efficiently
            mask = (attributes_df['domain'] == domain) & (attributes_df['concept'] == concept)
            positive_attrs = attributes_df[mask]
            
            if len(positive_attrs) == 0:
                raise ValueError(f"No attributes found for domain={domain}, concept={concept}")

            concept_def = self.data_processor.get_concept_definition(domain, concept)
            
            # Create positive pairs efficiently
            positive_pairs = [
                (self.data_processor.get_attribute_text(row['attribute_name'], row['description']), concept_def)
                for _, row in positive_attrs.iterrows()
            ]
            
            n_required_negatives = len(positive_pairs)
            
            # Sample negatives in parallel
            with ThreadPoolExecutor() as executor:
                hard_future = executor.submit(
                    self._sample_hard_negatives,
                    domain, concept, positive_attrs,
                    int(0.4 * n_required_negatives), attributes_df
                )
                
                medium_future = executor.submit(
                    self._sample_medium_negatives,
                    domain, concept,
                    n_required_negatives - int(0.4 * n_required_negatives),
                    attributes_df
                )
                
                hard_negatives = hard_future.result()
                medium_negatives = medium_future.result()
            
            # Combine all negatives
            negative_pairs = hard_negatives + medium_negatives
            
            stats = {
                'n_positives': len(positive_pairs),
                'n_hard_negatives': len(hard_negatives),
                'n_medium_negatives': len(medium_negatives)
            }
            
            return SamplingResult(positive_pairs, negative_pairs, stats)
            
        except Exception as e:
            self.logger.error(f"Error sampling pairs for {domain}-{concept}: {str(e)}")
            raise

    def _sample_hard_negatives(self, domain: str, concept: str,
                             positive_attrs: pd.DataFrame, n_required: int,
                             attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample hard negatives using batched operations."""
        hard_negatives = []
        other_concepts = [c for c in self.data_processor.get_concepts_for_domain(domain) 
                         if c != concept]
        
        if not other_concepts:
            return hard_negatives
            
        # Get potential negative attributes efficiently
        mask = (attributes_df['domain'] == domain) & (attributes_df['concept'].isin(other_concepts))
        potential_negatives = attributes_df[mask]
        
        if len(potential_negatives) == 0:
            return hard_negatives
            
        # Prepare texts for batch processing
        pos_texts = [
            row['description'] for _, row in positive_attrs.iterrows()
        ]
        neg_texts = [
            row['description'] for _, row in potential_negatives.iterrows()
        ]
        
        # Compute similarities in batches
        batch_size = min(128, len(pos_texts))
        similarities = []
        
        for i in range(0, len(pos_texts), batch_size):
            pos_batch = pos_texts[i:i+batch_size]
            sim_batch = self.compute_batch_similarities(pos_batch, neg_texts)
            similarities.append(sim_batch)
            
        # Combine similarities
        similarities = torch.cat(similarities, dim=0)
        
        # Find high similarity pairs
        high_sim_indices = torch.nonzero(similarities > 0.8)
        
        # Create hard negatives
        for pos_idx, neg_idx in high_sim_indices[:n_required]:
            neg_row = potential_negatives.iloc[neg_idx]
            neg_text = self.data_processor.get_attribute_text(
                neg_row['attribute_name'],
                neg_row['description']
            )
            neg_def = self.data_processor.get_concept_definition(domain, neg_row['concept'])
            hard_negatives.append((neg_text, neg_def, 1.5))
            
        return hard_negatives

    def _sample_medium_negatives(self, domain: str, concept: str,
                               n_required: int, attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample medium negatives efficiently."""
        medium_negatives = []
        other_domains = [d for d in self.data_processor.get_domains() if d != domain]
        
        if not other_domains:
            return medium_negatives
            
        # Batch sample from other domains
        sample_domains = np.random.choice(other_domains, n_required, replace=True)
        domain_batches = defaultdict(list)
        
        # Group by domain for efficient processing
        for d in sample_domains:
            domain_batches[d].append(d)
            
        # Process each domain batch
        for d, batch in domain_batches.items():
            concepts = self.data_processor.get_concepts_for_domain(d)
            if not concepts:
                continue
                
            # Sample concepts for entire batch
            batch_concepts = np.random.choice(list(concepts), len(batch))
            
            # Get attributes efficiently
            mask = (attributes_df['domain'] == d) & (attributes_df['concept'].isin(set(batch_concepts)))
            neg_attrs = attributes_df[mask]
            
            if len(neg_attrs) == 0:
                continue
                
            # Sample attributes for batch
            sampled_indices = np.random.randint(0, len(neg_attrs), len(batch))
            sampled_attrs = neg_attrs.iloc[sampled_indices]
            
            for _, row in sampled_attrs.iterrows():
                neg_text = self.data_processor.get_attribute_text(
                    row['attribute_name'],
                    row['description']
                )
                neg_def = self.data_processor.get_concept_definition(d, row['concept'])
                medium_negatives.append((neg_text, neg_def, 1.0))
                
                if len(medium_negatives) >= n_required:
                    return medium_negatives[:n_required]
                    
        return medium_negatives

class OptimizedDataset(Dataset):
    def __init__(self, sampling_results: List[TrainingBatch], device: torch.device):
        self.batches = sampling_results
        self.device = device
        
    def __len__(self) -> int:
        return len(self.batches)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = self.batches[idx]
        return batch.to_device(self.device)

class CustomInfoNCELoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature: float = 0.07):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.cache = {}
        
    @torch.no_grad()
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings with caching."""
        uncached_texts = [text for text in texts if text not in self.cache]
        
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                convert_to_tensor=True,
                batch_size=32
            )
            
            # Update cache
            for text, emb in zip(uncached_texts, new_embeddings):
                self.cache[text] = emb
                
        # Get all embeddings from cache
        return torch.stack([self.cache[text] for text in texts])
        
    def forward(self, anchors: List[str], positives: List[str],
                negatives: List[str], negative_weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted InfoNCE loss with batched operations."""
        # Get embeddings using cache
        anchor_embeddings = self._get_embeddings(anchors)
        positive_embeddings = self._get_embeddings(positives)
        negative_embeddings = self._get_embeddings(negatives)
        
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
        
        # Compute similarities in parallel
        positive_similarities = torch.sum(
            anchor_embeddings * positive_embeddings, dim=-1
        ) / self.temperature
        
        negative_similarities = torch.matmul(
            anchor_embeddings, negative_embeddings.transpose(0, 1)
        ) / self.temperature
        
        # Apply weights to negative similarities
        negative_similarities = negative_similarities * negative_weights.unsqueeze(0)
        
        # Compute loss
        logits = torch.cat([positive_similarities.unsqueeze(-1), negative_similarities], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)

class OptimizedModelTrainer:
    def __init__(self,
                model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                batch_size: int = 32,
                num_epochs: int = 10,
                learning_rate: float = 2e-5,
                temperature: float = 0.07,
                output_dir: str = './model_output',
                patience: int = 5,
                num_workers: int = multiprocessing.cpu_count()):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.output_dir = output_dir
        self.patience = patience
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        
        # Initialize model with optimizations
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def prepare_batches(self, data_processor: DataProcessor,
                       sampler: BatchedPairSampler,
                       is_training: bool = True) -> List[TrainingBatch]:
        """Prepare batches in parallel."""
        batches = []
        
        # Create tasks for parallel processing
        tasks = [
            (domain, concept)
            for domain in data_processor.get_domains()
            for concept in data_processor.get_concepts_for_domain(domain)
        ]
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_task = {
                executor.submit(sampler.sample_pairs, domain, concept, is_training): (domain, concept)
                for domain, concept in tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks),
                             desc="Preparing batches"):
                domain, concept = future_to_task[future]
                try:
                    result = future.result()
                    batch = TrainingBatch(
                        anchors=[result.positive_pairs[0][1]] * len(result.positive_pairs),
                        positives=[pair[0] for pair in result.positive_pairs],
                        negatives=[pair[0] for pair in result.negative_pairs],
                        negative_weights=[pair[2] for pair in result.negative_pairs]
                    )
                    batches.append(batch)
                except Exception as e:
                    self.logger.error(f"Error preparing batch for {domain}-{concept}: {str(e)}")
                    
        return batches

    def train_epoch(self, dataloader: DataLoader,
                   loss_fn: CustomInfoNCELoss,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> float:
        """Train for one epoch with optimizations."""
        total_loss = 0
        batch_count = 0
        
        # Enable automatic mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        with tqdm(dataloader, desc="Training") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    loss = loss_fn(
                        batch['anchors'],
                        batch['positives'],
                        batch['negatives'],
                        batch['negative_weights']
                    )
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                if scheduler:
                    scheduler.step()
                
                total_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({'loss': total_loss / batch_count})
                
        return total_loss / batch_count

    def validate(self, dataloader: DataLoader, loss_fn: CustomInfoNCELoss) -> float:
        """Run validation with optimizations."""
        total_loss = 0
        batch_count = 0
        
        self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            with tqdm(dataloader, desc="Validation") as pbar:
                for batch in pbar:
                    loss = loss_fn(
                        batch['anchors'],
                        batch['positives'],
                        batch['negatives'],
                        batch['negative_weights']
                    )
                    
                    total_loss += loss.item()
                    batch_count += 1
                    pbar.set_postfix({'val_loss': total_loss / batch_count})
                    
        self.model.train()
        return total_loss / batch_count

    def train(self, data_processor: DataProcessor, sampler: BatchedPairSampler) -> None:
        """Train the model with optimizations."""
        try:
            self.logger.info("Preparing training data...")
            
            # Initialize trackers
            metrics_tracker = MetricsTracker(self.output_dir)
            early_stopping = EarlyStopping(patience=self.patience)
            
            # Prepare batches in parallel
            train_batches = self.prepare_batches(data_processor, sampler, is_training=True)
            val_batches = self.prepare_batches(data_processor, sampler, is_training=False)
            
            if not train_batches or not val_batches:
                raise ValueError("No training or validation batches could be prepared")
                
            # Create datasets with device placement
            train_dataset = OptimizedDataset(train_batches, self.device)
            val_dataset = OptimizedDataset(val_batches, self.device)
            
            # Create dataloaders with optimizations
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            # Initialize loss and optimizer
            loss_fn = CustomInfoNCELoss(self.model, self.temperature)
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
            
            # Add learning rate scheduler
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=self.num_epochs,
                steps_per_epoch=len(train_dataloader)
            )
            
            self.logger.info(f"Starting training for {self.num_epochs} epochs...")
            
            for epoch in range(self.num_epochs):
                # Training phase
                self.model.train()
                train_loss = self.train_epoch(train_dataloader, loss_fn, optimizer, scheduler)
                
                # Validation phase
                val_loss = self.validate(val_dataloader, loss_fn)
                
                metrics_tracker.add_metric('train_loss', train_loss, epoch)
                metrics_tracker.add_metric('val_loss', val_loss, epoch)
                
                # Create training stats
                stats = TrainingStats(
                    epoch=epoch,
                    loss=train_loss,
                    val_loss=val_loss,
                    num_samples=len(train_dataset),
                    positive_pairs=sum(len(b.positives) for b in train_batches),
                    hard_negatives=sum(sum(1 for w in b.negative_weights if w > 1.0) for b in train_batches),
                    medium_negatives=sum(sum(1 for w in b.negative_weights if w == 1.0) for b in train_batches),
                    learning_rate=scheduler.get_last_lr()[0],
                    batch_size=self.batch_size
                )
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}, "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
                
                # Early stopping check
                if early_stopping(val_loss, self.model, epoch, self.output_dir):
                    self.logger.info("Early stopping triggered")
                    break
                
                # Regular checkpoint saving
                if (epoch + 1) % 5 == 0:
                    save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
                    self.model.save(save_path)
                    self.logger.info(f"Saved checkpoint to {save_path}")
            
            # Save final metrics
            metrics_tracker.save_metrics()
            
            # Load best model if exists
            if early_stopping.best_model_path}")
            else:
                # Save final model if no best model
                final_path = os.path.join(self.output_dir, "final-model")
                self.model.save(final_path)
                self.logger.info(f"Saved final model to {final_path}")
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

class OptimizedPredictor:
    def __init__(self, 
                model_path: str,
                data_processor: DataProcessor,
                batch_size: int = 32,
                top_k: int = 3,
                num_workers: int = multiprocessing.cpu_count()):
        self.model_path = model_path
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.top_k = top_k
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        
        # Load model with optimizations
        try:
            self.model = SentenceTransformer(model_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            if torch.cuda.is_available():
                self.model = torch.nn.DataParallel(self.model)
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
        # Cache concept definitions with parallel processing
        self.concept_definitions = self._cache_concept_definitions()
        # Pre-compute concept embeddings
        self.concept_embeddings = self._precompute_concept_embeddings()
        
    def _cache_concept_definitions(self) -> Dict[Tuple[str, str], str]:
        """Cache all domain-concept definitions in parallel."""
        definitions = {}
        tasks = [
            (domain, concept)
            for domain in self.data_processor.get_domains()
            for concept in self.data_processor.get_concepts_for_domain(domain)
        ]
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.data_processor.get_concept_definition,
                    domain,
                    concept
                ): (domain, concept)
                for domain, concept in tasks
            }
            
            for future in as_completed(future_to_task):
                domain, concept = future_to_task[future]
                try:
                    definition = future.result()
                    definitions[(domain, concept)] = definition
                except Exception as e:
                    self.logger.warning(f"Could not cache definition for {domain}-{concept}: {str(e)}")
                    
        return definitions

    @torch.no_grad()
    def _precompute_concept_embeddings(self) -> Dict[Tuple[str, str], torch.Tensor]:
        """Pre-compute embeddings for all concept definitions."""
        embeddings = {}
        
        # Prepare batches of definitions
        definitions_list = []
        keys_list = []
        
        for (domain, concept), definition in self.concept_definitions.items():
            definitions_list.append(definition)
            keys_list.append((domain, concept))
            
        # Compute embeddings in batches
        for i in range(0, len(definitions_list), self.batch_size):
            batch_definitions = definitions_list[i:i + self.batch_size]
            batch_keys = keys_list[i:i + self.batch_size]
            
            batch_embeddings = self.model.encode(
                batch_definitions,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Normalize embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            # Store in dictionary
            for key, embedding in zip(batch_keys, batch_embeddings):
                embeddings[key] = embedding
                
        return embeddings

    @torch.no_grad()
    def predict_single(self, attribute_name: str, description: str) -> List[PredictionResult]:
        """Predict domain-concept for a single attribute with optimizations."""
        try:
            # Combine attribute text
            attribute_text = f"{attribute_name} {description}"
            
            # Encode attribute text
            attribute_embedding = self.model.encode(
                attribute_text,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Normalize attribute embedding
            attribute_embedding = F.normalize(attribute_embedding, p=2, dim=0)
            
            # Compute similarities with all concept embeddings efficiently
            results = []
            for (domain, concept), concept_embedding in self.concept_embeddings.items():
                similarity = torch.dot(attribute_embedding, concept_embedding).item()
                results.append(PredictionResult(domain, concept, similarity))
            
            # Sort by confidence and return top-k
            results.sort(key=lambda x: x.confidence, reverse=True)
            return results[:self.top_k]
            
        except Exception as e:
            self.logger.error(f"Error in single prediction: {str(e)}")
            raise

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict domain-concepts for a batch of attributes with parallel processing."""
        try:
            # Validate input
            required_cols = ['attribute_name', 'description']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            results = []
            
            # Process in batches with parallel execution
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                for i in range(0, len(df), self.batch_size):
                    batch_df = df.iloc[i:i+self.batch_size]
                    
                    for _, row in batch_df.iterrows():
                        futures.append(
                            executor.submit(
                                self.predict_single,
                                row['attribute_name'],
                                row['description']
                            )
                        )
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing predictions"):
                    try:
                        predictions = future.result()
                        if predictions:
                            top_pred = predictions[0]
                            results.append({
                                'predicted_domain': top_pred.domain,
                                'predicted_concept': top_pred.concept,
                                'confidence': top_pred.confidence
                            })
                        else:
                            results.append({
                                'predicted_domain': None,
                                'predicted_concept': None,
                                'confidence': 0.0
                            })
                    except Exception as e:
                        self.logger.error(f"Error processing prediction: {str(e)}")
                        results.append({
                            'predicted_domain': None,
                            'predicted_concept': None,
                            'confidence': 0.0
                        })
            
            # Add predictions to DataFrame efficiently
            result_df = pd.DataFrame(results)
            return pd.concat([df.reset_index(drop=True), result_df], axis=1)
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise

    def predict_csv(self, input_path: str, output_path: str) -> None:
        """Predict domain-concepts for attributes in a CSV file with optimizations."""
        try:
            # Read input CSV efficiently
            self.logger.info(f"Reading input CSV from {input_path}")
            df = pd.read_csv(input_path)
            
            # Make predictions
            self.logger.info("Making predictions...")
            result_df = self.predict_batch(df)
            
            # Save results efficiently
            self.logger.info(f"Saving predictions to {output_path}")
            result_df.to_csv(output_path, index=False)
            
            self.logger.info("Prediction complete!")
            
        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            raise

def setup_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> Path:
    """Create and setup experiment directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp
        
    experiment_dir = Path(base_dir) / dir_name
    
    # Create required subdirectories
    subdirs = ['models', 'logs', 'metrics', 'predictions', 'cache']
    for subdir in subdirs:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
        
    return experiment_dir

def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration file efficiently."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Required configuration sections
        required_sections = ['data', 'model', 'training', 'prediction', 'output_dir']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")
            
        # Validate data paths exist
        data_paths = [
            Path(config['data']['attributes_path']),
            Path(config['data']['concepts_path'])
        ]
        
        # Check paths in parallel
        with ThreadPoolExecutor() as executor:
            path_exists = list(executor.map(lambda p: p.exists(), data_paths))
            
        missing_paths = [str(path) for path, exists in zip(data_paths, path_exists) if not exists]
        if missing_paths:
            raise FileNotFoundError(f"Data files not found: {missing_paths}")
                
        return config
        
    except Exception as e:
        logger.error(f"Error in configuration: {str(e)}")
        raise

def predict_interactive(predictor: OptimizedPredictor):
    """Interactive prediction mode with optimizations."""
    logger = logging.getLogger(__name__)
    logger.info("Starting interactive prediction mode. Type 'quit' to exit.")
    
    while True:
        try:
            attribute_name = input("\nEnter attribute name (or 'quit' to exit): ").strip()
            if attribute_name.lower() == 'quit':
                break
                
            description = input("Enter description: ").strip()
            
            predictions = predictor.predict_single(attribute_name, description)
            
            print("\nPredictions:")
            print("-" * 50)
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. Domain: {pred.domain}")
                print(f"   Concept: {pred.concept}")
                print(f"   Confidence: {pred.confidence:.4f}")
                print("-" * 50)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            print("An error occurred. Please try again.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or predict with optimized sentence transformer model')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--mode', choices=['train', 'predict', 'interactive'], required=True,
                      help='Operation mode: train, predict, or interactive')
    parser.add_argument('--attributes', type=str, required=True,
                      help='Path to attributes CSV file')
    parser.add_argument('--concepts', type=str, required=True,
                      help='Path to concepts CSV file')
    parser.add_argument('--output-dir', type=str, default='./output',
                      help='Output directory for model and predictions')
    parser.add_argument('--model-path', type=str,
                      help='Path to saved model (required for predict mode)')
    parser.add_argument('--input', type=str,
                      help='Input CSV file for prediction')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training and prediction')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Fraction of data to use for validation')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=multiprocessing.cpu_count(),
                      help='Number of worker processes for parallel processing')
    
    args = parser.parse_args()
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        
        # Initialize data processor with optimizations
        data_processor = DataProcessor(
            args.attributes, 
            args.concepts,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        if args.mode == 'train':
            # Training mode
            sampler = BatchedPairSampler(data_processor, args.batch_size)
            trainer = OptimizedModelTrainer(
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                output_dir=args.output_dir,
                num_workers=args.num_workers
            )
            
            trainer.train(data_processor, sampler)
            
        elif args.mode == 'predict':
            # Prediction mode
            if not args.model_path:
                raise ValueError("--model-path required for predict mode")
            if not args.input:
                raise ValueError("--input required for predict mode")
                
            predictor = OptimizedPredictor(
                model_path=args.model_path,
                data_processor=data_processor,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            output_path = os.path.join(args.output_dir, 'predictions.csv')
            predictor.predict_csv(args.input, output_path)
            
        else:  # interactive mode
            if not args.model_path:
                raise ValueError("--model-path required for interactive mode")
                
            predictor = OptimizedPredictor(
                model_path=args.model_path,
                data_processor=data_processor,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            predict_interactive(predictor)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main():
                self.model = SentenceTransformer(early_stopping.best_model_path)
                self.logger.info(f"Loaded best model from {early_stopping.best_model_path
