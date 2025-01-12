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
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import argparse
import yaml
import json
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Add performance monitoring
import time
class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        end_time = time.time()
        duration = end_time - self.start_time
        logger.info(f"{self.name} took {duration:.2f} seconds")

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
        self.metrics: Dict[str, List[float]] = {}
        self.epoch_metrics: Dict[str, float] = {}
        
    def add_metric(self, name: str, value: float, epoch: Optional[int] = None) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
        if epoch is not None:
            self.epoch_metrics[f"{name}_epoch_{epoch}"] = value
            
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

class ConceptDataset(Dataset):
    def __init__(self, sampling_results: List[TrainingBatch]):
        self.batches = sampling_results
        
    def __len__(self) -> int:
        return len(self.batches)
        
    def __getitem__(self, idx: int) -> TrainingBatch:
        return self.batches[idx]

class DataProcessor:
    def __init__(self, attributes_path: str, concepts_path: str, test_size: float = 0.2, 
                 random_state: int = 42, num_workers: int = 4):
        self.attributes_path = Path(attributes_path)
        self.concepts_path = Path(concepts_path)
        self.test_size = test_size
        self.random_state = random_state
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        
        # Initialize caches
        self.attributes_df = None
        self.concepts_df = None
        self.domain_concepts = {}
        self.concept_definitions_cache = {}
        self.train_indices = None
        self.val_indices = None
        
        with Timer("Data Loading"):
            self.load_data()

    def load_data(self) -> None:
        try:
            self.logger.info("Loading and preprocessing data...")
            
            # Load data in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                attr_future = executor.submit(pd.read_csv, self.attributes_path)
                concepts_future = executor.submit(pd.read_csv, self.concepts_path)
                
                self.attributes_df = attr_future.result()
                self.concepts_df = concepts_future.result()
            
            # Validate columns
            required_attr_cols = ['attribute_name', 'description', 'domain', 'concept']
            required_concept_cols = ['domain', 'concept', 'concept_definition']
            
            self._validate_columns(self.attributes_df, required_attr_cols, 'attributes')
            self._validate_columns(self.concepts_df, required_concept_cols, 'concepts')
            
            # Create train/validation split with stratification
            all_indices = np.arange(len(self.attributes_df))
            self.train_indices, self.val_indices = train_test_split(
                all_indices,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.attributes_df['domain'].values
            )
            
            # Build caches in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                executor.submit(self._build_domain_concepts_cache)
                executor.submit(self._build_concept_definitions_cache)
            
            self.logger.info(
                f"Loaded {len(self.attributes_df)} attributes and {len(self.concepts_df)} concepts. "
                f"Split into {len(self.train_indices)} train and {len(self.val_indices)} validation samples"
            )
        
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_columns(self, df: pd.DataFrame, required_cols: List[str], df_name: str) -> None:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {df_name} CSV: {missing_cols}")

    def _build_domain_concepts_cache(self) -> None:
        for _, row in self.concepts_df.iterrows():
            if row['domain'] not in self.domain_concepts:
                self.domain_concepts[row['domain']] = set()
            self.domain_concepts[row['domain']].add(row['concept'])
        
        self.logger.info(f"Built domain concepts cache with {len(self.domain_concepts)} domains")

    def _build_concept_definitions_cache(self) -> None:
        for _, row in self.concepts_df.iterrows():
            key = (row['domain'], row['concept'])
            self.concept_definitions_cache[key] = row['concept_definition']
        
        self.logger.info(f"Built concept definitions cache with {len(self.concept_definitions_cache)} entries")

    def get_attribute_text(self, row: pd.Series) -> str:
        return f"{row['attribute_name']} {row['description']}"

    def get_concept_definition(self, domain: str, concept: str) -> str:
        key = (domain, concept)
        if key not in self.concept_definitions_cache:
            raise ValueError(f"No definition found for domain={domain}, concept={concept}")
        return self.concept_definitions_cache[key]

    def get_domains(self) -> List[str]:
        return list(self.domain_concepts.keys())

    def get_concepts_for_domain(self, domain: str) -> List[str]:
        return list(self.domain_concepts.get(domain, set()))

    def get_train_attributes(self) -> pd.DataFrame:
        return self.attributes_df.iloc[self.train_indices]

    def get_val_attributes(self) -> pd.DataFrame:
        return self.attributes_df.iloc[self.val_indices]

class PairSampler:
    def __init__(self, data_processor: DataProcessor, batch_size: int = 32, num_workers: int = 4):
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentence transformer for similarity computation
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compute_description_similarity(self, desc1: str, desc2: str) -> float:
        with torch.no_grad():
            embeddings1 = self.model.encode(desc1, convert_to_tensor=True)
            embeddings2 = self.model.encode(desc2, convert_to_tensor=True)
            similarity = F.cosine_similarity(embeddings1, embeddings2, dim=0).item()
        return similarity

    def _sample_hard_negatives(self, domain: str, concept: str, 
                             positive_attrs: pd.DataFrame, n_required: int,
                             attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        hard_negatives = []
        
        # Get all potential negatives
        other_mask = attributes_df['domain'] != domain | \
                     (attributes_df['domain'] == domain & attributes_df['concept'] != concept)
        potential_negatives = attributes_df[other_mask]
        
        # Calculate similarities
        all_similarities = []
        all_rows = []
        
        # Compute similarities between each positive and potential negative
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for _, pos_row in positive_attrs.iterrows():
                pos_desc = pos_row['description']
                for _, neg_row in potential_negatives.iterrows():
                    futures.append(
                        executor.submit(
                            self.compute_description_similarity,
                            pos_desc,
                            neg_row['description']
                        )
                    )
                    all_rows.append(neg_row)
                    
            # Collect similarities
            similarities = [f.result() for f in futures]
            all_similarities.extend(similarities)
            
        # Create array for easier filtering    
        similarities_array = np.array(all_similarities)
        high_sim_indices = np.where(similarities_array > 0.8)[0]
        
        # Sort by similarity to get hardest negatives first
        sorted_indices = high_sim_indices[np.argsort(-similarities_array[high_sim_indices])]
        
        # Take top n_required high similarity negatives
        for idx in sorted_indices[:n_required]:
            row = all_rows[idx]
            neg_text = self.data_processor.get_attribute_text(row)
            neg_def = self.data_processor.get_concept_definition(row['domain'], row['concept'])
            
            # Weight based on domain
            weight = 1.5 if row['domain'] == domain else 1.3
            hard_negatives.append((neg_text, neg_def, weight))
            
        return hard_negatives

    def _sample_medium_negatives(self, domain: str, concept: str, 
                               n_required: int, attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample medium negatives from different domains using parallel processing."""
        medium_negatives = []
        other_domains = [d for d in self.data_processor.get_domains() if d != domain]
        
        if not other_domains:
            self.logger.warning("No other domains found for medium negatives")
            return medium_negatives
            
        sample_domains = np.random.choice(other_domains, n_required, replace=True)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for d in sample_domains:
                concepts = self.data_processor.get_concepts_for_domain(d)
                if not concepts:
                    continue
                    
                c = np.random.choice(list(concepts))
                futures.append(
                    executor.submit(
                        self._get_medium_negative,
                        d, c, attributes_df
                    )
                )
            
            for future in futures:
                result = future.result()
                if result is not None:
                    medium_negatives.append(result)
                    
                if len(medium_negatives) >= n_required:
                    break
                    
        return medium_negatives[:n_required]

    def _get_medium_negative(self, domain: str, concept: str, 
                           attributes_df: pd.DataFrame) -> Optional[Tuple[str, str, float]]:
        """Helper method to get a single medium negative example."""
        mask = (attributes_df['domain'] == domain) & \
               (attributes_df['concept'] == concept)
        neg_attrs = attributes_df[mask]
        
        if len(neg_attrs) == 0:
            return None
            
        neg_row = neg_attrs.iloc[np.random.randint(len(neg_attrs))]
        neg_text = self.data_processor.get_attribute_text(neg_row)
        neg_def = self.data_processor.get_concept_definition(domain, concept)
        
        return (neg_text, neg_def, 1.0)

    def sample_pairs(self, domain: str, concept: str, is_training: bool = True) -> SamplingResult:
        """Sample positive and negative pairs using parallel processing."""
        try:
            with Timer(f"Sampling pairs for {domain}-{concept}"):
                # Get attributes based on split
                attributes_df = (
                    self.data_processor.get_train_attributes() if is_training 
                    else self.data_processor.get_val_attributes()
                )
                
                # Get all positive attributes
                mask = (attributes_df['domain'] == domain) & \
                      (attributes_df['concept'] == concept)
                positive_attrs = attributes_df[mask]
                
                if len(positive_attrs) == 0:
                    raise ValueError(f"No attributes found for domain={domain}, concept={concept}")

                concept_def = self.data_processor.get_concept_definition(domain, concept)
                
                # Create positive pairs in parallel
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    positive_pairs = list(executor.map(
                        lambda row: (self.data_processor.get_attribute_text(row[1]), concept_def),
                        positive_attrs.iterrows()
                    ))
                
                n_required_negatives = len(positive_pairs)
                
                # Sample hard negatives (40%) and medium negatives (60%) in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    n_hard = int(0.4 * n_required_negatives)
                    hard_future = executor.submit(
                        self._sample_hard_negatives,
                        domain, concept, positive_attrs, n_hard, attributes_df
                    )
                    
                    n_medium = n_required_negatives - n_hard
                    medium_future = executor.submit(
                        self._sample_medium_negatives,
                        domain, concept, n_medium, attributes_df
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
                
                self.logger.info(
                    f"Sampled pairs for {domain}-{concept}: "
                    f"{stats['n_positives']} positives, "
                    f"{stats['n_hard_negatives']} hard negatives, "
                    f"{stats['n_medium_negatives']} medium negatives"
                )
                
                return SamplingResult(positive_pairs, negative_pairs, stats)
                
        except Exception as e:
            self.logger.error(f"Error sampling pairs for {domain}-{concept}: {str(e)}")
            raise

class CustomInfoNCELoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature: float = 0.07):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.scaler = GradScaler()
        
    def forward(self, anchors: List[str], positives: List[str], 
                negatives: List[str], negative_weights: List[float]) -> torch.Tensor:
        """Compute weighted InfoNCE loss with mixed precision training."""
        with autocast():
            # Encode all texts in parallel batches
            all_texts = anchors + positives + negatives
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(all_texts), batch_size):
                batch = all_texts[i:i + batch_size]
                with torch.no_grad():
                    embeddings = self.model.encode(batch, convert_to_tensor=True)
                    all_embeddings.append(embeddings)
            
            all_embeddings = torch.cat(all_embeddings)
            
            # Split embeddings
            n = len(anchors)
            anchor_embeddings = all_embeddings[:n]
            positive_embeddings = all_embeddings[n:2*n]
            negative_embeddings = all_embeddings[2*n:]
            
            # Normalize embeddings
            anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
            positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
            
            # Compute similarities
            positive_similarities = torch.sum(
                anchor_embeddings * positive_embeddings, dim=-1
            ) / self.temperature
            
            negative_similarities = torch.matmul(
                anchor_embeddings, negative_embeddings.transpose(0, 1)
            ) / self.temperature
            
            # Apply weights to negative similarities
            negative_weights = torch.tensor(negative_weights).to(negative_similarities.device)
            negative_similarities = negative_similarities * negative_weights.unsqueeze(0)
            
            # Compute InfoNCE loss
            logits = torch.cat([positive_similarities.unsqueeze(-1), negative_similarities], dim=-1)
            labels = torch.zeros(len(anchors), dtype=torch.long, device=logits.device)
            
            return F.cross_entropy(logits, labels)

class ModelTrainer:
    def __init__(self, 
                model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                batch_size: int = 32,
                num_epochs: int = 10,
                learning_rate: float = 2e-5,
                temperature: float = 0.07,
                output_dir: str = './model_output',
                patience: int = 5,
                num_workers: int = 4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.output_dir = output_dir
        self.patience = patience
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and move to GPU if available
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize mixed precision training
        self.scaler = GradScaler()

    def prepare_batches(self, data_processor: DataProcessor, sampler: PairSampler, 
                       is_training: bool = True) -> List[TrainingBatch]:
        """Prepare batches for training or validation using parallel processing."""
        batches = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for domain in data_processor.get_domains():
                for concept in data_processor.get_concepts_for_domain(domain):
                    futures.append(
                        executor.submit(
                            self._prepare_single_batch,
                            sampler, domain, concept, is_training
                        )
                    )
            
            for future in futures:
                try:
                    batch = future.result()
                    if batch is not None:
                        batches.append(batch)
                except Exception as e:
                    self.logger.error(f"Error preparing batch: {str(e)}")
                    continue
                    
        return batches

    def _prepare_single_batch(self, sampler: PairSampler, domain: str, 
                            concept: str, is_training: bool) -> Optional[TrainingBatch]:
        """Prepare a single batch for a domain-concept pair."""
        try:
            result = sampler.sample_pairs(domain, concept, is_training)
            
            return TrainingBatch(
                anchors=[result.positive_pairs[0][1]] * len(result.positive_pairs),
                positives=[pair[0] for pair in result.positive_pairs],
                negatives=[pair[0] for pair in result.negative_pairs],
                negative_weights=[pair[2] for pair in result.negative_pairs]
            )
            
        except Exception as e:
            self.logger.error(f"Error preparing batch for {domain}-{concept}: {str(e)}")
            return None

    def train_epoch(self, dataloader: DataLoader, loss_fn: CustomInfoNCELoss, 
                   optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler) -> float:
        """Train for one epoch using mixed precision training."""
        total_loss = 0
        batch_count = 0
        
        with tqdm(dataloader, desc="Training") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                with autocast():
                    loss = loss_fn(
                        batch.anchors,
                        batch.positives,
                        batch.negatives,
                        batch.negative_weights
                    )
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                scheduler.step()
                
                total_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({
                    'loss': total_loss / batch_count,
                    'lr': scheduler.get_last_lr()[0]
                })
                
        return total_loss / batch_count

    def validate(self, dataloader: DataLoader, loss_fn: CustomInfoNCELoss) -> float:
        """Run validation using mixed precision."""
        total_loss = 0
        batch_count = 0
        
        self.model.eval()
        with torch.no_grad(), autocast():
            with tqdm(dataloader, desc="Validation") as pbar:
                for batch in pbar:
                    loss = loss_fn(
                        batch.anchors,
                        batch.positives,
                        batch.negatives,
                        batch.negative_weights
                    )
                    
                    total_loss += loss.item()
                    batch_count += 1
                    pbar.set_postfix({'val_loss': total_loss / batch_count})
                    
        self.model.train()
        return total_loss / batch_count

    def train(self, data_processor: DataProcessor, sampler: PairSampler) -> None:
        """Train the model using InfoNCE loss with optimizations."""
        try:
            self.logger.info("Preparing training data...")
            
            # Initialize components
            metrics_tracker = MetricsTracker(self.output_dir)
            early_stopping = EarlyStopping(patience=self.patience)
            
            # Prepare training and validation batches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                train_future = executor.submit(self.prepare_batches, data_processor, sampler, True)
                val_future = executor.submit(self.prepare_batches, data_processor, sampler, False)
                
                train_batches = train_future.result()
                val_batches = val_future.result()
            
            if not train_batches or not val_batches:
                raise ValueError("No training or validation batches could be prepared")
                
            train_dataset = ConceptDataset(train_batches)
            val_dataset = ConceptDataset(val_batches)
            
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=1, 
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=1, 
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            # Initialize loss and optimizer
            loss_fn = CustomInfoNCELoss(self.model, self.temperature)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            
            # Initialize learning rate scheduler
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5,
                T_mult=2,
                eta_min=1e-6
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
                metrics_tracker.add_metric('learning_rate', scheduler.get_last_lr()[0], epoch)
                
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
                    f"Val Loss: {val_loss:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
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
            if early_stopping.best_model_path:
                self.model = SentenceTransformer(early_stopping.best_model_path)
                self.logger.info(f"Loaded best model from {early_stopping.best_model_path}")
            else:
                # Save final model if no best model
                final_path = os.path.join(self.output_dir, "final-model")
                self.model.save(final_path)
                self.logger.info(f"Saved final model to {final_path}")
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

class ModelPredictor:
    def __init__(self, 
                model_path: str,
                data_processor: DataProcessor,
                batch_size: int = 32,
                top_k: int = 3,
                num_workers: int = 4):
        self.model_path = model_path
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.top_k = top_k
        self.num_workers = num_workers
        self.logger = logging.getLogger(__name__)
        
        # Load model with CUDA optimization
        try:
            self.model = SentenceTransformer(model_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            if self.device.type == 'cuda':
                self.model = torch.nn.DataParallel(self.model)
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
        # Cache concept definitions for faster prediction
        with Timer("Caching concept definitions"):
            self.concept_definitions = self._cache_concept_definitions()
        
    def _cache_concept_definitions(self) -> Dict[Tuple[str, str], torch.Tensor]:
        """Cache all domain-concept definition embeddings for faster prediction."""
        definitions = {}
        
        with torch.no_grad(), autocast():
            for domain in self.data_processor.get_domains():
                for concept in self.data_processor.get_concepts_for_domain(domain):
                    try:
                        definition = self.data_processor.get_concept_definition(domain, concept)
                        embedding = self.model.encode(
                            definition,
                            convert_to_tensor=True,
                            show_progress_bar=False
                        )
                        definitions[(domain, concept)] = embedding
                    except Exception as e:
                        self.logger.warning(f"Could not cache definition for {domain}-{concept}: {str(e)}")
                        
        return definitions
        
    def predict_single(self, attribute_name: str, description: str) -> List[PredictionResult]:
        """Predict domain-concept for a single attribute using cached embeddings."""
        try:
            attribute_text = f"{attribute_name} {description}"
            
            # Encode attribute text with mixed precision
            with torch.no_grad(), autocast():
                attribute_embedding = self.model.encode(
                    attribute_text,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Compute similarities with all cached concept definitions
                results = []
                for (domain, concept), definition_embedding in self.concept_definitions.items():
                    similarity = torch.nn.functional.cosine_similarity(
                        attribute_embedding,
                        definition_embedding,
                        dim=0
                    ).item()
                    
                    results.append(PredictionResult(domain, concept, similarity))
                
                # Sort by confidence and return top-k
                results.sort(key=lambda x: x.confidence, reverse=True)
                return results[:self.top_k]
            
        except Exception as e:
            self.logger.error(f"Error in single prediction: {str(e)}")
            raise

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict domain-concepts for a batch of attributes using parallel processing."""
        try:
            # Validate input
            required_cols = ['attribute_name', 'description']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            results = []
            
            # Process in batches with parallel execution
            with Timer("Batch prediction"):
                for i in tqdm(range(0, len(df), self.batch_size)):
                    batch_df = df.iloc[i:i+self.batch_size]
                    
                    # Process each row in batch using ThreadPool
                    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        futures = []
                        for _, row in batch_df.iterrows():
                            futures.append(
                                executor.submit(
                                    self.predict_single,
                                    row['attribute_name'],
                                    row['description']
                                )
                            )
                        
                        # Collect results
                        for future in futures:
                            predictions = future.result()
                            if predictions:  # Take top prediction
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
            
            # Add predictions to DataFrame
            result_df = pd.DataFrame(results)
            return pd.concat([df.reset_index(drop=True), result_df], axis=1)
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise

    def predict_csv(self, input_path: str, output_path: str) -> None:
        """Predict domain-concepts for attributes in a CSV file with progress tracking."""
        try:
            with Timer("CSV prediction"):
                # Read input CSV
                self.logger.info(f"Reading input CSV from {input_path}")
                df = pd.read_csv(input_path)
                
                # Make predictions
                self.logger.info(f"Making predictions for {len(df)} rows...")
                result_df = self.predict_batch(df)
                
                # Save results
                self.logger.info(f"Saving predictions to {output_path}")
                result_df.to_csv(output_path, index=False)
                
                self.logger.info("Prediction complete!")
                
        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            raise

def predict_interactive(predictor: ModelPredictor):
    """Interactive prediction mode with improved error handling."""
    logger = logging.getLogger(__name__)
    logger.info("Starting interactive prediction mode. Type 'quit' to exit.")
    
    while True:
        try:
            attribute_name = input("\nEnter attribute name (or 'quit' to exit): ").strip()
            if attribute_name.lower() == 'quit':
                break
                
            description = input("Enter description: ").strip()
            
            with Timer("Interactive prediction"):
                predictions = predictor.predict_single(attribute_name, description)
            
            print("\nPredictions:")
            print("-" * 50)
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. Domain: {pred.domain}")
                print(f"   Concept: {pred.concept}")
                print(f"   Confidence: {pred.confidence:.4f}")
                print("-" * 50)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            print("An error occurred. Please try again.")

def setup_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> Path:
    """Create and setup experiment directory with timestamp and proper logging."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp
        
    experiment_dir = Path(base_dir) / dir_name
    
    # Create required subdirectories with parallel execution
    subdirs = ['models', 'logs', 'metrics', 'predictions', 'checkpoints']
    with ThreadPoolExecutor(max_workers=len(subdirs)) as executor:
        futures = [
            executor.submit(lambda d: (experiment_dir / d).mkdir(parents=True, exist_ok=True))
            for d in subdirs
        ]
        
        # Wait for all directories to be created
        for future in futures:
            future.result()
            
    return experiment_dir

def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration file with enhanced error checking."""
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
        
        with ThreadPoolExecutor(max_workers=len(data_paths)) as executor:
            futures = [
                executor.submit(lambda p: p.exists())
                for p in data_paths
            ]
            
            # Check all paths exist
            for path, future in zip(data_paths, futures):
                if not future.result():
                    raise FileNotFoundError(f"Data file not found: {path}")
                    
        return config
        
    except Exception as e:
        logger.error(f"Error in configuration: {str(e)}")
        raise

def main():
    # Parse command line arguments with validation
    parser = argparse.ArgumentParser(description='Train or predict with sentence transformer model')
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
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of worker threads for parallel processing')
    
    args = parser.parse_args()
    
    try:
        # Initialize data processor with train/test split
        with Timer("Initializing data processor"):
            data_processor = DataProcessor(
                args.attributes, 
                args.concepts,
                test_size=args.test_size,
                random_state=args.random_state,
                num_workers=args.num_workers
            )
        
        if args.mode == 'train':
            # Training mode
            sampler = PairSampler(
                data_processor, 
                args.batch_size,
                num_workers=args.num_workers
            )
            trainer = ModelTrainer(
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
                
            predictor = ModelPredictor(
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
                
            predictor = ModelPredictor(
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
    main()
