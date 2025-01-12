# First, let's modify the ContrastiveModel class to remove MLP
class ContextualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Pretrained transformer
        self.encoder = SentenceTransformer(config.model_name)
        
        # Additional context understanding layers
        self.context_projection = nn.Sequential(
            nn.Linear(self.encoder.get_sentence_embedding_dimension(), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256)  # Normalize to stabilize learning
        )
    
    def forward(self, texts):
        # Get base embeddings with rich context
        base_embeddings = self.encoder.encode(
            texts, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Add contextual understanding projection
        contextual_embeddings = self.context_projection(base_embeddings)
        
        return F.normalize(contextual_embeddings, p=2, dim=1)

class AdvancedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, anchors, positives, negatives):
        # Compute similarities with margin
        pos_similarities = torch.sum(anchors * positives, dim=1) / self.temperature
        neg_similarities = torch.sum(anchors * negatives, dim=1) / self.temperature
        
        # Triplet-like loss mechanism
        loss = F.relu(neg_similarities - pos_similarities + self.margin)
        
        return loss.mean()

def load_and_split_data(self, attributes_path: str, concepts_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load data and create train/val/test splits with caching."""
    self.logger.info("Loading datasets...")
    
    # Define cache paths
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_data.json")
    val_cache = os.path.join(cache_dir, "val_data.json")
    test_cache = os.path.join(cache_dir, "test_data.json")
    
    # Try to load from cache first
    if all(os.path.exists(f) for f in [train_cache, val_cache, test_cache]):
        self.logger.info("Loading from cache...")
        try:
            with open(train_cache, 'r') as f:
                train_data = json.load(f)
            with open(val_cache, 'r') as f:
                val_data = json.load(f)
            with open(test_cache, 'r') as f:
                test_data = json.load(f)
            self.logger.info("Successfully loaded from cache!")
            return train_data, val_data, test_data
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}. Processing data from scratch...")
    
    # If cache doesn't exist or failed to load, process normally
    self.logger.info("Processing data from scratch...")
    
    # Load data
    attributes_df = pd.read_csv(attributes_path)
    concepts_df = pd.read_csv(concepts_path)
    
    # [Rest of your existing data processing code]
    
    # Save to cache
    self.logger.info("Saving processed data to cache...")
    try:
        with open(train_cache, 'w') as f:
            json.dump(train_data, f)
        with open(val_cache, 'w') as f:
            json.dump(val_data, f)
        with open(test_cache, 'w') as f:
            json.dump(test_data, f)
        self.logger.info("Successfully cached processed data!")
    except Exception as e:
        self.logger.warning(f"Failed to cache data: {e}")
    
    return train_data, val_data, test_data


def _validate(self, val_loader: DataLoader) -> float:
    """Validate the model."""
    self.model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for anchors, others, labels in val_loader:
            # Move tensors to device
            anchors = anchors.to(self.device)
            others = others.to(self.device)
            labels = labels.to(self.device)
            
            # Get embeddings
            anchor_embeds = self.model(anchors)
            other_embeds = self.model(others)
            
            # Get positives and negatives based on labels
            positives = other_embeds[labels == 1]
            anchors_pos = anchor_embeds[labels == 1]
            negatives = other_embeds[labels == 0]
            
            if len(positives) > 0 and len(negatives) > 0:
                # Compute loss
                loss = self.criterion(anchors_pos, positives, negatives)
                total_loss += loss.item()
                
    return total_loss / len(val_loader)

def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
    """Train for one epoch using explicit negatives."""
    self.model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as pbar:
        for batch_idx, (anchors, others, labels) in enumerate(pbar):
            # Get embeddings
            anchor_embeds = self.model(anchors)
            other_embeds = self.model(others)
            
            # Get positives and negatives based on labels
            positives = other_embeds[labels == 1]
            anchors_pos = anchor_embeds[labels == 1]  # Get corresponding anchors
            negatives = other_embeds[labels == 0]
            
            if len(positives) > 0 and len(negatives) > 0:
                # Compute loss
                loss = self.criterion(anchors_pos, positives, negatives)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            
    return total_loss / len(train_loader)

class ContrastiveModel(nn.Module):
    """Model using only the sentence transformer encoder."""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.encoder = SentenceTransformer(config.model_name)
        
        # Freeze encoder if in MLP mode
        if config.train_mode == 'mlp':
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        # Get embeddings and normalize them
        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Normalize outputs
        outputs = F.normalize(embeddings, p=2, dim=1)
        return outputs

# Now modify the training loop to use explicit negatives
def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
    """Train for one epoch using explicit negatives."""
    self.model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as pbar:
        for batch_idx, (anchors, positives, negatives) in enumerate(pbar):
            # Get embeddings
            anchor_embeds = self.model(anchors)
            positive_embeds = self.model(positives)
            negative_embeds = self.model(negatives)
            
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

# And update the validation function similarly
def _validate(self, val_loader: DataLoader) -> float:
    """Validate the model using explicit negatives."""
    self.model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for anchors, positives, negatives in val_loader:
            # Get embeddings
            anchor_embeds = self.model(anchors)
            positive_embeds = self.model(positives)
            negative_embeds = self.model(negatives)
            
            # Compute loss
            loss = self.criterion(anchor_embeds, positive_embeds, negative_embeds)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)








def evaluate_test(self, test_data: Dict, output_dir: str) -> None:
    """Evaluate model on test set and save metrics."""
    self.logger.info("Evaluating on test set...")
    
    # Create test dataset and loader
    test_dataset = ContrastiveDataset(**test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=self.config.batch_size,
        shuffle=False,
        num_workers=self.config.num_workers,
        pin_memory=True
    )
    
    # Get test loss
    test_loss = self._validate(test_loader)
    
    # Calculate additional metrics
    metrics = {
        'test_loss': test_loss,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    self.logger.info(f"Test Loss: {test_loss:.4f}")
    self.logger.info(f"Saved test metrics to {metrics_path}")





class BatchSiameseDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        attributes_df: pd.DataFrame,
        concepts_df: pd.DataFrame,
        batch_size: int = 64,
        num_workers: int = 4
    ):
        # Validate input dataframes
        if attributes_df is None or len(attributes_df) == 0:
            raise ValueError("Attributes DataFrame is empty or None")
        if concepts_df is None or len(concepts_df) == 0:
            raise ValueError("Concepts DataFrame is empty or None")
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Validate required columns
        required_attr_cols = ['attribute_name', 'description', 'domain', 'concept']
        required_concept_cols = ['domain', 'concept', 'concept_definition']
        
        missing_attr_cols = [col for col in required_attr_cols if col not in attributes_df.columns]
        missing_concept_cols = [col for col in required_concept_cols if col not in concepts_df.columns]
        
        if missing_attr_cols:
            raise ValueError(f"Missing columns in attributes DataFrame: {missing_attr_cols}")
        if missing_concept_cols:
            raise ValueError(f"Missing columns in concepts DataFrame: {missing_concept_cols}")
        
        # Parallel pair creation with efficient memory management
        self.pairs, self.labels = self._create_pairs_parallel(
            attributes_df, 
            concepts_df
        )
        
        # Additional validation
        if not self.pairs or len(self.pairs) == 0:
            raise ValueError("No pairs could be generated. Check your input data.")
    
    def _create_pairs_parallel(
        self, 
        attributes_df: pd.DataFrame, 
        concepts_df: pd.DataFrame
    ) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Parallel pair creation with efficient chunk processing
        """
        def process_chunk(attr_chunk, concepts_df):
            chunk_pairs = []
            chunk_labels = []
            
            for _, attr_row in attr_chunk.iterrows():
                attribute_text = f"{attr_row['attribute_name']} {attr_row['description']}"
                
                for _, concept_row in concepts_df.iterrows():
                    concept_text = (
                        f"{concept_row['domain']}-{concept_row['concept']}: "
                        f"{concept_row['concept_definition']}"
                    )
                    
                    # Label based on domain and concept match
                    label = 1.0 if (
                        attr_row['domain'] == concept_row['domain'] and 
                        attr_row['concept'] == concept_row['concept']
                    ) else 0.0
                    
                    chunk_pairs.append((attribute_text, concept_text))
                    chunk_labels.append(label)
            
            return chunk_pairs, chunk_labels
        
        # Ensure we have at least as many workers as chunks
        num_workers = min(self.num_workers, len(attributes_df))
        
        # Split attributes into chunks for parallel processing
        chunks = np.array_split(attributes_df, num_workers)
        
        # Use ThreadPoolExecutor for parallel pair generation
        all_pairs = []
        all_labels = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_chunk, chunk, concepts_df) 
                for chunk in chunks if not chunk.empty
            ]
            
            for future in as_completed(futures):
                pairs, labels = future.result()
                all_pairs.extend(pairs)
                all_labels.extend(labels)
        
        return all_pairs, all_labels

class SiameseNetwork(nn.Module):
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        # Load pretrained transformer model
        self.encoder = SentenceTransformer(model_name)
        
        # Get embedding dimension
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Custom similarity network for batch processing
        self.similarity_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Batch encoding with parallel processing
        Ensures gradient computation and handles large batches efficiently
        """
        # Validate input
        if not texts or len(texts) == 0:
            raise ValueError("Input texts list is empty")
        
        # Encode texts with batch processing
        embeddings = self.encoder.encode(
            texts, 
            convert_to_tensor=True,
            batch_size=len(texts),  # Use full batch size
            show_progress_bar=False
        )
        
        # Ensure embeddings have correct dimensions
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # Normalize embeddings
        return F.normalize(embeddings, p=2, dim=1)
    
    def forward(self, text1: List[str], text2: List[str]) -> torch.Tensor:
        # Validate input
        if not text1 or not text2:
            raise ValueError("Input text lists cannot be empty")
        
        # Ensure equal length inputs
        if len(text1) != len(text2):
            raise ValueError(f"Input lists must have equal length. text1: {len(text1)}, text2: {len(text2)}")
        
        # Batch encode both sets of texts
        batch_embeddings1 = self.encode_batch(text1)
        batch_embeddings2 = self.encode_batch(text2)
        
        # Combine embeddings
        combined = torch.cat([batch_embeddings1, batch_embeddings2], dim=1)
        
        # Compute similarity using learned network
        similarities = self.similarity_network(combined).squeeze()
        
        return similarities

class BatchModelTrainer:
    def __init__(
        self, 
        model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        batch_size: int = 64,
        num_epochs: int = 15,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        output_dir: str = './model_output'
    ):
        # Validate input parameters
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if num_epochs < 1:
            raise ValueError("Number of epochs must be at least 1")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup model with batch-aware architecture
        self.model = SiameseNetwork(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and scheduler for batch training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )
        
        # Training hyperparameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def train_batch(self, dataloader, criterion):
        """
        Batch-aware training method
        Supports efficient gradient computation and tracking
        """
        # Validate dataloader
        if len(dataloader) == 0:
            raise ValueError("Empty dataloader. No data to train on.")
        
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_text1, batch_text2, batch_labels in tqdm(dataloader, desc="Training"):
            # Validate batch
            if len(batch_text1) == 0 or len(batch_text2) == 0:
                self.logger.warning("Skipping empty batch")
                continue
            
            # Move data to device
            batch_labels = batch_labels.float().to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with batch processing
            try:
                similarities = self.model(batch_text1, batch_text2)
            except Exception as e:
                self.logger.error(f"Forward pass error: {e}")
                continue
            
            # Compute loss
            loss = criterion(similarities, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = (similarities > 0.5).float()
            correct_predictions += (predictions == batch_labels).sum().item()
            total_predictions += len(batch_labels)
        
        # Compute average metrics
        if total_predictions == 0:
            self.logger.warning("No predictions made during training")
            return 0, 0
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy





import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SiameseNetwork(nn.Module):
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        # Load pretrained transformer model
        self.encoder = SentenceTransformer(model_name)
        
        # Get embedding dimension
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Custom similarity network for batch processing
        self.similarity_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Batch encoding with parallel processing
        Ensures gradient computation and handles large batches efficiently
        """
        # Encode texts with batch processing
        embeddings = self.encoder.encode(
            texts, 
            convert_to_tensor=True,
            batch_size=len(texts),  # Use full batch size
            show_progress_bar=False
        )
        
        # Normalize embeddings
        return F.normalize(embeddings, p=2, dim=1)
    
    def forward(self, text1: List[str], text2: List[str]) -> torch.Tensor:
        # Batch encode both sets of texts
        batch_embeddings1 = self.encode_batch(text1)
        batch_embeddings2 = self.encode_batch(text2)
        
        # Combine embeddings
        combined = torch.cat([batch_embeddings1, batch_embeddings2], dim=1)
        
        # Compute similarity using learned network
        similarities = self.similarity_network(combined).squeeze()
        
        return similarities

class BatchSiameseDataset(torch.utils.data.Dataset):
    """
    Enhanced dataset with batch-aware pair creation
    Supports efficient batch processing and parallel pair generation
    """
    def __init__(
        self, 
        attributes_df: pd.DataFrame,
        concepts_df: pd.DataFrame,
        batch_size: int = 64,
        num_workers: int = 4
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Parallel pair creation with efficient memory management
        self.pairs, self.labels = self._create_pairs_parallel(
            attributes_df, 
            concepts_df
        )
    
    def _create_pairs_parallel(
        self, 
        attributes_df: pd.DataFrame, 
        concepts_df: pd.DataFrame
    ) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Parallel pair creation with efficient chunk processing
        """
        def process_chunk(attr_chunk, concepts_df):
            chunk_pairs = []
            chunk_labels = []
            
            for _, attr_row in attr_chunk.iterrows():
                attribute_text = f"{attr_row['attribute_name']} {attr_row['description']}"
                
                for _, concept_row in concepts_df.iterrows():
                    concept_text = (
                        f"{concept_row['domain']}-{concept_row['concept']}: "
                        f"{concept_row['concept_definition']}"
                    )
                    
                    # Label based on domain and concept match
                    label = 1.0 if (
                        attr_row['domain'] == concept_row['domain'] and 
                        attr_row['concept'] == concept_row['concept']
                    ) else 0.0
                    
                    chunk_pairs.append((attribute_text, concept_text))
                    chunk_labels.append(label)
            
            return chunk_pairs, chunk_labels
        
        # Split attributes into chunks for parallel processing
        chunks = np.array_split(attributes_df, self.num_workers)
        
        # Use ThreadPoolExecutor for parallel pair generation
        all_pairs = []
        all_labels = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(process_chunk, chunk, concepts_df) 
                for chunk in chunks
            ]
            
            for future in as_completed(futures):
                pairs, labels = future.result()
                all_pairs.extend(pairs)
                all_labels.extend(labels)
        
        return all_pairs, all_labels
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, float]:
        return (*self.pairs[idx], self.labels[idx])

class BatchModelTrainer:
    def __init__(
        self, 
        model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        batch_size: int = 64,
        num_epochs: int = 15,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        output_dir: str = './model_output'
    ):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup model with batch-aware architecture
        self.model = SiameseNetwork(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and scheduler for batch training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )
        
        # Training hyperparameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def train_batch(self, dataloader, criterion):
        """
        Batch-aware training method
        Supports efficient gradient computation and tracking
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_text1, batch_text2, batch_labels in tqdm(dataloader, desc="Training"):
            # Move data to device
            batch_labels = batch_labels.float().to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with batch processing
            similarities = self.model(batch_text1, batch_text2)
            
            # Compute loss
            loss = criterion(similarities, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = (similarities > 0.5).float()
            correct_predictions += (predictions == batch_labels).sum().item()
            total_predictions += len(batch_labels)
        
        # Compute average metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy

    def train(self, attributes_df: pd.DataFrame, concepts_df: pd.DataFrame):
        """
        Comprehensive training method with batch optimization
        """
        # Create batch-aware dataset
        dataset = BatchSiameseDataset(
            attributes_df,
            concepts_df,
            batch_size=self.batch_size
        )
        
        # Create dataloader with batch processing
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Best model tracking
        best_accuracy = 0
        
        # Training loop
        for epoch in range(self.num_epochs):
            avg_loss, accuracy = self.train_batch(dataloader, criterion)
            
            # Learning rate scheduling
            self.scheduler.step(accuracy)
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = os.path.join(self.output_dir, 'best_model')
                os.makedirs(best_model_path, exist_ok=True)
                
                # Save model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': accuracy,
                    'epoch': epoch
                }, os.path.join(best_model_path, 'model_checkpoint.pt'))
                
                # Save encoder
                self.model.encoder.save(best_model_path)
                
                self.logger.info(f"New best model saved with accuracy {accuracy:.4f}")
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, 'final_model')
        os.makedirs(final_model_path, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': best_accuracy
        }, os.path.join(final_model_path, 'model_checkpoint.pt'))
        self.model.encoder.save(final_model_path)
        
        self.logger.info(f"Training completed. Best accuracy: {best_accuracy:.4f}")
        return best_accuracy

class ModelPredictor:
    def __init__(
        self, 
        model_path: str, 
        batch_size: int = 64, 
        top_k: int = 3
    ):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = SiameseNetwork()
        self.model.encoder = SentenceTransformer(model_path)
        
        # Load model state
        checkpoint = torch.load(os.path.join(model_path, 'model_checkpoint.pt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Prediction parameters
        self.batch_size = batch_size
        self.top_k = top_k
    
    def predict_batch(self, attributes_df: pd.DataFrame, concept_texts: List[str]):
        """
        Batch prediction with parallel processing
        """
        results = []
        
        # Process in batches
        for i in range(0, len(attributes_df), self.batch_size):
            batch_df = attributes_df.iloc[i:i+self.batch_size]
            
            # Prepare texts
            attribute_texts = [
                f"{row['attribute_name']} {row['description']}" 
                for _, row in batch_df.iterrows()
            ]
            
            # Batch prediction
            with torch.no_grad():
                similarities = self.model(
                    attribute_texts, 
                    concept_texts * len(attribute_texts)
                )
                
                # Reshape similarities
                similarities = similarities.view(
                    len(attribute_texts), 
                    len(concept_texts)
                )
            
            # Process predictions for each attribute
            for attr_similarities in similarities:
                # Get top-k predictions
                top_indices = torch.topk(attr_similarities, self.top_k).indices
                
                batch_results = []
                for idx in top_indices:
                    concept_text = concept_texts[idx]
                    domain, concept = concept_text.split(':')[0].split('-')
                    confidence = attr_similarities[idx].item()
                    
                    batch_results.append({
                        'domain': domain,
                        'concept': concept,
                        'confidence': confidence
                    })
                
                results.append(batch_results)
        
        return results

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Batch Siamese Network Training and Prediction')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--attributes', type=str, required=True, help='Path to attributes CSV')
    parser.add_argument('--concepts', type=str, required=True, help='Path to concepts CSV')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--model-path', type=str, help='Path to saved model (for prediction)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=15)
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load data
        attributes_df = pd.read_csv(args.attributes)
        concepts_df = pd.read_csv(args.concepts)
        
        if args.mode == 'train':
            # Training mode
            trainer = BatchModelTrainer(
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                output_dir=args.output_dir
            )
            
            # Train the model
            trainer.train(attributes_df, concepts_df)
            
        elif args.mode == 'predict':
            # Prediction mode
            if not args.model_path:
                raise ValueError("Model path is required for prediction")
            
            # Prepare concept texts
            concept_texts = [
                f"{row['domain']}-{row['concept']}: {row['concept_definition']}"
                for _, row in concepts_df.iterrows()
            ]
            
            # Initialize predictor
            predictor = ModelPredictor(












class SiameseNetwork(nn.Module):
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        
        # Create a neural network layer for final similarity computation
        self.similarity_layer = nn.Sequential(
            nn.Linear(self.encoder.get_sentence_embedding_dimension(), 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, text1: List[str], text2: List[str]) -> torch.Tensor:
        try:
            # Convert to tensors with gradient computation
            embeddings1 = torch.stack([
                torch.tensor(self.encoder.encode(t, convert_to_tensor=False), requires_grad=True) 
                for t in text1
            ])
            
            embeddings2 = torch.stack([
                torch.tensor(self.encoder.encode(t, convert_to_tensor=False), requires_grad=True) 
                for t in text2
            ])
            
            # Normalize embeddings
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)
            
            # Compute similarity using learned layer
            similarities = []
            for emb1, emb2 in zip(embeddings1, embeddings2):
                combined = torch.abs(emb1 - emb2)
                similarity = self.similarity_layer(combined).squeeze()
                similarities.append(similarity)
            
            return torch.stack(similarities)
        
        except Exception as e:
            print(f"Forward pass error: {e}")
            raise




class BatchedPairSampler:
    # ...

    def _sample_hard_negatives(self, domain: str, concept: str,
                              positive_attrs: pd.DataFrame, n_required: int,
                              attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample hard negatives using batched operations."""
        hard_negatives = []
        other_concepts = [c for c in self.data_processor.get_concepts_for_domain(domain) 
                         if c != concept]
        
        self.logger.info(f"Sampling hard negatives for domain: {domain}, concept: {concept}")
        
        # Validate input DataFrames
        self._validate_dataframes(positive_attrs, attributes_df)
        
        # Get potential negative attributes efficiently
        mask = (attributes_df['domain'] == domain) & (attributes_df['concept'].isin(other_concepts))
        potential_negatives = attributes_df[mask]
        
        if len(potential_negatives) == 0:
            self.logger.info("No potential negative attributes found, skipping hard negatives.")
            return hard_negatives
            
        # Prepare texts for batch processing
        pos_texts = [
            row['description'] for _, row in positive_attrs.iterrows()
        ]
        neg_texts = [
            row['description'] for _, row in potential_negatives.iterrows()
        ]
        
        self.logger.info(f"Positive texts count: {len(pos_texts)}")
        self.logger.info(f"Negative texts count: {len(neg_texts)}")
        
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
        
        self.logger.info(f"Number of high similarity pairs: {len(high_sim_indices)}")
        
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
        
        self.logger.info(f"Sampling medium negatives for domain: {domain}, concept: {concept}")
        
        # Validate input DataFrame
        self._validate_dataframes(attributes_df)
        
        if not other_domains:
            self.logger.info("No other domains found, skipping medium negatives.")
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
                self.logger.info(f"No concepts found for domain: {d}, skipping.")
                continue
                
            # Sample concepts for entire batch
            batch_concepts = np.random.choice(list(concepts), len(batch))
            
            # Get attributes efficiently
            mask = (attributes_df['domain'] == d) & (attributes_df['concept'].isin(set(batch_concepts)))
            neg_attrs = attributes_df[mask]
            
            if len(neg_attrs) == 0:
                self.logger.info(f"No negative attributes found for domain: {d}, skipping.")
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

    def _validate_dataframes(self, *dfs: pd.DataFrame) -> None:
        """Validate input DataFrames."""
        for df in dfs:
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input must be a pandas DataFrame, got {type(df)}")
            
            if df.empty:
                raise ValueError(f"DataFrame is empty.")
            
            if df.index.dtype != 'int64':
                raise ValueError(f"DataFrame index must be integer, got {df.index.dtype}")






class BatchedPairSampler:
    # ...

    def _sample_hard_negatives(self, domain: str, concept: str,
                              positive_attrs: pd.DataFrame, n_required: int,
                              attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample hard negatives using batched operations."""
        hard_negatives = []
        other_concepts = [c for c in self.data_processor.get_concepts_for_domain(domain) 
                         if c != concept]
        
        self.logger.info(f"Sampling hard negatives for domain: {domain}, concept: {concept}")
        self.logger.info(f"Positive attributes shape: {positive_attrs.shape}")
        self.logger.info(f"Other concepts: {other_concepts}")
        
        # Reset index to ensure consistent integer indexing
        positive_attrs = positive_attrs.reset_index(drop=True)
        attributes_df = attributes_df.reset_index(drop=True)
        
        # Get potential negative attributes efficiently
        mask = (attributes_df['domain'] == domain) & (attributes_df['concept'].isin(other_concepts))
        potential_negatives = attributes_df[mask]
        
        if len(potential_negatives) == 0:
            self.logger.info("No potential negative attributes found, skipping hard negatives.")
            return hard_negatives
            
        # Prepare texts for batch processing
        pos_texts = [
            row['description'] for _, row in positive_attrs.iterrows()
        ]
        neg_texts = [
            row['description'] for _, row in potential_negatives.iterrows()
        ]
        
        self.logger.info(f"Positive texts count: {len(pos_texts)}")
        self.logger.info(f"Negative texts count: {len(neg_texts)}")
        
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
        
        self.logger.info(f"Number of high similarity pairs: {len(high_sim_indices)}")
        
        # Create hard negatives
        for pos_idx, neg_idx in high_sim_indices[:n_required]:
            try:
                neg_row = potential_negatives.iloc[neg_idx]
            except IndexError:
                self.logger.error(f"Non-integer index encountered when accessing potential_negatives DataFrame.")
                self.logger.error(f"Domain: {domain}, Concept: {concept}")
                self.logger.error(f"Positive attributes shape: {positive_attrs.shape}")
                self.logger.error(f"Potential negatives shape: {potential_negatives.shape}")
                self.logger.error(f"High similarity pairs count: {len(high_sim_indices)}")
                self.logger.error(f"Required negatives: {n_required}")
                raise
            
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
        
        self.logger.info(f"Sampling medium negatives for domain: {domain}, concept: {concept}")
        self.logger.info(f"Other domains: {other_domains}")
        
        # Reset index to ensure consistent integer indexing
        attributes_df = attributes_df.reset_index(drop=True)
        
        if not other_domains:
            self.logger.info("No other domains found, skipping medium negatives.")
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
                self.logger.info(f"No concepts found for domain: {d}, skipping.")
                continue
                
            # Sample concepts for entire batch
            batch_concepts = np.random.choice(list(concepts), len(batch))
            
            # Get attributes efficiently
            mask = (attributes_df['domain'] == d) & (attributes_df['concept'].isin(set(batch_concepts)))
            neg_attrs = attributes_df[mask]
            
            if len(neg_attrs) == 0:
                self.logger.info(f"No negative attributes found for domain: {d}, skipping.")
                continue
                
            # Sample attributes for batch
            try:
                sampled_indices = np.random.randint(0, len(neg_attrs), len(batch))
                sampled_attrs = neg_attrs.iloc[sampled_indices]
            except IndexError:
                self.logger.error(f"Non-integer index encountered when accessing neg_attrs DataFrame.")
                self.logger.error(f"Domain: {d}, Concepts: {batch_concepts}")
                self.logger.error(f"Negative attributes shape: {neg_attrs.shape}")
                self.logger.error(f"Batch size: {len(batch)}")
                raise
            
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





class BatchedPairSampler:
    # ...

    def _sample_hard_negatives(self, domain: str, concept: str,
                              positive_attrs: pd.DataFrame, n_required: int,
                              attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample hard negatives using batched operations."""
        hard_negatives = []
        other_concepts = [c for c in self.data_processor.get_concepts_for_domain(domain) 
                         if c != concept]
        
        self.logger.info(f"Sampling hard negatives for domain: {domain}, concept: {concept}")
        self.logger.info(f"Positive attributes shape: {positive_attrs.shape}")
        self.logger.info(f"Other concepts: {other_concepts}")
        
        if not other_concepts:
            self.logger.info("No other concepts found for the domain, skipping hard negatives.")
            return hard_negatives
            
        # Get potential negative attributes efficiently
        mask = (attributes_df['domain'] == domain) & (attributes_df['concept'].isin(other_concepts))
        potential_negatives = attributes_df[mask]
        
        if len(potential_negatives) == 0:
            self.logger.info("No potential negative attributes found, skipping hard negatives.")
            return hard_negatives
            
        # Prepare texts for batch processing
        pos_texts = [
            row['description'] for _, row in positive_attrs.iterrows()
        ]
        neg_texts = [
            row['description'] for _, row in potential_negatives.iterrows()
        ]
        
        self.logger.info(f"Positive texts count: {len(pos_texts)}")
        self.logger.info(f"Negative texts count: {len(neg_texts)}")
        
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
        
        self.logger.info(f"Number of high similarity pairs: {len(high_sim_indices)}")
        
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
        
        self.logger.info(f"Sampling medium negatives for domain: {domain}, concept: {concept}")
        self.logger.info(f"Other domains: {other_domains}")
        
        if not other_domains:
            self.logger.info("No other domains found, skipping medium negatives.")
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
                self.logger.info(f"No concepts found for domain: {d}, skipping.")
                continue
                
            # Sample concepts for entire batch
            batch_concepts = np.random.choice(list(concepts), len(batch))
            
            # Get attributes efficiently
            mask = (attributes_df['domain'] == d) & (attributes_df['concept'].isin(set(batch_concepts)))
            neg_attrs = attributes_df[mask]
            
            if len(neg_attrs) == 0:
                self.logger.info(f"No negative attributes found for domain: {d}, skipping.")
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
