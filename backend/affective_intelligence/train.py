"""
Training script for facial emotion recognition model.

Supports:
- Fine-tuning on macro-expressions (FER2013)
- Fine-tuning on micro-expressions (CASME II, SAMM)
- Combined training on both datasets
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from affective_intelligence.models import EmotionNet, EmotionNetConfig
from affective_intelligence.datasets import (
    FER2013Dataset,
    CASMEIIDataset,
    SAMMDataset,
    get_train_transforms,
    get_val_transforms,
)
from affective_intelligence.losses import CombinedEmotionLoss


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionRecognitionTrainer:
    """Trainer for emotion recognition model."""
    
    def __init__(
        self,
        model: EmotionNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        criterion: nn.Module,
        device: str = "cuda",
        output_dir: str = "./checkpoints",
    ):
        """Initialize trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float("inf")
        self.best_macro_acc = 0.0
        self.best_micro_acc = 0.0
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        macro_correct = 0
        micro_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            # Separate macro and micro labels
            # Assume labels are stacked: [macro_label, micro_label]
            if isinstance(labels, (tuple, list)):
                macro_labels, micro_labels = labels[0], labels[1]
            else:
                # Single label - use for both
                macro_labels = micro_labels = labels
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(images, return_embeddings=True)
            
            # Compute loss
            loss_dict = self.criterion(
                output["macro_logits"],
                output["micro_logits"],
                macro_labels,
                micro_labels,
                output["confidence"],
                output["embeddings"],
            )
            
            loss = loss_dict["total_loss"]
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            macro_preds = torch.argmax(output["macro_logits"], dim=1)
            micro_preds = torch.argmax(output["micro_logits"], dim=1)
            
            macro_correct += (macro_preds == macro_labels).sum().item()
            micro_correct += (micro_preds == micro_labels).sum().item()
            total_samples += images.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": total_loss / (batch_idx + 1),
                "macro_acc": macro_correct / total_samples,
                "micro_acc": micro_correct / total_samples,
            })
        
        return {
            "loss": total_loss / len(self.train_loader),
            "macro_acc": macro_correct / total_samples,
            "micro_acc": micro_correct / total_samples,
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        macro_correct = 0
        micro_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            if isinstance(labels, (tuple, list)):
                macro_labels, micro_labels = labels[0], labels[1]
            else:
                macro_labels = micro_labels = labels
            
            # Forward pass
            output = self.model(images, return_embeddings=True)
            
            # Compute loss
            loss_dict = self.criterion(
                output["macro_logits"],
                output["micro_logits"],
                macro_labels,
                micro_labels,
                output["confidence"],
                output["embeddings"],
            )
            
            loss = loss_dict["total_loss"]
            total_loss += loss.item()
            
            # Metrics
            macro_preds = torch.argmax(output["macro_logits"], dim=1)
            micro_preds = torch.argmax(output["micro_logits"], dim=1)
            
            macro_correct += (macro_preds == macro_labels).sum().item()
            micro_correct += (micro_preds == micro_labels).sum().item()
            total_samples += images.size(0)
        
        return {
            "loss": total_loss / len(self.val_loader),
            "macro_acc": macro_correct / total_samples,
            "micro_acc": micro_correct / total_samples,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_macro_acc": self.best_macro_acc,
            "best_micro_acc": self.best_micro_acc,
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def train(self, num_epochs: int):
        """Train model for specified number of epochs."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Macro Acc: {train_metrics['macro_acc']:.4f}, "
                       f"Micro Acc: {train_metrics['micro_acc']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Macro Acc: {val_metrics['macro_acc']:.4f}, "
                       f"Micro Acc: {val_metrics['micro_acc']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.best_macro_acc = max(self.best_macro_acc, val_metrics["macro_acc"])
                self.best_micro_acc = max(self.best_micro_acc, val_metrics["micro_acc"])
            
            self.save_checkpoint(epoch, is_best)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train emotion recognition model")
    
    # Data arguments
    parser.add_argument("--fer2013_path", type=str, help="Path to FER2013 dataset")
    parser.add_argument("--casme_path", type=str, help="Path to CASME II dataset")
    parser.add_argument("--samm_path", type=str, help="Path to SAMM dataset")
    
    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--use_attention", action="store_true", default=True)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create model
    config = EmotionNetConfig(
        embedding_dim=args.embedding_dim,
        dropout_rate=args.dropout_rate,
        use_attention=args.use_attention,
    )
    model = EmotionNet(config).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load datasets
    datasets = []
    
    if args.fer2013_path:
        logger.info(f"Loading FER2013 from {args.fer2013_path}")
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        
        fer_train = FER2013Dataset(args.fer2013_path, split="train", transform=train_transform)
        fer_val = FER2013Dataset(args.fer2013_path, split="test", transform=val_transform)
        datasets.append((fer_train, fer_val))
    
    if args.casme_path:
        logger.info(f"Loading CASME II from {args.casme_path}")
        casme_train = CASMEIIDataset(args.casme_path, split="train")
        casme_val = CASMEIIDataset(args.casme_path, split="test")
        datasets.append((casme_train, casme_val))
    
    if args.samm_path:
        logger.info(f"Loading SAMM from {args.samm_path}")
        samm_train = SAMMDataset(args.samm_path)
        # For demo, use 80-20 split
        train_size = int(0.8 * len(samm_train))
        from torch.utils.data import random_split
        samm_train, samm_val = random_split(samm_train, [train_size, len(samm_train) - train_size])
        datasets.append((samm_train, samm_val))
    
    if not datasets:
        raise ValueError("No datasets provided! Use --fer2013_path, --casme_path, or --samm_path")
    
    # Concatenate datasets
    train_ds = ConcatDataset([d[0] for d in datasets])
    val_ds = ConcatDataset([d[1] for d in datasets])
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )
    
    # Loss function
    criterion = CombinedEmotionLoss(
        num_macro_classes=config.num_macro_classes,
        num_micro_classes=config.num_micro_classes,
        embedding_dim=config.embedding_dim,
    )
    
    # Trainer
    trainer = EmotionRecognitionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=str(device),
        output_dir=args.output_dir,
    )
    
    # Train
    trainer.train(args.epochs)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
