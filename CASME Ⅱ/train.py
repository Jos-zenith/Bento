# ============================================================================
# CASME II Training Script - 3D-CNN-LSTM with Transfer Learning
# ============================================================================

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, cast
import logging

from config import TrainConfig, Config
from dataset import create_dataloaders, CASME2Dataset
from models import build_model
from torch.utils.tensorboard import SummaryWriter


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Training pipeline for CASME II micro-expression classification"""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = config.DEVICE
        # If running on CPU, reduce DataLoader and training pressure
        if self.device.type == 'cpu':
            if getattr(self.config, 'PIN_MEMORY', False):
                logger.info("No CUDA detected — disabling pin_memory for DataLoader")
                self.config.PIN_MEMORY = False
            if getattr(self.config, 'NUM_WORKERS', 0) > 0:
                logger.info("Running on CPU — setting num_workers=0 to avoid worker overhead")
                self.config.NUM_WORKERS = 0
            if getattr(self.config, 'BATCH_SIZE', 8) > 2:
                logger.info("Running on CPU — reducing batch size to 2 for faster iterations")
                self.config.BATCH_SIZE = 2
        self.start_time = datetime.now()
        
        # Create directories
        self._setup_directories()
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        
        # Build dataset
        logger.info("Building dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            root_dir=config.get_data_root(config.DATA_STAGE),
            labels_file=config.LABELS_FILE,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            data_stage=config.DATA_STAGE,
            frame_size=config.FRAME_SIZE,
            temporal_length=config.TEMPORAL_LENGTH,
            augmentation_params=config.AUGMENTATION_PARAMS,
            pin_memory=config.PIN_MEMORY,
        )
        
        logger.info(f"Train samples: {len(cast(CASME2Dataset, self.train_loader.dataset))}")
        logger.info(f"Val samples: {len(cast(CASME2Dataset, self.val_loader.dataset))}")
        logger.info(f"Test samples: {len(cast(CASME2Dataset, self.test_loader.dataset))}")
        
        # Build model
        logger.info(f"Building model: {config.MODEL_TYPE}...")
        self.model = build_model(
            model_type=config.MODEL_TYPE,
            num_classes=config.NUM_CLASSES,
            temporal_length=config.TEMPORAL_LENGTH,
            input_channels=config.MODEL_INPUT_CHANNELS,  # RGB + Combined Optical Flow
            pretrained_backbone=config.PRETRAINED_BACKBONE,
            freeze_backbone=config.FREEZE_BACKBONE_EPOCHS > 0,
            device=self.device,
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(
            weight=config.CLASS_WEIGHTS.to(self.device)
        )
        
        # Setup optimizer with backbone fine-tuning
        self._setup_optimizer()
        
        # Setup learning rate scheduler
        self._setup_scheduler()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.global_step = 0
    
    def _setup_directories(self):
        """Create output directories"""
        Path(self.config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for backbone vs. head"""
        
        # Identify backbone parameters
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'spatial_encoder' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Create param groups with different learning rates
        param_groups = [
            {'params': head_params, 'lr': self.config.LEARNING_RATE},
            {'params': backbone_params, 'lr': self.config.LEARNING_RATE * self.config.FINETUNE_LR_RATIO}
        ]
        
        if self.config.OPTIMIZER == "adamw":
            self.optimizer = optim.AdamW(
                param_groups,
                betas=(0.9, 0.999),
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == "adam":
            self.optimizer = optim.Adam(param_groups, weight_decay=self.config.WEIGHT_DECAY)
        elif self.config.OPTIMIZER == "sgd":
            self.optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=self.config.WEIGHT_DECAY)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
        
        logger.info(f"Optimizer: {self.config.OPTIMIZER}")
        logger.info(f"Backbone LR: {self.config.LEARNING_RATE * self.config.FINETUNE_LR_RATIO}")
        logger.info(f"Head LR: {self.config.LEARNING_RATE}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config.SCHEDULER == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.NUM_EPOCHS - self.config.WARMUP_EPOCHS,
            )
        elif self.config.SCHEDULER == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        elif self.config.SCHEDULER == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.WARMUP_EPOCHS
            )
        else:
            self.scheduler = None
    
    def _unfreeze_backbone(self, epoch: int):
        """Unfreeze backbone parameters after warmup"""
        if epoch == self.config.FREEZE_BACKBONE_EPOCHS:
            logger.info("Unfreezing backbone parameters...")
            for name, param in self.model.named_parameters():
                if 'spatial_encoder' in name:
                    param.requires_grad = True
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train one epoch.
        
        Returns:
            loss: Average training loss
            acc: Training accuracy
        """
        self.model.train()
        self._unfreeze_backbone(epoch)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (frames, labels) in enumerate(self.train_loader):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(frames)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Logging
            if batch_idx % self.config.LOG_INTERVAL == 0:
                acc = 100. * correct / total
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}, Acc: {acc:.2f}%"
                )
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/acc', acc, self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def val_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Validate one epoch.
        
        Returns:
            loss: Average validation loss
            acc: Validation accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, labels in self.val_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(frames)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
        # Clear GPU cache to prevent memory fragmentation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return avg_loss, avg_acc
    
    def test(self) -> Tuple[float, float]:
        """
        Test the model on test set.
        
        Returns:
            loss: Average test loss
            acc: Test accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, labels in self.test_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(frames)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config.__dict__,
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.CHECKPOINT_DIR, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.config.BEST_MODEL_PATH
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
    
    def train(self):
        """Full training loop"""
        logger.info("=" * 50)
        logger.info("Starting training...")
        logger.info("=" * 50)
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.val_epoch(epoch)
            
            # Learning rate scheduling
            if epoch >= self.config.WARMUP_EPOCHS and self.scheduler:
                self.scheduler.step()
            
            # Logging with memory info
            mem_info = ""
            if self.device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                mem_info = f" | GPU Mem: {mem_allocated:.2f}/{mem_reserved:.2f}GB"
            
            logger.info(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%{mem_info}"
            )
            
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/val_acc', val_acc, epoch)
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.config.SAVE_FREQ == 0:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= 15:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Test
        logger.info("=" * 50)
        logger.info("Testing on test set...")
        test_loss, test_acc = self.test()
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save final model
        self.save_checkpoint(self.config.NUM_EPOCHS - 1)
        
        # Summary
        elapsed = datetime.now() - self.start_time
        logger.info("=" * 50)
        logger.info("Training completed!")
        logger.info(f"Best Val Acc: {self.best_val_acc:.2f}%")
        logger.info(f"Test Acc: {test_acc:.2f}%")
        logger.info(f"Elapsed time: {elapsed}")
        logger.info("=" * 50)
        
        self.writer.close()


if __name__ == "__main__":
    # Setup config
    config = TrainConfig()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()
