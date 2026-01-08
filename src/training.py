import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime

from models.hybrid_model import HybridMentalHealthModel

class MentalHealthTrainer:
    def __init__(self, model, device='cuda', model_dir='saved_models'):
        self.model = model.to(device)
        self.device = device
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train(self, train_loader, val_loader, num_epochs=50, 
              learning_rate=0.001, weight_decay=1e-4):
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                               weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            
            for batch_idx, (images, labels) in enumerate(train_bar):
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(self.model, HybridMentalHealthModel):
                    # For hybrid model, we only have images in this example
                    outputs, _ = self.model(images)
                else:
                    outputs = self.model(images)
                # -------- FORCE LABEL REMAPPING (FINAL SAFETY FIX) --------
                # Convert EMNIST labels (10–61) → 0,1,2
                labels = labels.clone()

                labels[labels <= 20] = 0
                labels[(labels > 20) & (labels <= 40)] = 1
                labels[labels > 40] = 2
                labels = labels.long()
                # ---------------------------------------------------------

                
                loss = self.classification_criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_loss += loss.item()
                
                # Update progress bar
                train_bar.set_postfix({
                    'Loss': loss.item(),
                    'Acc': 100. * train_correct / train_total
                })
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            # Update metrics
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(f'best_model.pth')
                print(f"  ✓ Saved best model (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping triggered")
                    break
        
        # Save final model
        self.save_model(f'final_model_epoch{num_epochs}.pth')
        
        # Plot training history
        self.plot_training_history()
        
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                if isinstance(self.model, HybridMentalHealthModel):
                    outputs, _ = self.model(images)
                else:
                    outputs = self.model(images)
                
                loss = self.classification_criterion(outputs, labels)
                val_loss += loss.item()
                
                # -------- FORCE LABEL REMAPPING (VALIDATION FIX) --------
                labels = labels.clone()

                labels[labels <= 20] = 0
                labels[(labels > 20) & (labels <= 40)] = 1
                labels[labels > 40] = 2
                labels = labels.long()
                # ------------------------------------------------------

                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Store for metrics calculation
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Calculate additional metrics
        metrics = self.calculate_metrics(
            np.array(all_labels), 
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        return avg_val_loss, val_acc, metrics
    
    def calculate_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Classification report
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # ROC-AUC for multi-class
        if len(np.unique(true_labels)) > 2:
            try:
                auc_score = roc_auc_score(true_labels, probabilities, multi_class='ovr')
                metrics['auc_roc'] = auc_score
            except:
                metrics['auc_roc'] = 0.0
        
        # Store metrics
        metrics['precision'] = report['weighted avg']['precision']
        metrics['recall'] = report['weighted avg']['recall']
        metrics['f1_score'] = report['weighted avg']['f1-score']
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracies
        axes[1].plot(self.train_accs, label='Train Acc', linewidth=2)
        axes[1].plot(self.val_accs, label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'), dpi=150)
        plt.show()
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        torch.save(checkpoint, os.path.join(self.model_dir, filename))
        print(f"Model saved to {os.path.join(self.model_dir, filename)}")
    
    def load_model(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(self.model_dir, filename), 
                               map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accs = checkpoint['train_accs']
            self.val_accs = checkpoint['val_accs']
        
        print(f"Model loaded from {os.path.join(self.model_dir, filename)}")

class CrossValidationTrainer:
    """
    Trainer for cross-validation experiments
    """
    
    def __init__(self, model_class, device='cuda', n_folds=5):
        self.model_class = model_class
        self.device = device
        self.n_folds = n_folds
        self.fold_results = []
        
    def cross_validate(self, npz_path, batch_size=32, num_epochs=30, **kwargs):
        """Perform k-fold cross validation"""
        from sklearn.model_selection import KFold
        
        # Load all data
        data_dict = np.load(npz_path, allow_pickle=True)
        images = data_dict['images']
        labels = data_dict['labels']
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{self.n_folds}")
            print(f"{'='*50}")
            
            # Split data
            X_train, X_val = images[train_idx], images[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Create datasets and loaders
            train_dataset = self._create_dataset(X_train, y_train)
            val_dataset = self._create_dataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            model = self.model_class(**kwargs)
            trainer = MentalHealthTrainer(model, device=self.device)
            
            # Train
            trainer.train(train_loader, val_loader, num_epochs=num_epochs)
            
            # Evaluate
            val_loss, val_acc, metrics = trainer.validate(val_loader)
            
            fold_metrics.append({
                'fold': fold + 1,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'auc_roc': metrics.get('auc_roc', 0)
            })
            
            self.fold_results.append(fold_metrics[-1])
            
            print(f"Fold {fold + 1} Results:")
            for key, value in fold_metrics[-1].items():
                if key != 'fold':
                    print(f"  {key}: {value:.4f}")
        
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(fold_metrics)
        
        print(f"\n{'='*50}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*50}")
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")
        
        return avg_metrics
    
    def _create_dataset(self, images, labels):
        """Create PyTorch dataset from numpy arrays"""
        class FoldDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels
                
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                image = self.images[idx]
                label = self.labels[idx]
                
                # Normalize
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                
                # Add channel dimension if needed
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=0)
                elif len(image.shape) == 3 and image.shape[-1] == 1:
                    image = image.transpose(2, 0, 1)
                
                return torch.FloatTensor(image), torch.LongTensor([label])
        
        return FoldDataset(images, labels)
    
    def _calculate_average_metrics(self, fold_metrics):
        """Calculate average metrics across all folds"""
        avg_metrics = {}
        
        # Initialize sums
        metrics_sum = {key: 0 for key in fold_metrics[0].keys() if key != 'fold'}
        
        # Sum across folds
        for metrics in fold_metrics:
            for key in metrics_sum.keys():
                metrics_sum[key] += metrics[key]
        
        # Calculate averages
        for key, total in metrics_sum.items():
            avg_metrics[f'avg_{key}'] = total / len(fold_metrics)
        
        # Calculate standard deviations
        for key in metrics_sum.keys():
            values = [metrics[key] for metrics in fold_metrics]
            avg_metrics[f'std_{key}'] = np.std(values)
        
        return avg_metrics