#!/usr/bin/env python3
"""
Main training script for Mental Health Assessment Model
"""

import argparse
import torch
import numpy as np
from src.data_loader import create_data_loaders, analyze_npz_dataset
from src.models.hybrid_model import HybridMentalHealthModel, MultiModalMentalHealthModel
from src.training import MentalHealthTrainer, CrossValidationTrainer
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Train Mental Health Assessment Model')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to NPZ dataset file or directory')
    parser.add_argument('--model_type', type=str, default='hybrid',
                       choices=['hybrid', 'multimodal', 'cnn', 'lstm'],
                       help='Type of model to train')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--cross_validate', action='store_true',
                       help='Perform k-fold cross validation')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross validation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to train on')
    parser.add_argument('--model_dir', type=str, default='saved_models',
                       help='Directory to save models')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    print(f"Data path: {args.data_path}")
    
    # Analyze dataset
    print("\nAnalyzing dataset...")
    data_dict = analyze_npz_dataset(args.data_path)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=4,
        temporal=(args.model_type == 'lstm')
    )
    
    # Initialize model
    print(f"\nInitializing {args.model_type} model...")
    
    if args.model_type == 'hybrid':
        model = HybridMentalHealthModel(num_classes=3)
    elif args.model_type == 'multimodal':
        model = MultiModalMentalHealthModel(num_classes=3)
    elif args.model_type == 'cnn':
        from src.models.cnn_model import CNNMentalHealthModel
        model = CNNMentalHealthModel(num_classes=3)
    elif args.model_type == 'lstm':
        from src.models.hybrid_model import LSTMMentalHealthModel
        model = LSTMMentalHealthModel(input_size=10, num_classes=3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train or cross-validate
    if args.cross_validate:
        print(f"\nPerforming {args.n_folds}-fold cross validation...")
        cv_trainer = CrossValidationTrainer(
            model_class=type(model),
            device=device,
            n_folds=args.n_folds
        )
        
        results = cv_trainer.cross_validate(
            args.data_path,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )
        
        # Save results
        import json
        with open(f'{args.model_dir}/cv_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
    else:
        print("\nStarting training...")
        trainer = MentalHealthTrainer(model, device=device, model_dir=args.model_dir)
        
        # Train the model
        trainer.train(
            train_loader, 
            val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        # Test the model
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_metrics = trainer.validate(test_loader)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        
        if 'auc_roc' in test_metrics:
            print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
        
        # Save test results
        import json
        test_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_f1': float(test_metrics['f1_score']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_auc_roc': float(test_metrics.get('auc_roc', 0))
        }
        
        with open(f'{args.model_dir}/test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Plot confusion matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = test_metrics['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Mild', 'Severe'],
                   yticklabels=['Normal', 'Mild', 'Severe'])
        plt.title('Confusion Matrix - Test Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{args.model_dir}/confusion_matrix.png', dpi=150)
        plt.show()

if __name__ == '__main__':
    main()