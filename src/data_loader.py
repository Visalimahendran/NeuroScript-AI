import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from sklearn.model_selection import train_test_split

class HandwritingNPZDataset(Dataset):
    """
    Dataset class for NPZ format handwriting data
    Expected NPZ structure:
    - images: array of shape (n_samples, height, width, channels) or (n_samples, height, width)
    - labels: array of shape (n_samples,) with stress levels (0: normal, 1: mild, 2: severe)
    - Optional: strokes, timestamps, pressures for temporal data
    """
    
    def __init__(self, npz_path, transform=None, mode='train', 
                 val_split=0.2, test_split=0.1, random_state=42):
        """
        Args:
            npz_path: Path to NPZ file or directory containing NPZ files
            transform: Optional transform to be applied
            mode: 'train', 'val', or 'test'
        """
        super().__init__()
        self.transform = transform
        self.mode = mode
        
        # Load NPZ data
        if os.path.isdir(npz_path):
            # Load from directory with train/val/test splits
            self.data, self.labels = self._load_from_directory(npz_path, mode)
        else:
            # Load single NPZ file and split
            data_dict = np.load(npz_path, allow_pickle=True)
            self.data, self.labels = self._prepare_data(data_dict, mode, 
                                                       val_split, test_split, 
                                                       random_state)
        
        print(f"Loaded {len(self.data)} samples for {mode} mode")
        
    def _load_from_directory(self, directory, mode):
        """Load from pre-split NPZ files"""
        if mode == 'train':
            file_path = os.path.join(directory, 'train.npz')
        elif mode == 'val':
            file_path = os.path.join(directory, 'val.npz')
        elif mode == 'test':
            file_path = os.path.join(directory, 'test.npz')
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        data_dict = np.load(file_path, allow_pickle=True)
        return data_dict['images'], data_dict['labels']
    
    def _prepare_data(self, data_dict, mode, val_split, test_split, random_state):
        """Prepare data from single NPZ file with splitting"""
        images = data_dict['images']
        labels = data_dict['labels']
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_split, 
            random_state=random_state, stratify=labels
        )
        
        val_ratio = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            random_state=random_state, stratify=y_temp
        )
        
        if mode == 'train':
            return X_train, y_train
        elif mode == 'val':
            return X_val, y_val
        elif mode == 'test':
            return X_test, y_test
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Ensure image is in correct format
        if len(image.shape) == 2:
            # Grayscale image, add channel dimension
            image = np.expand_dims(image, axis=-1)
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            # Change to channel-first format for PyTorch
            image = torch.FloatTensor(image.transpose(2, 0, 1))
        
        return image, torch.LongTensor([label])

class TemporalHandwritingDataset(Dataset):
    """
    Dataset for temporal handwriting data (stroke sequences)
    Expected NPZ structure:
    - sequences: array of shape (n_samples, seq_length, features)
    - labels: array of shape (n_samples,)
    """
    
    def __init__(self, npz_path, transform=None, mode='train', max_seq_length=100):
        super().__init__()
        self.transform = transform
        self.max_seq_length = max_seq_length
        
        data_dict = np.load(npz_path, allow_pickle=True)
        
        # Load sequences and labels
        if 'sequences' in data_dict:
            self.sequences = data_dict['sequences']
        elif 'strokes' in data_dict:
            self.sequences = data_dict['strokes']
        else:
            raise KeyError("NPZ file must contain 'sequences' or 'strokes' key")
        
        self.labels = data_dict['labels']
        
        # Pad sequences to fixed length
        self.sequences = self._pad_sequences(self.sequences)
        
        print(f"Loaded {len(self.sequences)} temporal sequences")
    
    def _pad_sequences(self, sequences):
        """Pad sequences to fixed length"""
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) > self.max_seq_length:
                # Truncate
                padded_seq = seq[:self.max_seq_length]
            else:
                # Pad
                pad_length = self.max_seq_length - len(seq)
                padding = np.zeros((pad_length, seq.shape[1]))
                padded_seq = np.vstack([seq, padding])
            
            padded_sequences.append(padded_seq)
        
        return np.array(padded_sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        raw_label = int(self.labels[idx])
        
        if raw_label <= 20:
            label = 0
        elif raw_label <= 40:
            label = 1 
        else:
            label = 2
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        # Normalize
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        return image, torch.tensor(label, dtype=torch.long)


def create_data_loaders(npz_path, batch_size=32, num_workers=4, 
                        temporal=False, **kwargs):
    """
    Create data loaders for training, validation, and testing
    """
    if temporal:
        DatasetClass = TemporalHandwritingDataset
    else:
        DatasetClass = HandwritingNPZDataset
    
    train_dataset = DatasetClass(npz_path, mode='train', **kwargs)
    val_dataset = DatasetClass(npz_path, mode='val', **kwargs)
    test_dataset = DatasetClass(npz_path, mode='test', **kwargs)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

def analyze_npz_dataset(npz_path):
    """Analyze and display NPZ dataset statistics"""
    data_dict = np.load(npz_path, allow_pickle=True)
    
    print("=" * 50)
    print("NPZ DATASET ANALYSIS")
    print("=" * 50)
    
    print(f"Keys in NPZ file: {list(data_dict.keys())}")
    
    for key in data_dict.keys():
        data = data_dict[key]
        print(f"\n{key}:")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        
        if key == 'images':
            if len(data.shape) == 4:
                print(f"  Format: (samples, height, width, channels)")
                print(f"  Image size: {data.shape[1:3]}")
            elif len(data.shape) == 3:
                print(f"  Format: (samples, height, width)")
        
        elif key == 'labels':
            unique, counts = np.unique(data, return_counts=True)
            print(f"  Classes: {unique}")
            print(f"  Counts: {dict(zip(unique, counts))}")
    
            percentages = counts / len(data) * 100
            label_map = {0: 'Normal', 1: 'Mild', 2: 'Severe'}
    
            print("  Class distribution:")
            for label_id, count, pct in zip(unique, counts, percentages):
                label_name = label_map.get(label_id, f'Class {label_id}')
                print(f"    {label_name}: {pct:.1f}% ({count} samples)")

    return data_dict