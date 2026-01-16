import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridMentalHealthModel(nn.Module):
    """
    Hybrid CNN-LSTM model for mental health assessment from handwriting
    Combines spatial features (CNN) with temporal features (LSTM)
    """
    
    def __init__(self, num_classes=3, image_size=(128, 128), 
                 sequence_length=50, num_features=10):
        super().__init__()
        
        # CNN for image feature extraction
        self.cnn = CNNFeatureExtractor()
        cnn_output_size = 256  # Based on CNN architecture
        
        # LSTM for temporal feature extraction
        self.lstm = LSTMModule(input_size=num_features, hidden_size=128)
        lstm_output_size = 128 * 2  # bidirectional
        
        # Combined feature size
        self.cnn_output_size = cnn_output_size
        self.lstm_output_size = lstm_output_size

        
        # Classifier
        # CNN-only classifier (image-only training)
        self.cnn_classifier = nn.Sequential(
            nn.Linear(self.cnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )               

        # Hybrid classifier (CNN + LSTM)
        self.hybrid_classifier = nn.Sequential(
            nn.Linear(self.cnn_output_size + self.lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=cnn_output_size, num_heads=4)
        
    def forward(self, images, sequences=None):
        # Extract spatial features from images
        spatial_features = self.cnn(images)  # [batch_size, cnn_output_size]
        
        if sequences is not None:
            # Extract temporal features from sequences
            temporal_features = self.lstm(sequences)  # [batch_size, lstm_output_size]
            
            # Combine features
            combined = torch.cat([spatial_features, temporal_features], dim=1)
        else:
            # Use only spatial features
            combined = spatial_features
        
        # Apply attention
        spatial_reshaped = spatial_features.unsqueeze(0)  # [1, batch_size, features]
        attn_output, _ = self.attention(spatial_reshaped, spatial_reshaped, spatial_reshaped)
        attended_features = attn_output.squeeze(0)
        
        # Final classification
        if sequences is not None:
            output = self.hybrid_classifier(combined)
        else:
            output = self.cnn_classifier(spatial_features)
        return output

class CNNFeatureExtractor(nn.Module):
    """CNN for extracting spatial features from handwriting images"""
    
    def __init__(self):
        super().__init__()
        
        # Feature extraction layers
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for feature refinement
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
    
    def forward(self, x):
        # Input shape: [batch_size, 1, height, width]
        features = self.conv_layers(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc_layers(features)
        
        return features

class LSTMModule(nn.Module):
    """LSTM for extracting temporal features from stroke sequences"""
    
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)
        
    def forward(self, x):
        # Input shape: [batch_size, seq_length, features]
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_features = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Also get last hidden states
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Combine attended features and last hidden state
        combined_features = torch.cat([attended_features, hidden], dim=1)
        
        return combined_features

class MultiModalMentalHealthModel(nn.Module):
    """
    Multi-modal model combining image, stroke, and metadata features
    """
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Image branch
        self.image_branch = CNNFeatureExtractor()
        image_features = 256
        
        # Stroke features branch (MLP for extracted features)
        self.stroke_branch = nn.Sequential(
            nn.Linear(20, 64),  # 20 stroke features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        stroke_features = 32
        
        # Metadata branch (age, gender, etc.)
        self.metadata_branch = nn.Sequential(
            nn.Linear(5, 16),  # 5 metadata features
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        metadata_features = 8
        
        # Combined classifier
        total_features = image_features + stroke_features + metadata_features
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Output layer for stress score regression
        self.stress_regressor = nn.Sequential(
            nn.Linear(total_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0-1
        )
    
    def forward(self, image, stroke_features, metadata):
        image_feats = self.image_branch(image)
        stroke_feats = self.stroke_branch(stroke_features)
        metadata_feats = self.metadata_branch(metadata)
        
        combined = torch.cat([image_feats, stroke_feats, metadata_feats], dim=1)
        
        classification = self.classifier(combined)
        stress_score = self.stress_regressor(combined)
        
        return classification, stress_score