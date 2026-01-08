import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RNNMentalHealthModel(nn.Module):
    """RNN model for sequential handwriting analysis with multiple RNN types"""
    
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, 
                 num_classes=3, rnn_type='lstm', bidirectional=True, 
                 dropout=0.3, attention=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.attention = attention
        
        # RNN layer selection
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type.lower() == 'rnn':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
                nonlinearity='tanh',
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Attention mechanism
        if attention:
            self.attention_layer = BahdanauAttention(
                hidden_size * 2 if bidirectional else hidden_size
            )
        
        # Calculate FC input size
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        if attention:
            fc_input_size = fc_input_size * 2  # Context + last hidden
        
        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
        
        # Regression branch for stress score
        self.stress_regressor = nn.Sequential(
            nn.Linear(fc_input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Feature extraction for detailed analysis
        self.feature_extractor = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # 16 interpretable features
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input sequence [batch_size, seq_len, features]
            return_attention: Whether to return attention weights
        Returns:
            classification, stress_score, features, (attention_weights)
        """
        batch_size = x.size(0)
        
        # Initialize hidden states
        if self.rnn_type.lower() == 'lstm':
            h0 = torch.zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                batch_size, self.hidden_size
            ).to(x.device)
            c0 = torch.zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                batch_size, self.hidden_size
            ).to(x.device)
            hidden = (h0, c0)
        else:
            hidden = torch.zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                batch_size, self.hidden_size
            ).to(x.device)
        
        # RNN forward pass
        rnn_out, hidden_final = self.rnn(x, hidden)
        
        # Get final hidden states
        if self.rnn_type.lower() == 'lstm':
            hidden_states, cell_states = hidden_final
            if self.bidirectional:
                last_hidden = torch.cat([hidden_states[-2], hidden_states[-1]], dim=1)
            else:
                last_hidden = hidden_states[-1]
        else:
            if self.bidirectional:
                last_hidden = torch.cat([hidden_final[-2], hidden_final[-1]], dim=1)
            else:
                last_hidden = hidden_final[-1]
        
        # Apply attention if enabled
        if self.attention:
            context_vector, attention_weights = self.attention_layer(rnn_out, last_hidden.unsqueeze(1))
            context_vector = context_vector.squeeze(1)
            
            # Combine context vector with last hidden state
            combined_features = torch.cat([context_vector, last_hidden], dim=1)
        else:
            # Use last output
            combined_features = rnn_out[:, -1, :]
            attention_weights = None
        
        # Classification
        classification = self.fc_layers(combined_features)
        
        # Stress score regression (0-1)
        stress_score = self.stress_regressor(combined_features)
        
        # Extract interpretable features
        features = self.feature_extractor(combined_features)
        
        if return_attention:
            return classification, stress_score, features, attention_weights
        else:
            return classification, stress_score, features

class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism"""
    
    def __init__(self, hidden_size):
        super().__init__()
        
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        
    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: [batch_size, seq_len, hidden_size]
            decoder_hidden: [batch_size, 1, hidden_size]
        """
        # Add time dimension to decoder hidden
        decoder_hidden = decoder_hidden.transpose(0, 1)  # [1, batch_size, hidden_size]
        
        # Calculate attention scores
        scores = self.v(torch.tanh(
            self.W(encoder_outputs) + self.U(decoder_hidden)
        ))  # [batch_size, seq_len, 1]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Calculate context vector
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)
        
        return context_vector.unsqueeze(1), attention_weights.squeeze(-1)

class BidirectionalRNNWithFeatures(nn.Module):
    """Bidirectional RNN with explicit feature extraction"""
    
    def __init__(self, input_size, feature_sizes, num_classes=3):
        """
        Args:
            input_size: Size of input features
            feature_sizes: List of sizes for each feature type
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Feature-specific RNNs
        self.feature_rnns = nn.ModuleList()
        current_start = 0
        self.feature_indices = []
        
        for i, feat_size in enumerate(feature_sizes):
            feature_rnn = nn.LSTM(
                input_size=feat_size,
                hidden_size=64,
                bidirectional=True,
                batch_first=True
            )
            self.feature_rnns.append(feature_rnn)
            self.feature_indices.append((current_start, current_start + feat_size))
            current_start += feat_size
        
        # Combined RNN
        total_rnn_features = len(feature_sizes) * 64 * 2  # bidirectional
        self.combined_rnn = nn.LSTM(
            input_size=total_rnn_features,
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 128 * 2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Feature importance predictors
        self.feature_importance = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 32),  # 64 * 2 for bidirectional
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(len(feature_sizes))
        ])
    
    def forward(self, x):
        """Forward pass with feature decomposition"""
        batch_size, seq_len, _ = x.shape
        
        # Process each feature type separately
        feature_outputs = []
        feature_importances = []
        
        for i, (start_idx, end_idx) in enumerate(self.feature_indices):
            # Extract feature subset
            feature_data = x[:, :, start_idx:end_idx]
            
            # Process through feature-specific RNN
            feature_out, _ = self.feature_rnns[i](feature_data)
            feature_outputs.append(feature_out[:, -1, :])  # Take last output
            
            # Predict feature importance
            importance = self.feature_importance[i](feature_out[:, -1, :])
            feature_importances.append(importance)
        
        # Concatenate all feature outputs
        combined_features = torch.cat(feature_outputs, dim=1).unsqueeze(1)
        
        # Process through combined RNN
        combined_out, _ = self.combined_rnn(combined_features)
        
        # Classification
        output = self.classifier(combined_out[:, -1, :])
        
        # Combine feature importances
        feature_importances = torch.cat(feature_importances, dim=1)
        
        return output, feature_importances

class TemporalAttentionModel(nn.Module):
    """Multi-head attention with temporal dependencies"""
    
    def __init__(self, input_size=10, num_heads=4, num_classes=3, num_layers=3):
        super().__init__()
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(input_size)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # Temporal feature extractor
        self.temporal_features = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # 8 temporal features
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, features]
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Global temporal pooling
        encoded = encoded.transpose(1, 2)  # [batch, features, seq]
        pooled = self.global_pool(encoded).squeeze(-1)
        
        # Classification
        classification = self.classifier(pooled)
        
        # Extract temporal features
        temp_features = self.temporal_features(pooled)
        
        return classification, temp_features

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x

class StackedRNNEnsemble(nn.Module):
    """Ensemble of different RNN architectures"""
    
    def __init__(self, input_size=10, num_classes=3):
        super().__init__()
        
        # Different RNN architectures
        self.lstm_model = RNNMentalHealthModel(
            input_size=input_size,
            rnn_type='lstm',
            num_classes=num_classes
        )
        
        self.gru_model = RNNMentalHealthModel(
            input_size=input_size,
            rnn_type='gru',
            num_classes=num_classes
        )
        
        self.attention_model = TemporalAttentionModel(
            input_size=input_size,
            num_classes=num_classes
        )
        
        # Ensemble combiner
        self.combiner = nn.Sequential(
            nn.Linear(num_classes * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through ensemble"""
        # Get predictions from each model
        lstm_out, lstm_stress, lstm_feats = self.lstm_model(x)
        gru_out, gru_stress, gru_feats = self.gru_model(x)
        attn_out, attn_feats = self.attention_model(x)
        
        # Combine outputs
        combined = torch.cat([lstm_out, gru_out, attn_out], dim=1)
        final_output = self.combiner(combined)
        
        # Average stress scores
        stress_score = (lstm_stress + gru_stress) / 2
        
        # Combine features
        features = torch.cat([lstm_feats, gru_feats, attn_feats], dim=1)
        
        return final_output, stress_score, features