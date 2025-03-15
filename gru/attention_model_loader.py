"""
Special loader for attention-based weather text generation models.
This script provides a modified AttentionWeatherGRU implementation that matches
the checkpoint structure of the pre-trained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class CompatibleAttentionWeatherGRU(nn.Module):
    """
    AttentionWeatherGRU implementation that matches the structure of the pre-trained checkpoints.
    
    This model uses flat GRU structures instead of nested ones and simplifies the output layer
    to match the checkpoint structure.
    """
    def __init__(self, feature_dim, hidden_size, vocab_size, dropout=0.2):
        super(CompatibleAttentionWeatherGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Feature processing - separate projections for different feature types
        self.temp_proj = nn.Linear(1, hidden_size // 4)  # Temperature features
        self.humidity_proj = nn.Linear(1, hidden_size // 4)  # Humidity features
        self.cloud_proj = nn.Linear(1, hidden_size // 4)  # Cloudiness features
        self.time_proj = nn.Linear(2, hidden_size // 4)  # Cyclical time features (sin/cos)
        
        # For remaining features
        self.other_feat_size = feature_dim - 5  # Subtract temp, humidity, cloud, and time (2 features)
        self.other_proj = nn.Linear(self.other_feat_size, hidden_size)
        
        # Feature integration layer
        self.feature_integrator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )
        
        # Encoder GRU - FLAT STRUCTURE to match checkpoint
        self.encoder_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Token embedding for decoder
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Attention mechanism
        self.attention_projection = nn.Linear(hidden_size, hidden_size)
        
        # Decoder GRU - FLAT STRUCTURE to match checkpoint
        self.decoder_gru = nn.GRU(
            input_size=hidden_size * 2,  # Embedding + context
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Output projection - SIMPLE STRUCTURE to match checkpoint
        self.output = nn.Linear(hidden_size * 2, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, target_tokens=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for training the model.
        
        Args:
            features (torch.Tensor): Input features [batch_size, seq_len, feature_dim]
            target_tokens (torch.Tensor, optional): Target token indices
            teacher_forcing_ratio (float): Probability of using teacher forcing
            
        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size = features.size(0)
        seq_len = features.size(1)
        
        # Process features
        temp_features = features[:, :, 0].unsqueeze(-1)  # Temperature
        humidity_features = features[:, :, 1].unsqueeze(-1)  # Humidity
        cloud_features = features[:, :, 2].unsqueeze(-1)  # Cloudiness
        time_features = features[:, :, -5:-3]  # Time features
        
        # Project features
        temp_proj = self.temp_proj(temp_features)
        humidity_proj = self.humidity_proj(humidity_features)
        cloud_proj = self.cloud_proj(cloud_features)
        time_proj = self.time_proj(time_features)
        
        # Process remaining features
        other_features = torch.cat([
            features[:, :, 3:-5],  
            features[:, :, -3:]    
        ], dim=2)
        other_proj = self.other_proj(other_features)
        
        # Concatenate all projected features
        projected_features_concat = torch.cat(
            [temp_proj, humidity_proj, cloud_proj, time_proj, other_proj], dim=2
        )
        
        # Integrate features
        projected_features = self.feature_integrator(projected_features_concat)
        projected_features = self.dropout(projected_features)
        
        # Encode features
        encoder_outputs, encoder_hidden = self.encoder_gru(projected_features)
        
        # Initialize decoder hidden state with encoder final state
        decoder_hidden = encoder_hidden
        
        # Initialize first decoder input
        if target_tokens is not None:
            # Use the first token (CLS) from target for training
            decoder_input = target_tokens[:, 0].unsqueeze(1)
            max_length = target_tokens.size(1)
        else:
            # For inference, start with first token ID
            start_token_id = 0  
            decoder_input = torch.full((batch_size, 1), start_token_id, device=features.device)
            max_length = 100  # Max length for inference
        
        # Output tensor to store predictions
        outputs = torch.zeros(batch_size, max_length, self.vocab_size, device=features.device)
        
        # Initialize first token probability
        outputs[:, 0, decoder_input.squeeze(1)[0]] = 1.0
        
        # Generate sequence token by token
        for t in range(1, max_length):
            # Embed current input token
            embedded = self.dropout(self.embedding(decoder_input))
            
            # Calculate attention scores using scaled dot-product attention
            query = decoder_hidden.transpose(0, 1)  # [batch_size, 1, hidden_size]
            
            # Project encoder outputs as keys
            keys = self.attention_projection(encoder_outputs)  # [batch_size, seq_len, hidden_size]
            
            # Calculate attention scores
            attn_scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(self.hidden_size)
            
            # Apply softmax to get attention weights
            attn_weights = F.softmax(attn_scores, dim=2)  # [batch_size, 1, seq_len]
            
            # Apply attention weights to encoder outputs
            context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_size]
            
            # Combine embedding and context for decoder input
            rnn_input = torch.cat((embedded, context), dim=2)
            
            # Run through decoder GRU
            decoder_output, decoder_hidden = self.decoder_gru(rnn_input, decoder_hidden)
            
            # Combine decoder output and context for prediction
            combined = torch.cat((decoder_output.squeeze(1), context.squeeze(1)), dim=1)
            
            # Project to vocabulary
            prediction = self.output(combined)
            
            # Store prediction
            outputs[:, t] = prediction
            
            # Teacher forcing decision
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if target_tokens is not None and t < max_length - 1 and use_teacher_forcing:
                decoder_input = target_tokens[:, t].unsqueeze(1)
            else:
                # Use predicted token
                top_token = prediction.argmax(1).unsqueeze(1)
                decoder_input = top_token
        
        return outputs

def load_compatible_attention_model(checkpoint_path, feature_dim, device='cpu'):
    """
    Load an attention model with compatibility for the old checkpoint structure.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        feature_dim (int): Input feature dimension
        device (str): Device to load the model to
        
    Returns:
        tuple: (model, token_mappings)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get token mappings
    token_mappings = checkpoint['token_mappings']
    vocab_size = len(token_mappings['used_token_ids'])
    
    # Get model config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        model_config = {}
    
    # Detect dimensions from state dict
    state_dict = checkpoint['model_state_dict']
    
    # Find hidden dimension (check one of the projection layers)
    hidden_dim = model_config.get('hidden_dim')
    if not hidden_dim:
        if 'encoder_gru.weight_ih_l0' in state_dict:
            # For GRU, hidden size is output dimension / 3 (for the 3 gates)
            hidden_dim = state_dict['encoder_gru.weight_ih_l0'].shape[0] // 3
        elif 'embedding.weight' in state_dict:
            hidden_dim = state_dict['embedding.weight'].shape[1]
        else:
            hidden_dim = 256  # Default fallback
    
    print(f"Loading attention model with hidden_dim={hidden_dim}, vocab_size={vocab_size}")
    
    # Create compatible model
    model = CompatibleAttentionWeatherGRU(
        feature_dim=feature_dim,
        hidden_size=hidden_dim,
        vocab_size=vocab_size
    )
    
    # Load state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, token_mappings