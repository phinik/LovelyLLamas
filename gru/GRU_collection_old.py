import torch
import re
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm.notebook import tqdm
import numpy as np
import json
import os
import random

class WeatherGRU(nn.Module):
    def __init__(self, feature_dim, vocab_size, embedding_dim=256, hidden_dim=512, n_layers=2, dropout=0.1):
        super().__init__()
        
        self.timestep_feature_dim = feature_dim

        # Add layer normalization for better stability
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(self.timestep_feature_dim),
            nn.Linear(self.timestep_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Initialize embedding with Xavier/Glorot initialization
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Add gradient clipping to GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(self.timestep_feature_dim),
            nn.Linear(self.timestep_feature_dim, hidden_dim)
        )
        
        # Add layer normalization before final projection
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
    def forward(self, features, tokens, teacher_forcing_ratio=1.0):
        # Add input validation
        if torch.isnan(features).any():
            raise ValueError("NaN detected in input features")
        if torch.isinf(features).any():
            raise ValueError("Inf detected in input features")
            
        batch_size = features.size(0)
        seq_len = features.size(1)
        max_len = tokens.size(1) - 1
        
        # Scale features to prevent extreme values
        features = torch.clamp(features, -10, 10)
        
        features_reshaped = features.view(-1, self.timestep_feature_dim)
        encoded_features = self.feature_encoder(features_reshaped)
        encoded_features = encoded_features.view(batch_size, seq_len, -1)
        
        # Initialize hidden state with scaled features
        h_0 = self.feature_projection(features[:, 0])
        h_0 = torch.tanh(h_0)  # Ensure values are in [-1, 1]
        h_0 = h_0.unsqueeze(0).expand(self.n_layers, batch_size, self.hidden_dim).contiguous()
        
        outputs = torch.zeros(batch_size, max_len, self.output_layer[-1].out_features, device=features.device)
        decoder_input = tokens[:, 0].unsqueeze(1)
        
        for t in range(max_len):
            token_emb = self.embedding(decoder_input)
            current_features = encoded_features[:, min(t, seq_len-1)].unsqueeze(1)
            
            # Scale combined input
            combined_input = (token_emb + current_features) / 2
            
            output, h_0 = self.gru(combined_input, h_0)
            
            # Check for NaN in hidden state
            if torch.isnan(h_0).any():
                raise ValueError(f"NaN detected in hidden state at timestep {t}")
                
            prediction = self.output_layer(output)
            outputs[:, t:t+1] = prediction
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = tokens[:, t+1].unsqueeze(1) if teacher_force else prediction.argmax(dim=-1)
        
        return outputs

    def generate(self, features, max_length=100, token_mappings=None):
        """
        Generate text from features using argmax for deterministic generation.
        
        Args:
            features: Input features tensor [batch_size, seq_len, feature_dim]
            max_length: Maximum length of generated sequence
            token_mappings: Optional token mappings for vocabulary conversion
        
        Returns:
            torch.Tensor: Generated token indices
        """
        self.eval()
        with torch.no_grad():
            batch_size = features.size(0)
            seq_len = features.size(1)
            
            # Scale features
            features = torch.clamp(features, -10, 10)
            
            # Process features
            features_reshaped = features.view(-1, self.timestep_feature_dim)
            encoded_features = self.feature_encoder(features_reshaped)
            encoded_features = encoded_features.view(batch_size, seq_len, -1)
            
            # Initialize hidden state
            h_0 = self.feature_projection(features[:, 0])
            h_0 = torch.tanh(h_0)
            h_0 = h_0.unsqueeze(0).expand(self.n_layers, batch_size, self.hidden_dim).contiguous()
            
            # Start with CLS token (ID 101 for BERT)
            start_token_id = 101  # CLS token ID for BERT
            
            # Use mapped token ID if mappings provided
            if token_mappings and 'token_id_map' in token_mappings:
                start_token_id = token_mappings['token_id_map'].get(101, 0)
            
            # Ensure decoder_input is a 2D tensor
            decoder_input = torch.full(
                (batch_size, 1), 
                start_token_id, 
                dtype=torch.long, 
                device=features.device
            )
            
            generated_tokens = []
            
            for t in range(max_length):
                token_emb = self.embedding(decoder_input)
                current_features = encoded_features[:, min(t, seq_len-1)].unsqueeze(1)
                combined_input = (token_emb + current_features) / 2
                
                output, h_0 = self.gru(combined_input, h_0)
                logits = self.output_layer(output)
                
                # Use argmax for deterministic token selection
                # next_token = logits.argmax(dim=2)
                # Scale logits with very small temperature
                logits = logits.squeeze(1) / 0.1
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated_tokens.append(next_token)
                
                # Stop if we hit the SEP token (ID 102 for BERT)
                sep_token_id = 102
                if token_mappings and 'token_id_map' in token_mappings:
                    sep_token_id = token_mappings['token_id_map'].get(102, 0)
                    
                if (next_token == sep_token_id).any():
                    break
                    
                decoder_input = next_token
            
            # Concatenate generated tokens
            generated_tokens = torch.cat(generated_tokens, dim=1)
            return generated_tokens
        

class WeatherTextGRU(nn.Module):
    def __init__(self, feature_dim, hidden_size, vocab_size, dropout=0.2):
        super(WeatherTextGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Feature processing - separate projections for different feature types
        # This enhances correlation between features and text
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
        
        # Token embedding with positional encoding
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # Encoder GRU with residual connections
        self.encoder_gru = ResidualGRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Decoder GRU with residual connections
        self.decoder_gru = ResidualGRU(
            input_size=hidden_size * 2,  # Combined embedding and feature context
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Context vector generation
        self.context_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # Output projection with layer normalization
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),  # GELU activation often performs better than ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, vocab_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with improved schemes"""
        if isinstance(module, nn.Linear):
            # Slightly adjusted Xavier initialization
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param, gain=1.0)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif "bias" in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, features, target_tokens=None, teacher_forcing_ratio=0.5):
        batch_size = features.size(0)
        seq_len = features.size(1)
        
        # Normalize features by type for better stability
        features = torch.clamp(features, -10, 10)  # Prevent extreme values
        
        # Extract specific feature types - assuming specific positions in the feature tensor
        temp_features = features[:, :, 0].unsqueeze(-1)  # Temperature (first feature)
        humidity_features = features[:, :, 1].unsqueeze(-1)  # Humidity (second feature)
        cloud_features = features[:, :, 2].unsqueeze(-1)  # Cloudiness (third feature)
        time_features = features[:, :, -5:-3]  # Time features (cyclical sin/cos encoding)
        
        # Project each feature type separately
        temp_proj = self.temp_proj(temp_features)
        humidity_proj = self.humidity_proj(humidity_features)
        cloud_proj = self.cloud_proj(cloud_features)
        time_proj = self.time_proj(time_features)
        
        # Process remaining features
        other_features = torch.cat([
            features[:, :, 3:-5],  # Features after cloudiness and before time
            features[:, :, -3:]    # Features after time
        ], dim=2)
        other_proj = self.other_proj(other_features)
        
        # Concatenate all projected features
        projected_features_concat = torch.cat(
            [temp_proj, humidity_proj, cloud_proj, time_proj, other_proj], dim=2
        )
        
        # Integrate features
        projected_features = self.feature_integrator(projected_features_concat)
        
        # Process features through encoder
        encoder_outputs, encoder_hidden = self.encoder_gru(projected_features)
        
        # Generate global context from encoder outputs
        encoder_context = self.context_generator(encoder_outputs.mean(dim=1, keepdim=True))
        
        # Initialize decoder state
        decoder_hidden = encoder_hidden
        
        # Setup for decoder
        if target_tokens is not None:
            # Use the first token (CLS) from target for training
            decoder_input = target_tokens[:, 0].unsqueeze(1)
            max_length = target_tokens.size(1)
        else:
            # For inference, start with CLS token ID
            start_token_id = 0  # This will be mapped to appropriate token
            decoder_input = torch.full((batch_size, 1), start_token_id, device=features.device)
            max_length = 100  # Max length for inference
        
        # Output tensor to store predictions
        outputs = torch.zeros(batch_size, max_length, self.vocab_size, device=features.device)
        
        # Generate sequence token by token with scheduled sampling
        for t in range(1, max_length):
            # Get token embedding with positional encoding
            embedded = self.embedding(decoder_input)
            embedded = self.pos_encoder(embedded)
            
            # Combine with context
            current_time_idx = min(t-1, seq_len-1)
            current_context = encoder_outputs[:, current_time_idx:current_time_idx+1]
            decoder_input_combined = torch.cat([embedded, current_context], dim=2)
            
            # Pass through decoder
            decoder_output, decoder_hidden = self.decoder_gru(decoder_input_combined, decoder_hidden)
            
            # Generate prediction
            prediction = self.output(decoder_output.squeeze(1))
            outputs[:, t] = prediction
            
            # Progressive teacher forcing - reduce probability as sequence progresses
            seq_progress = t / max_length
            current_tf_ratio = teacher_forcing_ratio * (1 - 0.3 * seq_progress)
            use_teacher_forcing = random.random() < current_tf_ratio
            
            # Select next input
            if target_tokens is not None and t < max_length - 1 and use_teacher_forcing:
                decoder_input = target_tokens[:, t].unsqueeze(1)
            else:
                # Use nucleus sampling for more diverse outputs during training
                if self.training:
                    # Apply temperature scaling and nucleus sampling
                    temperature = 0.8
                    logits = prediction / temperature
                    probs = F.softmax(logits, dim=1)
                    
                    # Sort probabilities in descending order
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                    
                    # Calculate cumulative probabilities
                    cumulative_probs = torch.cumsum(sorted_probs, dim=1)
                    
                    # Create mask for nucleus
                    nucleus_mask = cumulative_probs < 0.9  # Top-p sampling with p=0.9
                    
                    # Ensure at least one token is selected
                    nucleus_mask[:, 0] = True
                    
                    # Get the number of tokens to keep for each item in batch
                    sorted_indices_to_remove = nucleus_mask.long().sum(dim=1)
                    
                    # Sample from the nucleus for each item in batch
                    sampled_tokens = []
                    for i in range(batch_size):
                        nucleus_size = sorted_indices_to_remove[i].item()
                        distribution = sorted_probs[i, :nucleus_size]
                        distribution = distribution / distribution.sum()  # Re-normalize
                        item_sorted_indices = sorted_indices[i, :nucleus_size]
                        sampled_idx = torch.multinomial(distribution, 1)
                        sampled_tokens.append(item_sorted_indices[sampled_idx])
                    
                    next_token = torch.cat(sampled_tokens, dim=0).unsqueeze(1)
                    decoder_input = next_token
                else:
                    # Use argmax for deterministic generation during evaluation
                    decoder_input = prediction.argmax(1).unsqueeze(1)
        
        return outputs


# Helper modules

class PositionalEncoding(nn.Module):
    """Positional encoding to provide sequence position information"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ResidualGRU(nn.Module):
    """GRU with residual connections for better gradient flow"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, batch_first=True):
        super(ResidualGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Input projection if input size doesn't match hidden size
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        
        # Create stacked GRU layers with layer normalization and residual connections
        self.gru_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = hidden_size
            self.gru_layers.append(nn.GRU(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=batch_first
            ))
            self.norm_layers.append(nn.LayerNorm(hidden_size))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden=None):
        # Project input if needed
        if self.input_proj is not None:
            output = self.input_proj(input)
        else:
            output = input
        
        # Initialize hidden state if not provided
        if hidden is None:
            batch_size = input.size(0) if self.batch_first else input.size(1)
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=input.device)
        
        # Process through each layer with residual connections
        hidden_states = []
        for i in range(self.num_layers):
            residual = output
            output, layer_hidden = self.gru_layers[i](output, hidden[i:i+1] if hidden is not None else None)
            hidden_states.append(layer_hidden)
            
            # Apply normalization and residual connection
            output = self.norm_layers[i](output + residual)
            output = self.dropout(output)
        
        # Combine hidden states
        combined_hidden = torch.cat(hidden_states, dim=0)
        
        return output, combined_hidden
    
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
import numpy as np
import time
import math
import os
import json
from collections import Counter
# Weather Text Generator Model with GRU and Attention
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class WeatherTextGRU(nn.Module):
    def __init__(self, feature_dim, hidden_size, vocab_size, dropout=0.2):
        super(WeatherTextGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Feature processing - separate projections for different feature types
        # This enhances correlation between features and text
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
        
        # Encoder GRU
        self.encoder_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Token embedding for decoder
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Attention mechanism - using scaled dot-product attention for efficiency
        self.attention_projection = nn.Linear(hidden_size, hidden_size)
        
        # Decoder GRU
        self.decoder_gru = nn.GRU(
            input_size=hidden_size * 2,  # Embedding + context
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Output projection
        self.output = nn.Linear(hidden_size * 2, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, target_tokens=None, teacher_forcing_ratio=0.5):
        batch_size = features.size(0)
        seq_len = features.size(1)
        
        # Process features with enhanced correlation
        # Extract specific feature types - assuming specific positions in the feature tensor
        temp_features = features[:, :, 0].unsqueeze(-1)  # Temperature (first feature)
        humidity_features = features[:, :, 1].unsqueeze(-1)  # Humidity (second feature)
        cloud_features = features[:, :, 2].unsqueeze(-1)  # Cloudiness (third feature)
        time_features = features[:, :, -5:-3]  # Time features (cyclical sin/cos encoding)
        
        # Project each feature type separately
        temp_proj = self.temp_proj(temp_features)
        humidity_proj = self.humidity_proj(humidity_features)
        cloud_proj = self.cloud_proj(cloud_features)
        time_proj = self.time_proj(time_features)
        
        # Process remaining features
        other_features = torch.cat([
            features[:, :, 3:-5],  # Features after cloudiness and before time
            features[:, :, -3:]    # Features after time
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
            # For inference, start with CLS token ID or the first token ID in reduced vocab
            start_token_id = 0  # This will be mapped to the appropriate token
            decoder_input = torch.full((batch_size, 1), start_token_id, device=device)
            max_length = 100  # Max length for inference
        
        # Output tensor to store predictions
        outputs = torch.zeros(batch_size, max_length, self.vocab_size, device=device)
        
        # Initialize first token probability
        outputs[:, 0, decoder_input[0, 0]] = 1.0
        
        # Generate sequence token by token
        for t in range(1, max_length):
            # Embed current input token
            embedded = self.dropout(self.embedding(decoder_input))
            
            # Calculate attention scores using scaled dot-product attention
            # Project decoder hidden state
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
                # temperature = 0.7  # Adjust between 0.5-1.0 for creativity/coherence balance
                # probs = F.softmax(prediction / temperature, dim=1)
                # decoder_input = torch.multinomial(probs, 1)
                # Replace your existing multinomial sampling code with this:
                # generated_tokens_mapped = torch.zeros(outputs[0].size(0), dtype=torch.long, device=device)

                # # Apply nucleus sampling for each position in the sequence
                # for pos in range(outputs[0].size(0)):
                #     logits = outputs[0][pos]  # Get logits for this position
                #     generated_tokens_mapped[pos] = nucleus_sampling(logits, p=0.9)
        return outputs

