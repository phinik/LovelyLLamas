"""
This script analyzes the state_dict of a PyTorch model checkpoint
to determine its architecture and parameter dimensions.
"""

import torch
import os
import sys
from collections import defaultdict

def analyze_state_dict(checkpoint_path):
    """
    Load a PyTorch checkpoint and analyze its state_dict.
    
    Args:
        checkpoint_path (str): Path to the PyTorch checkpoint file
    """
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: File {checkpoint_path} does not exist.")
        return
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check what's in the checkpoint
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        # Check if config exists and print it
        if 'config' in checkpoint:
            print("\nModel configuration:")
            for key, value in checkpoint['config'].items():
                print(f"  - {key}: {value}")
        
        # Extract and analyze the state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            print("No 'model_state_dict' found in checkpoint.")
            return
        
        # Get all parameter shapes
        print("\nState dict parameter shapes:")
        
        # Group parameters by component for better organization
        components = defaultdict(list)
        
        for key, tensor in state_dict.items():
            # Extract component name (part before first dot)
            if '.' in key:
                component = key.split('.')[0]
            else:
                component = 'base'
                
            components[component].append((key, tensor.shape))
        
        # Print organized by component
        for component, params in components.items():
            print(f"\n{component}:")
            for param_name, shape in params:
                print(f"  - {param_name}: {shape}")
                
        # Analyze dimensions across layers
        print("\nKey dimension analysis:")
        
        # Extract hidden dimensions
        hidden_dims = set()
        embedding_dims = set()
        
        for key, tensor in state_dict.items():
            if 'hidden' in key or 'gru' in key:
                for dim in tensor.shape:
                    if dim > 1 and dim != 3 and dim % 64 == 0:
                        hidden_dims.add(dim)
            if 'embedding' in key:
                for dim in tensor.shape:
                    if dim > 1 and dim % 32 == 0 and dim not in hidden_dims:
                        embedding_dims.add(dim)
        
        print(f"  - Likely hidden dimensions: {sorted(hidden_dims)}")
        print(f"  - Likely embedding dimensions: {sorted(embedding_dims)}")
        
        # Check for token_mappings
        if 'token_mappings' in checkpoint:
            token_mappings = checkpoint['token_mappings']
            print(f"\nVocabulary size: {len(token_mappings['used_token_ids'])}")
        
        # Print model summary
        print("\nInferred model architecture:")
        
        # Try to determine model type
        model_type = "unknown"
        if any('attention' in key for key in state_dict.keys()):
            model_type = "AttentionWeatherGRU"
        elif any('encoder_gru' in key for key in state_dict.keys()):
            model_type = "AdvancedWeatherGRU"
        elif any('feature_encoder' in key for key in state_dict.keys()):
            model_type = "BasicWeatherGRU"
            
        hidden_dim = max(hidden_dims) if hidden_dims else "unknown"
        embedding_dim = max(embedding_dims) if embedding_dims else "unknown"
        
        print(f"  - Model type: {model_type}")
        print(f"  - Hidden dimension: {hidden_dim}")
        print(f"  - Embedding dimension: {embedding_dim}")
        
    except Exception as e:
        print(f"Error analyzing checkpoint: {str(e)}")

if __name__ == "__main__":
    # Get checkpoint path from command line or use default
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else 'gru/models/attention_7M.pt'
    analyze_state_dict(checkpoint_path)
    
    # Compare with other model if specified
    if len(sys.argv) > 2:
        print("\n" + "="*80 + "\n")
        analyze_state_dict(sys.argv[2])