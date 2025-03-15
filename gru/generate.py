"""
Example usage of the standardized weather datasets with GRU models for text generation.
This script demonstrates how to generate weather text for specific cities or random samples
using different GRU model types, automatically selecting the appropriate dataset.
"""

import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from weather_gru_models import BasicWeatherGRU, create_model_by_name
from weather_datasets import (
    prepare_weather_dataset,
    create_weather_dataloader,
    generate_for_city,
    list_available_cities,
    get_city_samples
)

# Import the special attention model loader
from attention_model_loader import load_compatible_attention_model

def generate_weather_text(model_type="basic", model_path=None, cities=None, num_samples=3):
    """
    Generate weather text for specific cities or random samples,
    automatically selecting the appropriate dataset for the model type.
    
    Args:
        model_type: Type of model to use ('basic', 'advanced', or 'attention')
        model_path: Path to the pretrained model (None to use default path for model_type)
        cities: List of city names (None for random cities)
        num_samples: Number of samples to generate per city
    """
    # Determine dataset type based on model type
    dataset_type = 'standard' if model_type == 'basic' else 'simple'
    
    # Determine model path if not specified
    if model_path is None:
        model_path = f"models/best_{model_type}_gru_model.pt"
    
    print(f"Using {model_type} model with {dataset_type} dataset")
    print(f"Model path: {model_path}")
    
    # 1. Prepare dataset (only clean dataset needed for generation)
    print(f"Preparing {dataset_type} weather dataset...")
    clean_dataset, _, _, city_mapping = prepare_weather_dataset(
        dataset_type=dataset_type,
        return_city_mapping=True
    )
    
    # 2. Setup tokenizer
    print("Setting up tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    special_tokens = {
        'additional_special_tokens': ['<city>','<temp>','<date>','<velocity>','<percentile>','<rainfall>', '<ne>']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # 3. Load model and token mappings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {model_type} model from {model_path} to {device}...")
    
    try:
        # For attention models, use special compatible loader
        if model_type.lower() == "attention":
            # Get feature dimension from the dataset
            feature_dim = clean_dataset.dataset.feature_dim
            
            # Use special loader for attention models
            model, token_mappings = load_compatible_attention_model(
                checkpoint_path=model_path,
                feature_dim=feature_dim,
                device=device
            )
            print(f"Successfully loaded attention model using compatible loader")
        else:
            # For basic and advanced models, use normal loading approach
            checkpoint = torch.load(model_path, map_location=device)
            token_mappings = checkpoint['token_mappings']
            
            # Get model config - check all possible locations based on model type
            model_config = {}
            
            # Different models store config in different places
            if 'config' in checkpoint:
                # For basic models
                model_config = checkpoint['config']
                print("Found configuration in 'config' key")
            elif 'model_config' in checkpoint:
                # For advanced/attention models
                model_config = checkpoint['model_config']
                print("Found configuration in 'model_config' key")
            else:
                print("No configuration found in checkpoint, will detect from state_dict")
                
            # If no config or incomplete config, detect from state dict
            if not model_config or 'hidden_dim' not in model_config:
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    
                    # Detect key dimensions from state dict
                    detected_config = detect_model_config_from_state_dict(state_dict, model_type)
                    
                    # Update model config with detected values
                    for key, value in detected_config.items():
                        if key not in model_config:
                            model_config[key] = value
                    
                    print(f"Detected configuration from state dict: {detected_config}")
            
            # Get feature dimension from the dataset
            feature_dim = clean_dataset.dataset.feature_dim
            
            # Get configuration parameters from checkpoint or use defaults
            hidden_size = model_config.get('hidden_dim', 512)
            n_layers = model_config.get('n_layers', 2 if model_type != 'basic' else 1)
            embedding_dim = model_config.get('embedding_dim', hidden_size // 2)
            dropout = model_config.get('dropout', 0.1)
            vocab_size = len(token_mappings['used_token_ids'])
            
            print(f"Creating model with feature_dim={feature_dim}, vocab_size={vocab_size}, " 
                f"embedding_dim={embedding_dim}, hidden_dim={hidden_size}, n_layers={n_layers}")
            
            # Create model based on type
            if model_type.lower() == "basic":
                # For BasicWeatherGRU, import and use the class directly
                model = BasicWeatherGRU(
                    feature_dim=feature_dim,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_size,
                    n_layers=n_layers,
                    dropout=dropout
                )
            else:
                # For advanced model, use the factory function
                model = create_model_by_name(
                    model_name=model_type,
                    feature_dim=feature_dim,
                    vocab_size=vocab_size,
                    hidden_size=hidden_size
                )
            
            # Load the model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            print(f"Successfully loaded {model_type} model")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # 4. Determine which cities to generate for
    if cities is None or len(cities) == 0:
        # List cities with at least 3 samples
        available_cities = list_available_cities(clean_dataset, min_samples=3)
        if len(available_cities) == 0:
            print("No cities with enough samples found in the dataset")
            return
            
        # Pick random cities if none specified
        if len(available_cities) > 5:
            selected_cities = random.sample([city for city, _ in available_cities], 5)
        else:
            selected_cities = [city for city, _ in available_cities]
            
        print(f"Randomly selected cities: {', '.join(selected_cities)}")
    else:
        # Verify that specified cities exist in the dataset
        valid_cities = []
        for city in cities:
            if city in city_mapping and len(city_mapping[city]) > 0:
                valid_cities.append(city)
            else:
                print(f"Warning: City '{city}' not found in dataset or has no samples")
        
        if len(valid_cities) == 0:
            print("None of the specified cities were found in the dataset")
            return
            
        selected_cities = valid_cities
        print(f"Generating for specified cities: {', '.join(selected_cities)}")
    
    # 5. Generate text for each selected city
    for city in selected_cities:
        print(f"\n--- Generating weather text for {city} with {model_type} model ---")
        
        # Get samples for this city
        city_samples = get_city_samples(clean_dataset, city, max_samples=num_samples)
        if not city_samples:
            print(f"No samples found for {city}")
            continue
        
        # Create a mini-batch for this city
        features = torch.stack([sample['features'] for sample in city_samples]).to(device)
        original_texts = [sample['text'] for sample in city_samples]
        
        # Process based on model type
        if model_type.lower() == "basic":
            # Direct generation for BasicWeatherGRU
            with torch.no_grad():
                model.eval()
                generated_tokens = model.generate(
                    features,
                    max_length=100,
                    token_mappings=token_mappings
                )
                
                # Process and display results
                for i, (tokens, original) in enumerate(zip(generated_tokens, original_texts)):
                    # Map tokens back to original vocabulary
                    original_token_ids = [token_mappings['reverse_token_id_map'][t.item()] for t in tokens]
                    
                    # Decode to text
                    generated_text = tokenizer.decode(original_token_ids, skip_special_tokens=False)
                    # Clean up text (remove special tokens if needed)
                    generated_text = generated_text.replace("[CLS]", "").replace("[SEP]", "").strip()
                    
                    # Get reference data
                    temp_values = features[i, :, 0].cpu().numpy()  # Temperature (first feature)
                    humidity_values = features[i, :, 1].cpu().numpy()  # Humidity (second feature)
                    cloud_values = features[i, :, 2].cpu().numpy()  # Cloudiness (third feature)
                    
                    print(f"\nSample {i+1}:")
                    print(f"Temperature: {[round(float(t), 1) for t in temp_values[:10]]}...")
                    print(f"Humidity: {[round(float(h), 1) for h in humidity_values[:10]]}...")
                    print(f"Cloudiness: {[round(float(c), 3) for c in cloud_values[:10]]}...")
                    print(f"Original: {original[:500]}..." if len(original) > 500 else f"Original: {original}")
                    print(f"Generated: {generated_text}")
                    print("-" * 80)
        else:
            # For advanced/attention models, use the DataLoader approach
            city_dataloader = create_weather_dataloader(
                city_samples,
                batch_size=len(city_samples),
                tokenizer=tokenizer,
                token_id_map=token_mappings['token_id_map'],
                shuffle=False
            )
            
            # Use generate_samples function for text generation
            print("\nGenerated samples:")
            generate_samples(model, tokenizer, city_dataloader, token_mappings, num_samples=num_samples)


def detect_model_config_from_state_dict(state_dict, model_type):
    """
    Detect model configuration from state dictionary.
    
    Args:
        state_dict: Model state dictionary
        model_type: Type of model ('basic', 'advanced', 'attention')
        
    Returns:
        dict: Detected model configuration
    """
    config = {}
    
    # Detect embedding dimension
    if 'embedding.weight' in state_dict:
        config['embedding_dim'] = state_dict['embedding.weight'].shape[1]
    
    # Detect hidden dimension based on model type
    if model_type == 'basic':
        # For BasicWeatherGRU
        if 'gru.weight_ih_l0' in state_dict:
            config['hidden_dim'] = state_dict['gru.weight_ih_l0'].shape[0] // 3
        elif 'output_layer.1.weight' in state_dict:
            config['hidden_dim'] = state_dict['output_layer.1.weight'].shape[1]
    elif model_type == 'attention':
        # For AttentionWeatherGRU
        if 'encoder_gru.weight_ih_l0' in state_dict:
            config['hidden_dim'] = state_dict['encoder_gru.weight_ih_l0'].shape[0] // 3
        elif 'decoder_gru.weight_ih_l0' in state_dict:
            config['hidden_dim'] = state_dict['decoder_gru.weight_ih_l0'].shape[0] // 3
    else:
        # For AdvancedWeatherGRU
        if 'encoder_gru.gru_layers.0.weight_ih_l0' in state_dict:
            config['hidden_dim'] = state_dict['encoder_gru.gru_layers.0.weight_ih_l0'].shape[1]
        elif 'decoder_gru.gru_layers.0.weight_ih_l0' in state_dict:
            config['hidden_dim'] = state_dict['decoder_gru.gru_layers.0.weight_ih_l0'].shape[1]
        
    # Detect number of layers
    n_layers = 1
    for key in state_dict.keys():
        if 'weight_ih_l' in key:
            layer_num = int(key.split('weight_ih_l')[1].split('_')[0]) + 1
            n_layers = max(n_layers, layer_num)
    
    config['n_layers'] = n_layers
    
    return config


def generate_samples(model, tokenizer, data_loader, token_mappings, num_samples=3):
    """Generate text samples from the model"""
    # Get device from model
    device = next(model.parameters()).device
    
    model.eval()
    
    # Get mapping for converting back to original token IDs
    reverse_map = token_mappings['reverse_token_id_map']
    
    # IDs of tokens to exclude from output
    tokens_to_exclude = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id, 
        tokenizer.sep_token_id
    }

    # Get samples from the dataloader
    samples = []
    data_iter = iter(data_loader)
    try:
        batch = next(data_iter)
        samples.append(batch)
    except StopIteration:
        print("No samples available in dataloader")
        return
    
    # Generate text for the batch
    batch = samples[0]  # Just use the first (and only) batch
    
    # Process each item in the batch
    for sample_idx in range(min(num_samples, len(batch['features']))):
        # Get features for this sample
        sample_features = batch['features'][sample_idx].unsqueeze(0).to(device)
        sample_tokens_mapped = batch['text'][sample_idx]
        
        # Get temperature, humidity and cloudiness data for reference
        temp_values = sample_features[0, :, 0].cpu().numpy()  # Temperature (first feature)
        humidity_values = sample_features[0, :, 1].cpu().numpy()  # Humidity (second feature) 
        cloud_values = sample_features[0, :, 2].cpu().numpy()  # Cloudiness (third feature)
        
        # Generate text
        with torch.no_grad():
            # For models with or without generate method
            if hasattr(model, 'generate') and callable(getattr(model, 'generate')):
                # Use model's built-in generate method if available (BasicWeatherGRU)
                generated_tokens = model.generate(
                    sample_features, 
                    max_length=100,
                    token_mappings=token_mappings
                )
                # Convert to the right format for processing below
                generated_tokens_mapped = generated_tokens[0]
            else:
                # For advanced/attention models that don't have generate method
                generated_output = model(sample_features)
                # Use temperature sampling for more natural text
                temperature = 0.1  # Use 0.1 for consistency with BasicWeatherGRU
                logits = generated_output[0] / temperature
                probs = F.softmax(logits, dim=1)
                generated_tokens_mapped = torch.multinomial(probs, num_samples=1).squeeze(-1)
                # For pure argmax approach uncomment this instead:
                # generated_tokens_mapped = generated_output[0].argmax(dim=1)
        
        # Map tokens back to original vocabulary
        original_tokens = torch.tensor([
            reverse_map[token.item()] for token in sample_tokens_mapped
        ])
        
        generated_tokens = torch.tensor([
            reverse_map[token.item()] for token in generated_tokens_mapped
        ])
        
        # Filter out unwanted special tokens from generation
        filtered_generated = [token.item() for token in generated_tokens 
                           if token.item() not in tokens_to_exclude]
        
        # Decode to text
        original_text = tokenizer.decode(original_tokens, skip_special_tokens=False)
        generated_text = tokenizer.decode(filtered_generated, skip_special_tokens=False)
        
        print(f"\nSample {sample_idx+1}:")
        print(f"Temperature: {[round(float(t), 1) for t in temp_values[:10]]}...")  # Show first 10 values
        print(f"Humidity: {[round(float(h), 1) for h in humidity_values[:10]]}...")
        print(f"Cloudiness: {[round(float(c), 2) for c in cloud_values[:10]]}...")
        print(f"Original: {original_text[:500]}..." if len(original_text) > 500 else f"Original: {original_text}")
        print(f"Generated: {generated_text}")
        print("-" * 80)

if __name__ == "__main__":
    # Specify which GRU model type to use
    # Options: "basic", "advanced", or "attention"
    model_type = "advanced"  # Try with attention model
    
    # Specific cities to generate for (use None or [] for random cities)
    specific_cities = ["Paraćin", "Bull Bay", "Makov", "Carroll", "Ormiston"]
    # specific_cities = None  # Uncomment to use random cities
    
    # Choose which model file to use
    model_path = 'gru/models/advanced_8M.pt'  # Try with attention model
    
    # Generate weather text
    generate_weather_text(
        model_path=model_path,
        model_type=model_type,
        cities=specific_cities,
        num_samples=1
    )