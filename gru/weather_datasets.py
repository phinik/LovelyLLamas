"""
Weather Datasets Module

This module provides standardized implementations of weather datasets with different feature sets:
- StandardWeatherDataset: Comprehensive dataset with NER and multithreading support
- SimpleWeatherDataset: Simplified dataset with reduced feature set

Both datasets can be used interchangeably with the different GRU models.
"""

import torch
import re
import math
import numpy as np
import os
import json
import time
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm
from collections import Counter, defaultdict
import concurrent.futures
from functools import partial
import multiprocessing
from sklearn.model_selection import train_test_split
import random


class BaseWeatherDataset(Dataset):
    """
    Base class for weather datasets with common functionality.
    
    This base class implements shared methods for text processing and feature extraction
    that are used by both dataset variants.
    """
    
    def __init__(self, weather_data, max_length=100):
        """
        Initialize the base weather dataset.
        
        Args:
            weather_data (list or dict): List of weather data samples or a single sample dict
            max_length (int): Maximum sequence length
        """
        self.data = [weather_data] if isinstance(weather_data, dict) else weather_data
        self.max_length = max_length
        
        # Initialize wind directions from all data points
        self.wind_directions = sorted(list(set([d for data in self.data for d in data['windrichtung']])))
        self.wind_dir_to_idx = {d: i for i, d in enumerate(self.wind_directions)}
        
        # City tracking - will be populated after dataset creation
        self.city_to_indices = defaultdict(list)
        self._build_city_index()
    
    def _build_city_index(self):
        """Build an index mapping city names to their respective dataset indices."""
        for idx, item in enumerate(self.data):
            city = item.get('city', '').strip()
            if city:
                self.city_to_indices[city].append(idx)
    
    def replace_dates(self, text: str) -> str:
        """Replace date patterns with <date> tag."""
        text = re.sub(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b", "<date>", text)
        return text
    
    def replace_city_and_units(self, text: str, city: str) -> str:
        """
        Replace city names and standardize units in text.
        
        Args:
            text (str): Input text
            city (str): City name to replace
            
        Returns:
            str: Processed text with standardized units and city replaced
        """
        # Replace the city name with <city> tag
        # Make sure city isn't empty before replacing
        if city and len(city) > 0:
            text = re.sub(r'\b' + re.escape(city) + r'\b', '<city>', text)

        unit_patterns = [
            # TEMPERATURE
            (r'(°[ ]*C|Grad)', ' <temp>'),
            # VELOCITY
            (r'[ ]*km/h', ' <velocity>'),
            # PERCENTILE
            (r'[ ]*%', ' <percentile>'),
            # RAINFALL DELTA
            (r'[ ]*l\/m²', ' <rainfall>')
        ]
        
        for pattern, replacement in unit_patterns:
            try:
                text = re.sub(pattern, replacement, text)
            except Exception:
                continue

        # REMOVE MARKUP
        text = re.sub(r'\**', '', text)

        # REMOVE WEIRD PUNCUATION
        text = re.sub(r' \.', '.', text)

        # REMOVE UNNECESSARY NEWLINES
        text = re.sub(r'\n\n', '\n', text)

        # REMOVE SPACE AFTER NEWLINE
        text = re.sub(r'\n ', '\n', text)

        # REPLACE MULTIPLE WHITESPACES WITH ONE
        text = re.sub(r' +', ' ', text)
        
        return text

    def _parse_time(self, time_str):
        """
        Parse time string and handle missing data.
        
        Args:
            time_str (str): Time string in format "HH:MM Uhr" or similar
            
        Returns:
            float or None: Hour as float (e.g., 14.5 for 14:30) or None if invalid
        """
        if time_str == '-' or not time_str:
            return None
        try:
            # Handle "HH:MM Uhr" format
            if ':' in time_str:
                hour, minute = map(int, time_str.split(' ')[0].split(':'))
                return hour + minute/60
            return None
        except (ValueError, IndexError):
            return None

    def _encode_time(self, time_str):
        """
        Convert time string to cyclic features (sin/cos encoding).
        
        Args:
            time_str (str): Time string to encode
            
        Returns:
            torch.Tensor: 2D tensor with sin/cos encoding of time
        """
        try:
            start_hour = int(time_str.split(' - ')[0])
            hour_sin = torch.sin(torch.tensor(2 * math.pi * start_hour / 24))
            hour_cos = torch.cos(torch.tensor(2 * math.pi * start_hour / 24))
            return torch.tensor([hour_sin, hour_cos])
        except (ValueError, IndexError):
            # Return neutral values for invalid time
            return torch.tensor([0.0, 1.0])
    
    def _encode_sun_info(self, sunrise, sunset, current_time):
        """
        Encode sun-related information (daylight, time since sunrise, time until sunset).
        
        Args:
            sunrise (str): Sunrise time string
            sunset (str): Sunset time string
            current_time (str): Current time string
            
        Returns:
            torch.Tensor: 3D tensor with encoded sun information
        """
        # Parse times, handling missing data
        sunrise_hour = self._parse_time(sunrise)
        sunset_hour = self._parse_time(sunset)
        
        try:
            current_hour = float(current_time.split(' - ')[0])
        except (ValueError, IndexError):
            # Return default values if current time is invalid
            return torch.tensor([0.0, 0.0, 0.0])
        
        # If sunrise or sunset is missing, use approximate values based on season
        if sunrise_hour is None or sunset_hour is None:
            # Return default encoding indicating uncertainty
            return torch.tensor([
                0.5,  # Unknown daylight status
                0.0,  # Neutral time since sunrise
                0.0   # Neutral time until sunset
            ])
        
        # Calculate daylight features
        is_daylight = (current_hour >= sunrise_hour) and (current_hour <= sunset_hour)
        
        if is_daylight:
            time_since_sunrise = (current_hour - sunrise_hour) / (sunset_hour - sunrise_hour)
            time_until_sunset = (sunset_hour - current_hour) / (sunset_hour - sunrise_hour)
        else:
            if current_hour < sunrise_hour:
                time_since_sunrise = -1 * (sunrise_hour - current_hour) / (24 - sunset_hour + sunrise_hour)
                time_until_sunset = -1
            else:
                time_since_sunrise = -1
                time_until_sunset = -1 * (current_hour - sunset_hour) / (24 - sunset_hour + sunrise_hour)
        
        return torch.tensor([float(is_daylight), time_since_sunrise, time_until_sunset])

    def one_hot_wind(self, wind_dir):
        """
        One-hot encode wind direction.
        
        Args:
            wind_dir (str): Wind direction string
            
        Returns:
            torch.Tensor: One-hot encoded wind direction
        """
        encoding = torch.zeros(len(self.wind_directions))
        encoding[self.wind_dir_to_idx[wind_dir]] = 1
        return encoding
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def get_city_samples(self, city_name):
        """
        Get all samples for a specific city.
        
        Args:
            city_name (str): Name of the city
            
        Returns:
            list: List of (index, sample) tuples for the city
        """
        indices = self.city_to_indices.get(city_name, [])
        return [(idx, self.data[idx]) for idx in indices]
    
    def get_all_cities(self):
        """
        Get dictionary of all cities and their sample indices.
        
        Returns:
            dict: Dictionary mapping city names to lists of sample indices
        """
        return dict(self.city_to_indices)


class StandardWeatherDataset(BaseWeatherDataset):
    """
    Comprehensive weather dataset with NER support and full feature set.
    
    This dataset includes:
    - Temperature, rain risk, rain amount, wind speed, pressure, humidity, cloudiness
    - Wind direction (one-hot encoded)
    - Time features (cyclic encoding)
    - Sun-related features
    - Named Entity Recognition for geographic entities
    """
    
    def __init__(self, weather_data, max_length=100, ner_workers=None):
        """
        Initialize the comprehensive weather dataset.
        
        Args:
            weather_data (list or dict): Weather data to process
            max_length (int): Maximum sequence length
            ner_workers (int, optional): Number of worker threads for NER processing
        """
        super().__init__(weather_data, max_length)
        
        # Set number of workers for NER processing
        if ner_workers is None:
            self.ner_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
        else:
            self.ner_workers = ner_workers
        
        # Calculate feature dimension
        self.feature_dim = (
            1 +  # temperature
            1 +  # rain risk
            1 +  # rain amount
            1 +  # wind speed
            1 +  # pressure
            1 +  # humidity
            1 +  # cloudiness
            len(self.wind_directions) +  # one-hot wind directions
            2 +  # time encoding (sin, cos)
            3 +  # sun features
            1    # sun hours
        )
        
        # Initialize spaCy NER model for German text
        self.has_ner = False
        self.nlp = None
        try:
            import spacy
            print("Loading German NER model...")
            # Disable components we don't need to speed up processing
            self.nlp = spacy.load("de_core_news_lg", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
            self.has_ner = True
            print("NER model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load spaCy model: {e}")
            print("Make sure to install it with:")
            print("pip install spacy")
            print("python -m spacy download de_core_news_lg")
    
    def apply_ner_replacement(self, text):
        """
        Apply Named Entity Recognition to replace geographic entities with <ne> tag.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with geographic entities replaced by <ne> tag
        """
        if not self.has_ner or not self.nlp:
            return text
            
        doc = self.nlp(text)
        
        # Collect entities and their positions
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["LOC", "GPE"]:  # Locations and Geopolitical entities
                entities.append((ent.start_char, ent.end_char, ent.text))
        
        # Sort entities by start position in reverse order to avoid offset issues
        entities.sort(reverse=True, key=lambda x: x[0])
        
        # Replace each entity with <ne> tag
        for start, end, entity in entities:
            text = text[:start] + "<ne>" + text[end:]
        
        return text
    
    def batch_process_ner(self, region_texts, batch_size=8):
        """
        Process a batch of texts with NER using threading instead of multiprocessing.
        
        Args:
            region_texts (list): List of text regions to process
            batch_size (int): Number of texts to process in each batch
            
        Returns:
            list: Processed text regions with entities replaced by <ne> tag
        """
        if not self.has_ner or not self.nlp:
            return region_texts
            
        # Function to process a single batch
        def process_batch(batch):
            return [self.apply_ner_replacement(text) for text in batch]
        
        # Split into batches
        batches = [region_texts[i:i+batch_size] for i in range(0, len(region_texts), batch_size)]
        
        # Process batches using threading (avoid pickling issues)
        processed_regions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.ner_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures):
                processed_regions.extend(future.result())
                
        return processed_regions
    
    def process_text(self, text, city):
        """
        Process text with all replacements in the correct order.
        
        Args:
            text (str): Original text
            city (str): City name to replace
            
        Returns:
            str: Processed text with all replacements applied
        """
        # First replace city and units 
        processed_text = self.replace_city_and_units(text, city)
        
        # Then replace dates
        processed_text = self.replace_dates(processed_text)
        
        # Apply NER replacement if available
        if self.has_ner:
            # Get all current special token positions
            special_tokens = ['<city>', '<temp>', '<date>', '<velocity>', '<percentile>', '<rainfall>']
            protected_regions = []
            
            for token in special_tokens:
                start_idx = 0
                while start_idx < len(processed_text):
                    pos = processed_text.find(token, start_idx)
                    if pos == -1:
                        break
                    protected_regions.append((pos, pos + len(token)))
                    start_idx = pos + len(token)
            
            # Sort protected regions
            protected_regions.sort()
            
            # Create a list of unprotected text regions that can be processed with NER
            unprotected_regions = []
            last_end = 0
            
            for start, end in protected_regions:
                if start > last_end:
                    unprotected_regions.append((last_end, start))
                last_end = end
            
            # Add the final region
            if last_end < len(processed_text):
                unprotected_regions.append((last_end, len(processed_text)))
            
            # Extract text regions to process
            region_texts = [processed_text[start:end] for start, end in unprotected_regions]
            
            # Skip NER if all regions are very short (not worth the overhead)
            if sum(len(text) for text in region_texts) > 20:
                # Process all regions with threading to avoid pickling issues
                processed_regions = self.batch_process_ner(region_texts)
                
                # Reconstruct the text with protected and processed regions
                result_text = ""
                last_pos = 0
                
                for i, (start, end) in enumerate(unprotected_regions):
                    # Add special tokens before this region
                    result_text += processed_text[last_pos:start]
                    # Add NER-processed region
                    result_text += processed_regions[i] if i < len(processed_regions) else processed_text[start:end]
                    last_pos = end
                
                # Add any remaining text
                result_text += processed_text[last_pos:]
                processed_text = result_text
        
        return processed_text
    
    def scan_dataset_for_named_entities(self, sample_size=None):
        """
        Scan the dataset to identify common named entities using threading
        to avoid multiprocessing pickling issues.
        
        Args:
            sample_size: Number of samples to analyze, None for all
            
        Returns:
            Counter of named entities and their frequencies
        """
        if not self.has_ner:
            print("NER model not available. Cannot scan for named entities.")
            return None
            
        # Get samples to analyze
        if sample_size is None:
            samples = self.data
        else:
            import random
            samples = random.sample(self.data, min(sample_size, len(self.data)))
        
        # Extract text from samples
        texts = []
        for item in samples:
            # Get the appropriate text field
            text_field = 'report_long'
            if text_field not in item:
                if 'gpt_rewritten_v2' in item:
                    text_field = 'gpt_rewritten_v2'
                elif 'gpt_rewritten_apokalyptisch_v2' in item:
                    text_field = 'gpt_rewritten_apokalyptisch_v2'
            
            # Skip if no valid text field
            if text_field in item:
                texts.append(item[text_field])
        
        print(f"Scanning {len(texts)} texts for named entities...")
        
        # Define a function to process a batch of texts
        def process_text_batch(text_batch):
            entities = []
            for text in text_batch:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ["LOC", "GPE"]:
                        entities.append(ent.text)
            return entities
        
        all_entities = []
        batch_size = 10  # Smaller batch size for better progress updates
        
        # Split texts into batches
        text_batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process using ThreadPoolExecutor to avoid pickling issues
        with tqdm(total=len(texts), desc="Processing samples") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.ner_workers) as executor:
                futures = [executor.submit(process_text_batch, batch) for batch in text_batches]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_entities = future.result()
                        all_entities.extend(batch_entities)
                        pbar.update(batch_size)
                    except Exception as e:
                        print(f"Error processing batch: {e}")
        
        # Return counter of entities
        return Counter(all_entities)
        
    def __getitem__(self, idx):
        """
        Get a dataset item by index.
        
        Args:
            idx (int): Item index
            
        Returns:
            dict: Dictionary with features and text
        """
        item = self.data[idx]
        
        # Get sequence length from the data
        seq_len = len(item['temperatur_in_deg_C'])
        
        # Initialize features tensor with correct shape
        features = torch.zeros((seq_len, self.feature_dim))
        
        # Fill features one by one, maintaining consistent shapes
        current_idx = 0
        
        # Temperature feature
        features[:, current_idx] = torch.tensor([float(t) for t in item['temperatur_in_deg_C']])
        current_idx += 1
        
        # Rain risk feature
        features[:, current_idx] = torch.tensor([float(r) for r in item['niederschlagsrisiko_in_perc']])
        current_idx += 1
        
        # Rain amount feature with forward filling for NaN values
        rain_values = []
        last_valid = 0.0
        for r in item['niederschlagsmenge_in_l_per_sqm']:
            try:
                if isinstance(r, str):
                    val = float(r)
                elif isinstance(r, float) and not math.isnan(r):
                    val = r
                else:
                    val = last_valid
                rain_values.append(val)
                last_valid = val
            except ValueError:
                rain_values.append(last_valid)
        features[:, current_idx] = torch.tensor(rain_values)
        current_idx += 1
        
        # Wind speed feature
        features[:, current_idx] = torch.tensor([float(w) for w in item['windgeschwindigkeit_in_km_per_s']])
        current_idx += 1
        
        # Pressure feature
        features[:, current_idx] = torch.tensor([float(p) for p in item['luftdruck_in_hpa']])
        current_idx += 1
        
        # Humidity feature
        features[:, current_idx] = torch.tensor([float(h) for h in item['relative_feuchte_in_perc']])
        current_idx += 1
        
        # Cloudiness feature
        features[:, current_idx] = torch.tensor([float(c.split('/')[0]) / 8 for c in item['bewölkungsgrad']])
        current_idx += 1
        
        # Wind directions (one-hot encoded)
        wind_features = torch.stack([self.one_hot_wind(w) for w in item['windrichtung']])
        features[:, current_idx:current_idx + len(self.wind_directions)] = wind_features
        current_idx += len(self.wind_directions)
        
        # Time features
        time_features = torch.stack([self._encode_time(t) for t in item['times']])
        features[:, current_idx:current_idx + 2] = time_features
        current_idx += 2
        
        # Sun features
        sun_features = torch.stack([
            self._encode_sun_info(
                item.get('sunrise', '-'), 
                item.get('sundown', '-'), 
                t
            ) for t in item['times']
        ])
        features[:, current_idx:current_idx + 3] = sun_features
        current_idx += 3
        
        # Sun hours feature
        sun_hours = torch.tensor([1.0 if "fast nicht zu sehen" in item.get('sunhours', '') else 0.0])
        features[:, current_idx] = sun_hours.expand(seq_len)
        
        # Get the appropriate text field - prefer "report_long" but use alternatives if needed
        text_field = 'report_long'
        if text_field not in item:
            if 'gpt_rewritten_v2' in item:
                text_field = 'gpt_rewritten_v2'
            elif 'gpt_rewritten_apokalyptisch_v2' in item:
                text_field = 'gpt_rewritten_apokalyptisch_v2'
        
        # Process the text 
        processed_text = self.process_text(item[text_field], item['city'])
        
        return {
            'features': features,
            'text': processed_text
        }

class SimpleWeatherDataset(BaseWeatherDataset):
    """
    Simplified weather dataset with reduced feature set.
    
    This dataset includes:
    - Temperature, humidity, cloudiness
    - Wind direction (one-hot encoded)
    - Time features (cyclic encoding)
    - Sun-related features
    - Sun hours
    """
    
    def __init__(self, weather_data, max_length=100):
        """
        Initialize the simplified weather dataset.
        
        Args:
            weather_data (list or dict): Weather data to process
            max_length (int): Maximum sequence length
        """
        super().__init__(weather_data, max_length)
        
        # Calculate feature dimension (reduced compared to StandardWeatherDataset)
        self.feature_dim = (
            1 +  # temperature
            1 +  # humidity
            1 +  # cloudiness
            len(self.wind_directions) +  # one-hot wind directions
            2 +  # time encoding (sin, cos)
            3 +  # sun features
            1    # sun hours
        )
    
    def __getitem__(self, idx):
        """
        Get a dataset item by index.
        
        Args:
            idx (int): Item index
            
        Returns:
            dict: Dictionary with features and text
        """
        item = self.data[idx]
        
        # Get sequence length from the data
        seq_len = len(item['temperatur_in_deg_C'])
        
        # Initialize features tensor with correct shape
        features = torch.zeros((seq_len, self.feature_dim))
        
        # Fill features one by one, maintaining consistent shapes
        current_idx = 0
        
        # Temperature feature
        features[:, current_idx] = torch.tensor([float(t) for t in item['temperatur_in_deg_C']])
        current_idx += 1
        
        # Humidity feature
        features[:, current_idx] = torch.tensor([float(h) for h in item['relative_feuchte_in_perc']])
        current_idx += 1
        
        # Cloudiness feature
        features[:, current_idx] = torch.tensor([float(c.split('/')[0]) / 8 for c in item['bewölkungsgrad']])
        current_idx += 1
        
        # Wind directions (one-hot encoded)
        wind_features = torch.stack([self.one_hot_wind(w) for w in item['windrichtung']])
        features[:, current_idx:current_idx + len(self.wind_directions)] = wind_features
        current_idx += len(self.wind_directions)
        
        # Time features
        time_features = torch.stack([self._encode_time(t) for t in item['times']])
        features[:, current_idx:current_idx + 2] = time_features
        current_idx += 2
        
        # Sun features
        sun_features = torch.stack([
            self._encode_sun_info(
                item.get('sunrise', '-'), 
                item.get('sundown', '-'), 
                t
            ) for t in item['times']
        ])
        features[:, current_idx:current_idx + 3] = sun_features
        current_idx += 3
        
        # Sun hours feature
        sun_hours = torch.tensor([1.0 if "fast nicht zu sehen" in item.get('sunhours', '') else 0.0])
        features[:, current_idx] = sun_hours.expand(seq_len)
        
        # Get the appropriate text field - prefer "gpt_rewritten_apokalyptisch_v2" for SimpleWeatherDataset
        text_field = 'gpt_rewritten_apokalyptisch_v2'
        if text_field not in item:
            if 'gpt_rewritten_v2' in item:
                text_field = 'gpt_rewritten_v2'
            elif 'report_long' in item:
                text_field = 'report_long'
        
        # Process text with simpler processing (no NER)
        processed_text = self.replace_dates(self.replace_city_and_units(item[text_field], item['city']))
        
        return {
            'features': features,
            'text': processed_text
        }


# Utility functions for data loading and processing

def check_file(path, dataset_type='standard'):
    """
    Check if a file contains valid keys and values for the given dataset type.
    
    Args:
        path (str): Path to the file to check
        dataset_type (str): 'standard' or 'simple', determines required fields
        
    Returns:
        bool: True if the file should be loaded, False otherwise
    """
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return False

        # Common required keys
        common_required = {'city', 'times', 'temperatur_in_deg_C', 'relative_feuchte_in_perc', 
                           'bewölkungsgrad', 'windrichtung'}
        
        # Dataset-specific required keys
        if dataset_type == 'standard':
            required_keys = common_required.union({
                'niederschlagsrisiko_in_perc', 'niederschlagsmenge_in_l_per_sqm',
                'windgeschwindigkeit_in_km_per_s', 'luftdruck_in_hpa',
                'report_long'
            })
            text_fields = ['report_long', 'gpt_rewritten_v2', 'gpt_rewritten_apokalyptisch_v2']
        else:  # simple
            required_keys = common_required
            text_fields = ['gpt_rewritten_apokalyptisch_v2', 'gpt_rewritten_v2', 'report_long']
        
        # Check if any of the required keys are missing
        if not required_keys.issubset(data.keys()):
            return False
            
        # Check if any of the text fields are present
        has_text_field = any(field in data for field in text_fields)
        if not has_text_field:
            return False
            
        # Check if city is valid
        if not isinstance(data['city'], str) or not data['city'].strip():
            return False
            
        return True

def load_data(path):
    """
    Load data from a JSON file.
    
    Args:
        path (str): Path to the JSON file
        
    Returns:
        dict: Loaded JSON data
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def process_item(idx, dataset):
    """
    Process and validate a single dataset item.
    
    Args:
        idx (int): Index of the item in the dataset
        dataset (Dataset): The dataset to process
        
    Returns:
        int or None: The index if valid, None if invalid
    """
    try:
        sample = dataset[idx]
        features = sample['features']
        
        # Check for NaN or Inf values
        if torch.isnan(features).any() or torch.isinf(features).any():
            return None
        
        return idx
    except Exception as e:
        return None

def validate_and_clean_data_multithreaded(dataset, num_workers=None):
    """
    Validate dataset and remove samples with NaN or Inf values using multiple threads.
    
    Args:
        dataset (Dataset): Dataset to validate
        num_workers (int, optional): Number of worker threads to use
        
    Returns:
        Subset: Dataset subset with only valid samples
    """
    # If num_workers is not specified, use CPU count
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    print(f"Validating data using {num_workers} workers...")
    
    # Use partial to create a function with the dataset already bound
    process_fn = partial(process_item, dataset=dataset)
    
    valid_indices = []
    total_items = len(dataset)
    
    # Use ThreadPoolExecutor for I/O bound operations
    with tqdm(total=total_items, desc="Validating items") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_fn, idx): idx for idx in range(total_items)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result is not None:
                        valid_indices.append(result)
                except Exception as e:
                    print(f"Error processing item {idx}: {e}")
                
                pbar.update(1)
    
    invalid_count = total_items - len(valid_indices)
    print(f"Found {invalid_count} invalid samples out of {total_items}")
    print(f"Keeping {len(valid_indices)} valid samples")
    
    # Create a new subset with valid samples
    subset = Subset(dataset, valid_indices)
    
    # Update city_to_indices for the valid subset
    for city, indices in dataset.city_to_indices.items():
        valid_city_indices = [i for i in indices if i in valid_indices]
        if valid_city_indices:
            subset.dataset.city_to_indices[city] = valid_city_indices
    
    return subset, valid_indices

def load_data_multithreaded(data_path, dataset_type='standard', num_workers=None):
    """
    Load weather data from JSON files using multiple threads.
    
    Args:
        data_path (str): Path to directory containing JSON files
        dataset_type (str): 'standard' or 'simple', determines required fields
        num_workers (int, optional): Number of worker threads to use
        
    Returns:
        list: List of loaded data samples
    """
    # If num_workers is not specified, use CPU count
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    # Check if we're in the right directory, navigate if needed
    if not os.path.exists(data_path):
        base_paths = ['.', '..', '../..']
        for base in base_paths:
            test_path = os.path.join(base, data_path)
            if os.path.exists(test_path):
                data_path = test_path
                break
    
    # List JSON files
    files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    print(f"Found {len(files)} JSON files")
    
    # Define function to process a single file
    def process_file(file):
        file_path = os.path.join(data_path, file)
        try:
            # Check if file is valid for the selected dataset type
            if not check_file(file_path, dataset_type):
                return None
                
            # Load data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # File is valid, return with a key
            key = (file.split('-')[-1]).split('_')[0]
            return (key, data)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None
    
    data_dict = {}
    
    # Process files in parallel
    print(f"Loading files using {num_workers} workers...")
    with tqdm(total=len(files), desc="Loading files") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_file, file): file for file in files}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    key, data = result
                    data_dict[key] = data
                pbar.update(1)
    
    print(f"Loaded {len(data_dict)} valid weather data points")
    
    # Convert to list
    return list(data_dict.values())


def prepare_weather_dataset(dataset_type='standard', data_path='data/files_for_chatGPT/2024-12-12/', 
                   num_workers=None, return_city_mapping=False, ner_workers=None):
    """
    Complete pipeline to prepare a weather dataset.
    
    Args:
        dataset_type (str): 'standard' or 'simple'
        data_path (str): Path to directory containing JSON files
        num_workers (int, optional): Number of worker threads for data loading and validation
        return_city_mapping (bool): Whether to return city to index mapping
        ner_workers (int, optional): Number of worker threads for NER processing
        
    Returns:
        tuple: (clean_dataset, train_dataset, val_dataset, [city_to_indices])
    """
    print(f"Preparing {dataset_type} weather dataset...")
    start_time = time.time()

    # If workers not specified, use CPU count
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    if ner_workers is None:
        ner_workers = num_workers  # Use same number of workers for NER by default
    
    # Cross-check number of workers (if given) with CPU count
    max_safe_workers = max(1, multiprocessing.cpu_count() - 1)
    if ner_workers > max_safe_workers:
        print(f"Warning: Requested {ner_workers} NER workers exceeds recommended maximum. "
            f"Limiting to {max_safe_workers} workers.")
        ner_workers = max_safe_workers
    
    # Load data
    weather_data = load_data_multithreaded(
        data_path=data_path, 
        dataset_type=dataset_type,
        num_workers=num_workers
    )
    
    # Create appropriate dataset based on type
    if dataset_type == 'standard':
        dataset = StandardWeatherDataset(weather_data, ner_workers=ner_workers)
    else:
        dataset = SimpleWeatherDataset(weather_data)
    
    # Clean dataset
    clean_dataset, valid_indices = validate_and_clean_data_multithreaded(
        dataset, 
        num_workers=num_workers
    )
    
    # Split into train/validation
    train_indices, val_indices = train_test_split(
        valid_indices, test_size=0.1, random_state=42
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    end_time = time.time()
    print(f"Dataset preparation completed in {end_time - start_time:.2f} seconds")
    
    if return_city_mapping:
        # The city mapping is already updated during cleaning
        return clean_dataset, train_dataset, val_dataset, clean_dataset.dataset.city_to_indices
    else:
        return clean_dataset, train_dataset, val_dataset

def create_weather_dataloader(dataset, batch_size, tokenizer, token_id_map=None, shuffle=True):
    """
    Create a DataLoader with proper collate function for weather datasets.
    
    Args:
        dataset: Weather dataset
        batch_size: Batch size
        tokenizer: Tokenizer for text encoding
        token_id_map: Optional mapping for token IDs in reduced vocabulary
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader: Configured data loader
    """
    def collate_fn(batch_list):
        # Extract features and texts
        features = torch.stack([item['features'] for item in batch_list])
        texts = [item['text'] for item in batch_list]
        
        # Normalize features within batch for better training stability
        features = (features - features.mean(dim=(0, 1), keepdim=True)) / (
            features.std(dim=(0, 1), keepdim=True) + 1e-8)
        
        # Get token IDs with padding
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        batch = {
            'features': features,
            'text': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
        # Apply token mapping if provided
        if token_id_map is not None:
            # Map the token IDs to new IDs
            old_tokens = batch['text']
            new_tokens = torch.zeros_like(old_tokens)
            
            for i in range(old_tokens.size(0)):
                for j in range(old_tokens.size(1)):
                    old_id = old_tokens[i, j].item()
                    new_tokens[i, j] = token_id_map.get(old_id, 0)  # Default to 0 if token not found
            
            batch['text'] = new_tokens
        
        return batch
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def reduce_vocabulary(tokenizer, dataset, batch_size=64):
    """
    Identify used tokens and create a reduced vocabulary mapping.
    
    Args:
        tokenizer: Tokenizer with full vocabulary
        dataset: Dataset containing texts to analyze
        batch_size: Batch size for processing
        
    Returns:
        tuple: (token_mappings, reduced_vocab_size)
    """
    print("Analyzing vocabulary usage to reduce model size...")
    
    # Count tokens
    token_counter = Counter()
    
    # Process the entire dataset directly without DataLoader
    for idx in tqdm(range(len(dataset)), desc="Scanning token usage"):
        # Get the raw text directly (handles Subset objects correctly)
        if isinstance(dataset, Subset):
            sample = dataset.dataset[dataset.indices[idx]]
        else:
            sample = dataset[idx]
        
        text = sample['text']
        
        # Tokenize directly
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_counter.update(tokens)
    
    # Always keep special tokens
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, str):
            token_id = tokenizer.convert_tokens_to_ids(special_token)
            if token_id not in token_counter:
                token_counter[token_id] = 1
        elif isinstance(special_token, list):
            for token in special_token:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id not in token_counter:
                    token_counter[token_id] = 1
    
    # Sort by frequency for efficient token ID assignment
    used_token_ids = sorted(token_counter.keys())
    
    # Create token ID mapping (old ID -> new ID)
    token_id_map = {old_id: new_id for new_id, old_id in enumerate(used_token_ids)}
    
    # Create reverse mapping for inference
    reverse_token_id_map = {new_id: old_id for old_id, new_id in token_id_map.items()}
    
    # Store the mappings for later use
    token_mappings = {
        'token_id_map': token_id_map,
        'reverse_token_id_map': reverse_token_id_map,
        'used_token_ids': used_token_ids
    }
    
    # Update vocabulary size to reduced size
    reduced_vocab_size = len(used_token_ids)
    original_vocab_size = len(tokenizer.vocab)
    print(f"Reduced vocabulary from {original_vocab_size:,} to {reduced_vocab_size:,} tokens " 
          f"({reduced_vocab_size/original_vocab_size*100:.1f}%)")
    
    return token_mappings, reduced_vocab_size


def generate_for_city(model, city_name, dataset, tokenizer, token_mappings, device='cuda', 
                     max_length=100, num_samples=3):
    """
    Generate weather text for a specific city.
    
    Args:
        model: Trained weather GRU model
        city_name: Name of the city to generate for
        dataset: Dataset containing city data
        tokenizer: Tokenizer for encoding/decoding
        token_mappings: Token ID mappings for vocabulary reduction
        device: Device to run generation on
        max_length: Maximum generation length
        num_samples: Number of samples to generate
        
    Returns:
        list: Generated texts for the city
    """
    # Get indices for the city
    city_indices = dataset.dataset.city_to_indices.get(city_name, [])
    
    if not city_indices:
        print(f"No data found for city: {city_name}")
        return []
    
    # Select random samples for this city
    sample_indices = random.sample(city_indices, min(num_samples, len(city_indices)))
    
    # Get samples
    if isinstance(dataset, Subset):
        samples = [dataset.dataset[idx] for idx in sample_indices]
    else:
        samples = [dataset[idx] for idx in sample_indices]
    
    # Prepare features
    features = torch.stack([sample['features'] for sample in samples]).to(device)
    
    # Generate text
    model.eval()
    with torch.no_grad():
        generated_tokens = model.generate(
            features,
            max_length=max_length,
            token_mappings=token_mappings
        )
        
        # Convert tokens to text
        generated_texts = []
        
        for tokens in generated_tokens:
            # Map tokens back to original vocabulary
            original_tokens = [token_mappings['reverse_token_id_map'][t.item()] for t in tokens]
            
            # Decode to text
            text = tokenizer.decode(original_tokens, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts


def list_available_cities(dataset, min_samples=1):
    """
    List all cities in the dataset with at least min_samples.
    
    Args:
        dataset: Weather dataset
        min_samples: Minimum number of samples required
        
    Returns:
        list: List of (city_name, sample_count) tuples
    """
    if isinstance(dataset, Subset):
        # For a Subset, we need to use the parent dataset's city mapping
        parent_dataset = dataset.dataset
        valid_indices = set(dataset.indices)
        
        # Filter city indices to only include those in this subset
        city_counts = []
        for city, indices in parent_dataset.city_to_indices.items():
            valid_city_indices = [idx for idx in indices if idx in valid_indices]
            if len(valid_city_indices) >= min_samples:
                city_counts.append((city, len(valid_city_indices)))
    else:
        # For regular dataset, use city_to_indices directly
        city_counts = [
            (city, len(indices)) 
            for city, indices in dataset.city_to_indices.items() 
            if len(indices) >= min_samples
        ]
    
    # Sort by city name
    return sorted(city_counts, key=lambda x: x[0])


def get_city_samples(dataset, city_name, max_samples=None):
    """
    Get samples for a specific city from the dataset.
    
    Args:
        dataset: Weather dataset
        city_name: Name of the city
        max_samples: Maximum number of samples to return (None for all)
        
    Returns:
        list: List of samples for the specified city
    """
    if isinstance(dataset, Subset):
        # For a Subset, filter samples that are in both the city's indices and the subset's indices
        parent_dataset = dataset.dataset
        subset_indices = set(dataset.indices)
        city_indices = parent_dataset.city_to_indices.get(city_name, [])
        valid_indices = [idx for idx in city_indices if idx in subset_indices]
        
        if max_samples is not None and max_samples < len(valid_indices):
            valid_indices = random.sample(valid_indices, max_samples)
            
        return [parent_dataset[idx] for idx in valid_indices]
    else:
        # For regular dataset, use city_to_indices directly
        city_indices = dataset.city_to_indices.get(city_name, [])
        
        if max_samples is not None and max_samples < len(city_indices):
            city_indices = random.sample(city_indices, max_samples)
            
        return [dataset[idx] for idx in city_indices]