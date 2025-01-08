import os
import re
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Union
from loader import (
    remove_emojis,
    replace_city_and_units,
    replace_dates,
    contains_chinese_or_russian,
    lemmatize_text
)

# Helper Functions
def encode_time_as_circle(time_str: str) -> Dict[str, float]:
    """Encode time as sine and cosine values on a circular scale."""
    try:
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        seconds_since_midnight = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        seconds_in_day = 24 * 3600
        angle = 2 * np.pi * seconds_since_midnight / seconds_in_day
        return {"sin_time": np.sin(angle), "cos_time": np.cos(angle)}
    except ValueError:
        return {"sin_time": 0.0, "cos_time": 0.0}

def normalize_values(values: list, min_val: float = None, max_val: float = None) -> list:
    """Normalize numerical values to a 0-1 range."""
    if not values:
        return []
    if min_val is None:
        min_val = min(map(float, values))
    if max_val is None:
        max_val = max(map(float, values))
    return [(float(val) - min_val) / (max_val - min_val) for val in values]

def one_hot_encode(categories: list, unique_values: list) -> list:
    """One-hot encode categorical data."""
    encoded = []
    for category in categories:
        encoding = [1 if category == value else 0 for value in unique_values]
        encoded.append(encoding)
    return encoded

def preprocess_text(text: str, city: str) -> str:
    """Preprocess textual data using loader functions."""
    text = remove_emojis(text)
    text = replace_city_and_units(text, city)
    text = replace_dates(text)
    text = lemmatize_text(text)
    return text

def preprocess_record(record: Dict[str, Any]) -> Dict[str, Union[str, list, dict]]:
    """Preprocess a single JSON record."""
    city = record.get("city", "")
    preprocessed = {}

    # Textual data
    gpt_rewritten = record.get("gpt_rewritten", "")
    if gpt_rewritten and not contains_chinese_or_russian(gpt_rewritten):
        preprocessed["gpt_rewritten"] = preprocess_text(gpt_rewritten, city)

    # Time and date encoding
    created_time = record.get("created_time", "")
    preprocessed.update(encode_time_as_circle(created_time))

    # Numerical encoding
    preprocessed["temperatur_in_deg_C"] = normalize_values(record.get("temperatur_in_deg_C", []))
    preprocessed["windgeschwindigkeit_in_km_per_s"] = normalize_values(record.get("windgeschwindigkeit_in_km_per_s", []))

    # Categorical encoding
    windrichtung = record.get("windrichtung", [])
    preprocessed["windrichtung"] = one_hot_encode(windrichtung, unique_values=["NO", "N", "NW", "O", "S", "SW", "W", "SO"])

    clearness = record.get("clearness", [])
    preprocessed["clearness"] = one_hot_encode(clearness, unique_values=["Leicht bewölkt", "Bewölkt", "Klar", "Regnerisch"])

    return preprocessed

# Main Preprocessor
def preprocess_data(data: list) -> list:
    """Preprocess a list of records."""
    return [preprocess_record(record) for record in data]

if __name__ == "__main__":
    # Example Usage
    filepath = os.path.join(os.getcwd(), 'data', '2024-12-12', 'aberdeen-AG3576398_standardised.json')  # Replace with actual file path

    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.load(f)]

    preprocessed_data = preprocess_data(data)
    print(json.dumps(preprocessed_data, indent=2, ensure_ascii=False))
