import os
import json
import re
import sentencepiece as spm
from typing import List, Dict, Union
from tqdm import tqdm
import spacy
from multiprocessing import Pool, cpu_count

# Load the German NLP model
nlp = spacy.load("de_core_news_sm")

# Dict for storing Stats
stats = {
    'clearness': (),
    'temperatur_min': float('inf'),
    'temperatur_max': -float('inf'),
    'niederschlagsmenge_in_l_per_sqm_min': float('inf'),
    'niederschlagsmenge_in_l_per_sqm_max': -float('inf'),
    'windgeschwindigkeit_min': float('inf'),
    'windgeschwindigkeit_max': -float('inf'),
    'luftdruck_min': float('inf'),
    'luftdruck_max': -float('inf'),
}

# Utility functions
def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\u2702-\u27B0"  # dingbats
        u"\u24C2-\U0001F251"  # enclosed characters
        u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r"", text)

def replace_city_and_units(text: str, city: str) -> str:
    text = text.replace(city, "[STADT]")
    unit_patterns = [
    # TEMPERATURE
    (r'zwischen [-]*\d+ und [-]*\d+[ ]*(°[ ]*C|Grad)', 'zwischen [TEMP] und [TEMP]'),
    # (r'zwischen [-]*\d+ und [-]*\d+[ ]*Grad', 'zwischen [TEMP] und [TEMP]'),
    (r'[-]*\d+ bis [-]*\d+[ ]*(°[ ]*C|Grad)', '[TEMP] bis [TEMP]'),
    (r'[-]*\d+ bis zu [-]*\d+[ ]*°[ ]*C', '[TEMP] bis zu [TEMP]'),
    (r'[-]*\d+[ ]*(°[ ]*C|Grad)', '[TEMP]'),
    # VELOCITY
    (r'zwischen [-]*\d+ und [-]*\d+[ ]*km/h', 'zwischen [VELOCITY] und [VELOCITY]'),
    # PERCENTILE
    (r'\d+[ ]*%', '[PERCENTILE]'),
    # RAINFALL DELTA
    (r'\d+\.\d+[ ]*l\/m²', '[RAINFALL]')
    ]
    for pattern, replacement in unit_patterns:
        text = re.sub(pattern, replacement, text)

    # REMOVE MARKUP
    text = re.sub(r'\**', '', text)

    # REMOVE UNNECESSARY NEWLINES
    text = re.sub(r'\n\n', '\n', text)

    # REMOVE SPACE AFTER NEWLINE
    text = re.sub(r'\n ', '\n', text)

    # REPLACE MULTIPLE WHITESPACES WITH ONE
    text = re.sub(r' +', ' ', text)
    return text

def replace_dates(text: str) -> str:
    text = re.sub(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b", "[DATE]", text)
    return text

def contains_chinese_or_russian(text: str) -> bool:
    chinese_russian_pattern = re.compile(r"[\u4E00-\u9FFF\u0400-\u04FF]")
    return bool(chinese_russian_pattern.search(text))

def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

def gather_stats(data: Dict[str, Union[str, List]]) -> None:
    temps = data.get('temperatur_in_deg_C', [])
    if isinstance(temps, list) and len(temps) > 0:
        temps = [int(temp) for temp in temps if str(temp).isdigit()]
        if temps:
            if max(temps) > stats['temperatur_max']:
                stats['temperatur_max'] = max(temps)
            if min(temps) < stats['temperatur_min']:
                stats['temperatur_min'] = min(temps)

# Dataloader
def load_json_files(directory: str) -> List[Dict[str, Union[str, List]]]:
    data = []
    for filename in tqdm(os.listdir(directory), desc="Loading JSON files"):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                try:
                    content = json.load(file)
                    data.append(content)
                except json.JSONDecodeError:
                    print(f"Failed to decode {filepath}")
    return data

def process_single_record(record: Dict[str, Union[str, List]]) -> List[str]:
    """Process a single JSON record and return its processed texts."""
    processed_texts = []
    city = record.get("city", "")
    gather_stats(record)
    if not city:
        return processed_texts
        
    texts_to_process = [
        record.get("report_short", ""),
        record.get("report_long", ""),
        record.get("report_short_wout_boeen", ""),
        record.get("gpt_rewritten", "")
    ]

    for text in texts_to_process:
        if not text or contains_chinese_or_russian(text):
            continue

        text = remove_emojis(text)
        text = replace_city_and_units(text, city)
        text = replace_dates(text)
        text = lemmatize_text(text)

        # Fix [STADT] formatting after lemmatization
        text = re.sub(r'\[ ', '[', text)
        text = re.sub(r' \]', ']', text)

        if "[STADT]" not in text:
            continue

        processed_texts.append(text)
        
    return processed_texts

def process_texts(data: List[Dict[str, Union[str, List]]]) -> List[str]:
    """Process texts using multiprocessing - one record per process."""
    num_processes = cpu_count() - 1  # Leave one CPU free
    print(f"Processing {len(data)} records using {num_processes} processes")
    
    # Process records in parallel
    with Pool(processes=num_processes) as pool:
        # Map each record to a process
        results = list(tqdm(
            pool.imap(process_single_record, data),
            total=len(data),
            desc="Processing records"
        ))
        
    # Flatten the results
    return [text for record_texts in results for text in record_texts]

def train_tokenizer(texts: List[str], model_prefix: str, vocab_size: int = 27458):
    with open("temp_corpus.txt", "w", encoding="utf-8") as f:
        for text in tqdm(texts, desc="Writing texts to temp corpus"):
            f.write(text + "\n")

    spm.SentencePieceTrainer.train(
        input="temp_corpus.txt",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        user_defined_symbols=["[STADT]", "[TEMP]", "[DATE]", "[VELOCITY]", "[PERCENTILE]", "[RAINFALL]"]
    )

    # os.remove("temp_corpus.txt")

# Main execution
def main():
    directory = os.path.join(os.getcwd(), 'data', '2024-12-12') # Replace with the actual directory

    print("Loading JSON files...")
    data = load_json_files(directory)

    print("Processing texts...")
    processed_texts = process_texts(data)

    print(f"Processed {len(processed_texts)} texts. Training tokenizer...")
    train_tokenizer(processed_texts, model_prefix="weather_tokenizer")

    print("Tokenizer training complete.")
    print(stats)

if __name__ == "__main__":
    main()
