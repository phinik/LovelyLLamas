from deep_translator import GoogleTranslator, MyMemoryTranslator
from tqdm import tqdm
import pandas as pd
import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Optional, Dict, Any
import random
import backoff

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='translation.log'
)

class RateLimiter:
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = {}
        self.lock = threading.Lock()

    def wait(self, thread_id: int):
        with self.lock:
            current_time = time.time()
            if thread_id in self.last_call_time:
                elapsed = current_time - self.last_call_time[thread_id]
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed + random.uniform(0.1, 0.3))
            self.last_call_time[thread_id] = time.time()

class MultiTranslator:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.translation_queue = Queue()
        self.rate_limiter = RateLimiter(calls_per_second=3)  # Conservative rate limit
        
        # Create separate translator instances for each thread
        self.translator_pools = {
            'mymemory': [
                MyMemoryTranslator(source='en-GB', target='de-DE')
                for _ in range(max_workers)
            ]
        }
    
    def get_translator(self, service: str, thread_id: int) -> Any:
        """Get a translator instance for the current thread."""
        return self.translator_pools[service][thread_id % self.max_workers]

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=5,
        max_time=300,
        giveup=lambda e: "quota exceeded" in str(e).lower()
    )
    def _try_translate(self, text: str, thread_id: int) -> str:
        """Attempt translation with exponential backoff retry."""
        self.rate_limiter.wait(thread_id)
        translator = self.get_translator('mymemory', thread_id)
        return translator.translate(text)

    def translate_single(self, text: str, thread_id: int) -> Dict[str, Any]:
        """Translate text using available services with improved error handling."""
        if not text or not isinstance(text, str):
            return {'original': text, 'index': thread_id}
            
        # Check cache
        with self.cache_lock:
            if text in self.cache:
                result = self.cache[text].copy()
                result['index'] = thread_id
                return result
            
        results = {'original': text, 'index': thread_id}
            
        # Try MyMemory with retries
        try:
            results['mymemory'] = self._try_translate(text, thread_id)
        except Exception as e:
            logging.error(f"Translation failed after retries for '{text}': {e}")
            results['mymemory'] = text
        
        # Update cache
        with self.cache_lock:
            cache_result = results.copy()
            del cache_result['index']
            self.cache[text] = cache_result
            
        return results

def translate_locations(
    input_file: str,
    output_file: str,
    max_workers: int = 4,
    batch_size: Optional[int] = None
):
    """Translate locations using multiple services and save results with batching support."""
    try:
        # Load data
        df = pd.read_csv(input_file)
        
        # Combine city and country
        df['Location'] = df['City'] + ' ' + df['Country']
        
        # Initialize translator
        translator = MultiTranslator(max_workers=max_workers)
        
        # Prepare data for parallel processing
        locations = df['Location'].tolist()
        
        if batch_size:
            # Process in batches
            for start_idx in range(0, len(locations), batch_size):
                end_idx = min(start_idx + batch_size, len(locations))
                batch_locations = locations[start_idx:end_idx]
                
                results = [None] * len(batch_locations)
                
                # Progress bar for current batch
                pbar = tqdm(
                    total=len(batch_locations),
                    desc=f"Translating batch {start_idx//batch_size + 1}"
                )
                
                def translate_with_progress(args):
                    text, thread_id = args
                    result = translator.translate_single(text, thread_id)
                    pbar.update(1)
                    return result
                
                # Process translations in parallel for current batch
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_text = {
                        executor.submit(
                            translate_with_progress,
                            (text, i)
                        ): i
                        for i, text in enumerate(batch_locations)
                    }
                    
                    for future in as_completed(future_to_text):
                        result = future.result()
                        original_index = result['index']
                        del result['index']
                        results[original_index] = result
                
                pbar.close()
                
                # Update DataFrame with batch results
                for key in results[0].keys():
                    df.loc[start_idx:end_idx-1, f'translated_{key}'] = [
                        r[key] for r in results
                    ]
                
                # Save intermediate results
                df.to_csv(output_file, index=False)
                logging.info(f"Saved batch {start_idx//batch_size + 1} results")
                
                # Add delay between batches
                if end_idx < len(locations):
                    time.sleep(5)  # Cool-down period between batches
        
        else:
            # Process all at once (original behavior)
            results = [None] * len(locations)
            pbar = tqdm(total=len(locations), desc="Translating locations")
            
            def translate_with_progress(args):
                text, thread_id = args
                result = translator.translate_single(text, thread_id)
                pbar.update(1)
                return result
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_text = {
                    executor.submit(translate_with_progress, (text, i)): i
                    for i, text in enumerate(locations)
                }
                
                for future in as_completed(future_to_text):
                    result = future.result()
                    original_index = result['index']
                    del result['index']
                    results[original_index] = result
            
            pbar.close()
            
            # Convert results to DataFrame columns
            for key in results[0].keys():
                df[f'translated_{key}'] = [r[key] for r in results]
            
            # Save results
            df.to_csv(output_file, index=False)
        
        print(f"\nTranslations saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"Translation process failed: {e}")
        raise

if __name__ == "__main__":
    input_path = f"{os.getcwd()}/data/wikipedia_city_gathering/merged.csv"
    output_path = f"{os.getcwd()}/data/wikipedia_city_gathering/merged_multi_translated.csv"
    
    # Process in batches of 100 with 4 workers
    translate_locations(
        input_path,
        output_path,
        max_workers=4,
        batch_size=100
    )