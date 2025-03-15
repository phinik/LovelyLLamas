from deep_translator import GoogleTranslator, MyMemoryTranslator
from tqdm import tqdm
import pandas as pd
import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='translation.log'
)

class MultiTranslator:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.translation_queue = Queue()
        
        # Create separate translator instances for each thread
        self.translator_pools = {
            # 'google': [
            #     GoogleTranslator(source='en', target='de')
            #     for _ in range(max_workers)
            # ],
            'mymemory': [
                MyMemoryTranslator(source='en-GB', target='de-DE')
                for _ in range(max_workers)
            ]
        }
        
    def get_translator(self, service, thread_id):
        """Get a translator instance for the current thread."""
        return self.translator_pools[service][thread_id % self.max_workers]
        
    def translate_single(self, text: str, thread_id: int) -> dict:
        """Translate text using available services."""
        if not text or not isinstance(text, str):
            return {'original': text}
            
        # Check cache
        with self.cache_lock:
            if text in self.cache:
                return self.cache[text]
            
        results = {'original': text}
        
        # Try Google Translate
        # try:
        #     translator = self.get_translator('google', thread_id)
        #     results['google'] = translator.translate(text)
        #     time.sleep(0.2)  # Reduced rate limiting due to multiple threads
        # except Exception as e:
        #     logging.warning(f"Google translation failed for '{text}': {e}")
        #     results['google'] = text
            
        # Try MyMemory
        try:
            translator = self.get_translator('mymemory', thread_id)
            results['mymemory'] = translator.translate(text)
            time.sleep(0.2)  # Reduced rate limiting due to multiple threads
        except Exception as e:
            logging.warning(f"MyMemory translation failed for '{text}': {e}")
            results['mymemory'] = text
        
        # Update cache
        with self.cache_lock:
            self.cache[text] = results
            
        return results

def translate_locations(input_file: str, output_file: str, max_workers: int = 4):
    """Translate locations using multiple services and save results."""
    try:
        # Load data
        df = pd.read_csv(input_file)
        
        # Combine city and country
        df['Location'] = df['City'] + ' ' + df['Country']
        
        # Initialize translator
        translator = MultiTranslator(max_workers=max_workers)
        
        # Prepare data for parallel processing
        locations = df['Location'].tolist()
        results = []
        
        # Progress bar setup
        pbar = tqdm(total=len(locations), desc="Translating locations")
        
        def translate_with_progress(args):
            text, thread_id = args
            result = translator.translate_single(text, thread_id)
            pbar.update(1)
            return result
        
        # Process translations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks with thread IDs
            future_to_text = {
                executor.submit(translate_with_progress, (text, i)): text
                for i, text in enumerate(locations)
            }
            
            # Collect results in order
            for future in as_completed(future_to_text):
                results.append(future.result())
        
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
    input_path = f"{os.getcwd()}/data/wikipedia_city_gathering/translated_example/merged_snippet.csv"
    output_path = f"{os.getcwd()}/data/wikipedia_city_gathering/merged_multi_translated.csv"
    
    # You can adjust the number of workers based on your CPU and the API limits
    translate_locations(input_path, output_path, max_workers=4)