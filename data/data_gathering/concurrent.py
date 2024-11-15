from data.weather_extractor.extractor import WeatherDataExtractor
import pandas as pd
import datetime
import os
import threading
from queue import Queue
from typing import List
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Worker %(worker_id)s] - %(message)s'
)

# Create a thread-local storage for worker IDs
thread_local = threading.local()

def ensure_directory_exists(directory_path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}", extra={'worker_id': 'MAIN'})
    else:
        logging.info(f"Directory already exists: {directory_path}", extra={'worker_id': 'MAIN'})

def process_city(city_data: pd.Series, output_dir: str) -> None:
    """Process a single city's weather data."""
    try:
        city_name = city_data['City']
        output_file = f"{output_dir}/{city_name}.json"
        
        # Skip if file already exists
        if os.path.exists(output_file):
            logging.info(f"Data already exists for {city_name}, skipping...", 
                        extra={'worker_id': thread_local.worker_id})
            return

        logging.info(f"Starting to process {city_name}", 
                    extra={'worker_id': thread_local.worker_id})
        
        extractor = WeatherDataExtractor(city_data['URL'])
        extractor.save_data_to_json(output_file)
        
        logging.info(f"Successfully processed {city_name}", 
                    extra={'worker_id': thread_local.worker_id})
    
    except Exception as e:
        logging.error(f"Error processing {city_name}: {str(e)}", 
                     extra={'worker_id': thread_local.worker_id})

def worker(queue: Queue, output_dir: str, worker_id: int) -> None:
    """Worker function for thread pool."""
    # Set worker ID in thread-local storage
    thread_local.worker_id = f"Thread-{worker_id}"
    
    logging.info("Worker started", extra={'worker_id': thread_local.worker_id})
    
    while True:
        try:
            # Get task with timeout
            city_data = queue.get(timeout=5)  # 5 second timeout
            if city_data is None:
                logging.info("Received shutdown signal", 
                           extra={'worker_id': thread_local.worker_id})
                queue.task_done()
                break
                
            process_city(city_data, output_dir)
            queue.task_done()
            
        except queue.Empty:
            logging.info("No tasks received for 5 seconds, shutting down", 
                        extra={'worker_id': thread_local.worker_id})
            break
            
    logging.info("Worker shutting down", extra={'worker_id': thread_local.worker_id})

def extract_weather_data(city_list_path: str, num_threads: int = 12) -> None:
    """
    Main function to extract weather data concurrently.
    
    Args:
        city_list_path: Path to CSV file containing city data
        num_threads: Number of concurrent threads to use
    """
    try:
        start_time = time.time()
        
        # Read city list
        city_list = pd.read_csv(city_list_path)
        logging.info(f"Loaded {len(city_list)} cities from {city_list_path}", 
                    extra={'worker_id': 'MAIN'})

        # Create output directory
        today = datetime.date.today()
        output_dir = f"{os.getcwd()}/data/{today}"
        ensure_directory_exists(output_dir)

        # Create queue and threads
        queue: Queue = Queue()
        threads: List[threading.Thread] = []

        # Start worker threads
        for i in range(num_threads):
            thread = threading.Thread(
                target=worker,
                args=(queue, output_dir, i+1),
                daemon=True
            )
            thread.start()
            threads.append(thread)

        # Add cities to queue
        tasks_added = 0
        for _, city_data in city_list.iterrows():
            queue.put(city_data)
            tasks_added += 1

        # Add sentinel values to stop threads
        for _ in range(num_threads):
            queue.put(None)

        # Wait for all tasks to complete with timeout
        queue.join()
        
        # Wait for all threads to finish with timeout
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout for each thread

        end_time = time.time()
        duration = end_time - start_time
        
        logging.info(
            f"Weather data extraction completed. Processed {tasks_added} cities in {duration:.2f} seconds", 
            extra={'worker_id': 'MAIN'}
        )

    except Exception as e:
        logging.error(f"Error in main process: {str(e)}", extra={'worker_id': 'MAIN'})
        raise
    finally:
        # Check if any threads are still alive
        alive_threads = [t for t in threads if t.is_alive()]
        if alive_threads:
            logging.warning(
                f"{len(alive_threads)} threads still running. They will be terminated.", 
                extra={'worker_id': 'MAIN'}
            )

if __name__ == "__main__":
    extract_weather_data("data/city_data.csv")