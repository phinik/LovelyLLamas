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
        city_id = str(city_data['URL']).split("/")[-1][:-5]
        output_file = f"{output_dir}/{city_name}-{city_id}.json"
        
        # Skip if file already exists
        if os.path.exists(output_file):
            logging.info(f"Data already exists for {city_name}, skipping...", 
                        extra={'worker_id': thread_local.worker_id})
            return

        logging.info(f"Starting to process {city_name}", 
                    extra={'worker_id': thread_local.worker_id})
        
        extractor = WeatherDataExtractor(city_data['URL'], False)
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

def extract_weather_data(city_list_path: str, num_threads: int = 4) -> None:
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
    import datetime
    import tzlocal
    from zoneinfo import ZoneInfo
    import multiprocessing

    def get_current_utc_offset(timezone_obj) -> float:
        """
        Get the current UTC offset for a timezone in hours.
        Returns offset as float (e.g., 2.0 for UTC+2, -4.5 for UTC-4:30)
        """
        try:
            # Get current offset in seconds
            offset = timezone_obj.utcoffset(datetime.datetime.now()).total_seconds()
            # Convert to hours and return as float
            return offset / 3600.0
        except Exception as e:
            print(f"Error getting offset: {e}")
            return None

    def get_next_timezone_offset(current_time: datetime.datetime, timezone_str: str) -> float:
        """
        Given the current time and timezone, determine the UTC offset of the timezone
        that will reach midnight (00:00) next.
        The UTC offset is constrained between UTC-10 and UTC+14 and rounded
        up to the next hour if the difference to midnight is greater than 30 minutes.
        """
        try:
            local_timezone = ZoneInfo(timezone_str)

            # Step 1: Get the current UTC offset for the timezone
            current_offset = get_current_utc_offset(local_timezone)
            
            # Step 2: Calculate midnight of the next day
            # If current time is 23:00, midnight_today should be 2024-11-18 00:00
            midnight_today = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            if current_time > midnight_today:  # If the current time is after midnight today
                midnight_today = midnight_today + datetime.timedelta(days=1)  # Move to next day

            # Step 3: Calculate the time difference between now and the next midnight
            time_to_midnight = midnight_today - current_time

            # Step 4: Determine whether to round up or down based on the time difference
            if abs(time_to_midnight.total_seconds()) > 1800:  # more than 30 minutes
                next_offset = current_offset + 1 if time_to_midnight.total_seconds() > 0 else current_offset - 1
            else:
                next_offset = current_offset  # No rounding needed if the difference is <= 30 minutes
            
            # Step 5: Ensure the offset is within the range UTC-10 to UTC+14
            next_offset = max(-10, min(14, next_offset))  # Clamp the offset between -10 and +14

            return next_offset

        except Exception as e:
            print(f"Error calculating next timezone offset for {timezone_str}: {e}")
            return None

    # Main execution
    local_timezone = tzlocal.get_localzone()  # Get local timezone of the server
    timezone_str = str(local_timezone)       # Convert to string format

    # Get the current time in the local timezone
    current_time = datetime.datetime.now(local_timezone)

    # Determine the next timezone's offset based on the current time
    next_offset = get_next_timezone_offset(current_time, timezone_str)

    if next_offset is not None:
        print(f"The UTC offset of the timezone with the next midnight is: UTC{next_offset:+.1f}")
        prefix = "minus" if next_offset < 0 else "plus"
        file_name = f"{os.getcwd()}/data/timezone_splits/UTC_{prefix}_{str(next_offset).replace(".","_")}.csv"
       
        # Check if the file exists before proceeding with the extraction
        if os.path.exists(file_name):
            print("File exists. Proceeding with the extraction.")
            extract_weather_data(city_list_path=file_name, num_threads=multiprocessing.cpu_count())

    else:
        print("Could not determine the next timezone's offset.")