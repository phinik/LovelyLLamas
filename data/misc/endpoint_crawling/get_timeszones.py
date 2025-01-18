import os
import pandas as pd
import requests
from lxml import etree
from timezonefinder import TimezoneFinder
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

def process_city(row: pd.Series, tf: TimezoneFinder) -> Dict[str, Any]:
    """Process a single city and return its data."""
    city = row["City"]
    url = row["URL"]
    result = {
        "City": city,
        "URL": url,
        "Latitude": None,
        "Longitude": None,
        "Timezone": None,
        "Error": None
    }
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.ok:
            dom = etree.HTML(response.content)
            latitude = dom.xpath('/html/head/meta[14]')[0].get("content")
            longitude = dom.xpath('/html/head/meta[15]')[0].get("content")
            
            lat_float = float(latitude)
            lng_float = float(longitude)
            
            result.update({
                "Latitude": lat_float,
                "Longitude": lng_float,
                "Timezone": tf.timezone_at(lat=lat_float, lng=lng_float)
            })
        else:
            result["Error"] = f"Failed to retrieve data. Status code: {response.status_code}"
            
    except Exception as e:
        result["Error"] = str(e)
    
    return result

def main():
    # Load city data from CSV
    df = pd.read_csv(f"{os.getcwd()}/data/city_data.csv")
    
    # Initialize TimezoneFinder (thread-safe, can be shared)
    tf = TimezoneFinder()
    
    # Store results
    results = []
    
    # Configure the progress bar
    pbar = tqdm(total=len(df), desc="Processing Cities")
    
    # Use ThreadPoolExecutor for parallel processing
    # Number of workers is 3x CPU cores for I/O-bound tasks
    max_workers = os.cpu_count() * 3
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_city = {
            executor.submit(process_city, row, tf): row["City"]
            for _, row in df.iterrows()
        }
        
        # Process completed tasks
        for future in as_completed(future_to_city):
            city = future_to_city[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Critical error processing {city}: {e}")
                results.append({
                    "City": city,
                    "Error": f"Critical error: {str(e)}",
                    "Latitude": None,
                    "Longitude": None,
                    "Timezone": None
                })
            finally:
                pbar.update(1)
    
    pbar.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print errors if any
    errors = results_df[results_df["Error"].notna()]
    if not errors.empty:
        print("\nErrors encountered:")
        for _, row in errors.iterrows():
            print(f"{row['City']}: {row['Error']}")
    
    # Clean and sort the data
    final_df = results_df.dropna(subset=["Timezone"]).sort_values(by="Timezone").reset_index(drop=True)
    
    # Save to CSV
    final_df.to_csv("city_data_tz.csv", index=False)
    print(f"\nProcessed {len(final_df)} cities successfully")
    print(f"Results saved to city_data_tz.csv")
    
    return final_df

if __name__ == "__main__":
    df_sorted = main()
    print("\nFinal sorted dataframe:")
    print(df_sorted)