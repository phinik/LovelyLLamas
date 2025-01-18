import pandas as pd
from tqdm import tqdm
from lxml import etree
from datetime import datetime
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_city_name(url: str):
    try:
        response = requests.get(url, timeout=10)  # Adding a timeout for robustness
        if response.ok:
            dom = etree.HTML(response.content)
            element_text: str = str(dom.xpath('//*[@id="wetter"]//text()')[0]).replace("Wie wird das Wetter heute in ", "")[:-1]
            return element_text
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def process_get_city_name(urls):
    results = [None] * len(urls)  # Preallocate a list with placeholders
    with ThreadPoolExecutor() as executor:
        # Submit tasks with their indices for matching results
        future_to_index = {executor.submit(get_city_name, url): idx for idx, url in enumerate(urls)}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Fetching city names"):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"Failed to fetch data for {urls[index]}: {e}")
    return results

def get_city_time(url: str):
    try:
        response = requests.get(url, timeout=10)  # Adding a timeout for robustness
        if response.ok:
            dom = etree.HTML(response.content)
            element_text: str = str(dom.xpath('//*[@id="uebersicht"]/div[1]/div[1]//text()')[0])
            # print(element_text)
            return f"{element_text} | {datetime.now().hour} | {datetime.now().minute}"
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def process_get_city_time(urls):
    results = [None] * len(urls)  # Preallocate a list with placeholders
    with ThreadPoolExecutor() as executor:
        # Submit tasks with their indices for matching results
        future_to_index = {executor.submit(get_city_time, url): idx for idx, url in enumerate(urls)}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Fetching city timezones"):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"Failed to fetch data for {urls[index]}: {e}")
    return results

# Load city data
cities = pd.read_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_with_true_names.csv")

# Add Identifier column
# cities["Identifier"] = cities["URL"].apply(lambda x: (str(x).split(".")[-2]).split("/")[-1])

# Fetch city names using ThreadPoolExecutor
cities["Time at Scrape"] = process_get_city_time(cities["URL"].tolist())

# Remove rows where 'True Name' is None
cleaned_cities = cities.dropna(subset=["Time at Scrape"]).reset_index(drop=True)

# cities = pd.read_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_with_true_names.csv")
# cities = cities.drop_duplicates(subset=["Identifier"])
# cities = cities.drop(columns = ["Unnamed: 0.1"])

cities.to_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_with_true_names_and_times.csv", index=False)
# print(cleaned_cities)
