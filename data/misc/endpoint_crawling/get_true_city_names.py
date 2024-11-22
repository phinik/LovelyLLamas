import pandas as pd
from tqdm import tqdm
from lxml import etree
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

# Load city data
cities = pd.read_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data.csv")

# Add Identifier column
cities["Identifier"] = cities["URL"].apply(lambda x: (str(x).split(".")[-2]).split("/")[-1])

# Fetch city names using ThreadPoolExecutor
cities["True Name"] = process_get_city_name(cities["URL"].tolist())

# Remove rows where 'True Name' is None
cleaned_cities = cities.dropna(subset=["True Name"]).reset_index(drop=True)


# cities = pd.read_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_with_true_names.csv")
# cities = cities.drop_duplicates(subset=["Identifier"])
# cities = cities.drop(columns = ["Unnamed: 0.1"])

cities.to_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_with_true_names.csv", index=False)
# print(cleaned_cities)
