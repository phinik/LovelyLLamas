import pandas as pd
from lxml import etree
import requests
import os
from tqdm import tqdm  # Import tqdm

def get_options(url: str) -> tuple[list, list]:
    response = requests.get(url)
    if response.ok:
        dom = etree.HTML(response.content)
        options: list = [str(option.get("value")) for option in dom.xpath('//option') if option is not None and (str(option.get("value")).startswith("/") or str(option.get("value")) == "")]
        newline_indexes: list = [i for i, x in enumerate(options) if str(x) == '']
        return options, newline_indexes
    
# Create an empty DataFrame with specified columns
city_data = pd.DataFrame(columns=["City", "Region", "Country", "Continent", "Link"])

# Load regions from a CSV file
region_data = pd.read_csv(f"{os.getcwd()}/data/endpoint_crawling/regions.csv")

# List to hold new city data rows
city_rows = []

# Iterate through the region data and fetch the cities, using tqdm to show progress
for i in tqdm(range(len(region_data)), desc="Fetching Region Data", unit="region"):
    options, indices = get_options(region_data.loc[i, "Link"])
    
    # Ensure that indices are valid
    if len(indices) > 0:
        cities = options[indices[-1]+1:]  # Extract cities starting from the last index onward
        # Use tqdm to show progress when iterating over cities
        for city in tqdm(cities, desc=f"Fetching Cities for {region_data.loc[i, 'Region']}", unit="city", leave=False):
            city_name = str(city).split("/")[2]  # Assuming the city is the third part in the path
            region_name = region_data.loc[i, "Region"]
            country_name = region_data.loc[i, "Country"]
            continent_name = region_data.loc[i, "Continent"]
            
            # Create a new row for city data
            new_row = {
                "City": city_name,
                "Region": region_name,
                "Country": country_name,
                "Continent": continent_name,
                "Link": city
            }
            
            # Add the new row to the city_rows list
            city_rows.append(new_row)

# Convert city_rows list to DataFrame
city_df = pd.DataFrame(city_rows)

# Concatenate the new data with the existing city_data DataFrame
city_data = pd.concat([city_data, city_df], ignore_index=True)

# Print the city data DataFrame
print(city_data)

# Optionally, you can save the DataFrame to a CSV file
city_data.to_csv("city_data.csv", index=False)

print("City data has been exported to city_data.csv.")
