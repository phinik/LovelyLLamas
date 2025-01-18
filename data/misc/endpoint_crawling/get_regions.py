import requests
from lxml import etree
import json
from tqdm import tqdm  # Import tqdm

def get_options(url: str) -> tuple[list, list]:
    response = requests.get(url)
    if response.ok:
        dom = etree.HTML(response.content)
        options: list = [str(option.get("value")) for option in dom.xpath('//option') if option is not None and (str(option.get("value")).startswith("/") or str(option.get("value")) == "")]
        newline_indexes: list = [i for i, x in enumerate(options) if str(x) == '']
        return options, newline_indexes

options, indexes = get_options("https://www.wetter.com/deutschland/EUDE.html")
continents = options[0:indexes[0]]
data: dict = {}

# Use tqdm to show progress when iterating over continents
for continent in tqdm(continents, desc="Continents", unit="continent"):
    continent_name = continent.split("/")[1]
    data[continent_name] = {
        "link": continent
    }
    
    # Fetch countries for each continent and show progress
    options, indexes = get_options(f"https://www.wetter.com{continent}")
    if len(indexes) > 1:
        countries = options[indexes[0]+1:indexes[1]]
    else:
        countries = options[indexes[0]+1:]  # Handle case where there is only one newline

    for country in tqdm(countries, desc=f"Countries in {continent_name}", unit="country", leave=False):
        country_name = country.split("/")[1]
        data[continent_name][country_name] = {
            "link": country
        }
        
        # Fetch regions for each country and show progress
        options, indexes = get_options(f"https://www.wetter.com{country}")
        if len(indexes) > 2:
            regions = options[indexes[1]+1:indexes[2]]
        else:
            regions = options[indexes[1]+1:]  # Handle case where there are not enough newline indexes

        for region in tqdm(regions, desc=f"Regions in {country_name}", unit="city", leave=False):
            region_name = region.split("/")[2]
            data[continent_name][country_name][region_name] = {
                "link": region
            }

# Export the data dictionary to a JSON file
with open("weather_data.json", "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print("Data has been exported to weather_data.json.")

# Export data to csv
import pandas as pd

# Flatten the data into a list of dictionaries
flattened_data = []

for continent, continent_data in data.items():
    for country, country_data in continent_data.items():
        if isinstance(country_data, dict):  # Check if it's a country (it should be)
            for region, region_data in country_data.items():
                if isinstance(region_data, dict):  # Check if it's a city (it should be)
                    flattened_data.append({
                        "Region": str(region).title(),
                        "Country": str(country).title(),
                        "Continent": str(continent).title(),
                        "Link": f"https://wetter.com{region_data.get('link', '')}"
                    })

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(flattened_data)

# Print or save the DataFrame
print(df)

# Optionally, you can save it to a CSV file
df.to_csv("regions.csv", index=False)
