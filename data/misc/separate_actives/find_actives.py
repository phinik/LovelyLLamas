import os
import pandas as pd
import json
from tqdm import tqdm

# Set current working directory
cwd = os.getcwd()

# Get timezone CSV names
timezone_cities_dir = os.listdir(f"{cwd}/data/timezone_splits/")
timezone_cities_dir.remove("_summary.csv")

# Get JSON folders
json_folders = {
    "2024-11-19": [file for file in os.listdir(f"{cwd}/data/2024-11-19") if "standardised" not in file],
    "2024-11-20": [file for file in os.listdir(f"{cwd}/data/2024-11-20") if "standardised" not in file]
}

# Function to check if a report exists dynamically
def check_if_report_exists(url: str) -> bool:
    identifier = url.split(".")[-2].split("/")[-1]
    
    # Iterate through the JSON folders dynamically
    for key, file_list in json_folders.items():
        for file_name in file_list:
            if identifier in file_name:
                file_path = f"{cwd}/data/{key}/{file_name}"
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = json.load(f)
                        # Return True if weather_description is not None
                        # print(file_content["strings"]["weather_description"])
                        return file_content["strings"]["weather_description"] is not None
                else:
                    print(f"File doesn't exist")
                    return False
    return False

# Process each timezone CSV
for file_name in tqdm(timezone_cities_dir, desc="Processing timezone CSVs"):
    dataframe_path = f"{cwd}/data/timezone_splits/{file_name}"
    dataframe = pd.read_csv(dataframe_path)
    
    # Apply the check dynamically across the DataFrame
    tqdm.pandas(desc=f"Processing {file_name}")
    dataframe["Weather Report?"] = dataframe["URL"].progress_apply(check_if_report_exists)
    
    # Save the updated DataFrame
    dataframe.to_csv(dataframe_path, index=False)
    print(f"Processed {file_name}")
