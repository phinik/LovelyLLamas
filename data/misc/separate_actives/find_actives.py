import os
import pandas as pd
import json
from tqdm import tqdm
from pprint import pprint
# Set current working directory
cwd = os.getcwd()

# Get JSON folders
json_folders = {
    # "2024-12-09": [file for file in os.listdir(f"{cwd}/data/2024-12-09") if "standardised" not in file],
    "2024-12-12": [file for file in os.listdir(f"{cwd}/data/2024-12-12") if "standardised" not in file]
}
starts_at_zero = {
    True : 0,
    False: 0
}
rest = 0
other_timezones = dict()
# Iterate through the JSON folders dynamically
for key, file_list in json_folders.items():
    for file_name in tqdm(file_list):
        file_path = f"{cwd}/data/{key}/{file_name}"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = json.load(f)
                if (str(file_content['tabular']['overview']).split('\n')[1]).startswith('01'):
                    starts_at_zero[False] += 1
                    # print((str(file_content['tabular']['overview']).split('\n')[1])[0:10])
                elif (str(file_content['tabular']['overview']).split('\n')[1]).startswith('00'):
                    starts_at_zero[True] += 1
                else:
                    rest += 1
                    if (str(file_content['tabular']['overview']).split('\n')[1])[0:11] not in other_timezones.keys():
                        other_timezones[(str(file_content['tabular']['overview']).split('\n')[1])[0:11]] = 1
                    else:
                        other_timezones[(str(file_content['tabular']['overview']).split('\n')[1])[0:11]] += 1

print(starts_at_zero)
print(rest)
pprint(other_timezones)
exit()

# Function to check if a report exists dynamically
def check_if_report_exists(identifier: str) -> bool:
    
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
                    print(f"File doesn't exist: ", file_path)
                    return False
    return False


dataframe = pd.read_csv(f"{cwd}/data/misc/crawled_information/city_data.csv")

# Apply the check dynamically across the DataFrame
tqdm.pandas(desc=f"Processing city_data.csv")
dataframe["Weather Report?"] = dataframe["Identifier"].progress_apply(check_if_report_exists)

# Save the updated DataFrame
dataframe.to_csv(f"{cwd}/data/misc/crawled_information/city_data.csv", index=False)
print(f"Processed city_data.csv")
