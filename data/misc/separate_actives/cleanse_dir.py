import os
import json
from tqdm import tqdm

def cleanse_folder(folder_path: str):
    """
    Deletes JSON files in the specified folder if their 'weather_description' key inside 'strings' is None.
    
    :param folder_path: Path to the folder containing JSON files.
    """
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Get all JSON files in the folder
    json_files = [file for file in os.listdir(folder_path) if file.endswith(".json") and "standardised" not in file]

    print(f"Processing {len(json_files)} files in folder: {folder_path}")

    # Iterate over files with tqdm for progress tracking
    for file_name in tqdm(json_files, desc="Cleansing files"):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Open and parse the JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = json.load(f)

            # Check the "weather_description" value in "strings"
            weather_description = file_content.get("strings", {}).get("weather_description")

            # If "weather_description" is None, delete the file
            if weather_description is None:
                os.remove(file_path)
                os.remove(f"{file_path[:-5]}_standardised.json")
                tqdm.write(f"Deleted file: {file_name}")
        except Exception as e:
            tqdm.write(f"Error processing file {file_name}: {e}")

    print("Cleansing complete.")

# Example usage
if __name__ == "__main__":
    folder_to_cleanse = f"{os.getcwd()}/data/2024-11-19/"
    cleanse_folder(folder_to_cleanse)
