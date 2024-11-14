import os
import pandas as pd

files = os.listdir(f"{os.getcwd()}/data/wikipedia_city_gathering/formatted")
merged_df = pd.DataFrame(columns=['City','Country', 'Scraping Link'])

for file in files:
    print(file)
    file_df = pd.read_csv(f"{os.getcwd()}/data/wikipedia_city_gathering/formatted/{file}")
    file_df = file_df.drop_duplicates()
    if file != 'South_America.csv':
        file_df['Country'] = file[:-4].replace("_", " ")
    merged_df = pd.concat([merged_df, file_df])

merged_df.to_csv("merged.csv", index = False)