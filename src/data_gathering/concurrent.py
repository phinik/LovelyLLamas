from ..weather_extractor.extractor import WeatherDataExtractor
import pandas as pd
import datetime
import os

city_list = pd.read_csv("data/cities.csv")
directory_name = f"{os.getcwd()}/data/{datetime.date.today()}/"
print(os.mkdir(directory_name))
# Create the directory
# os.mkdir(directory_name)

for i in range(len(city_list)):
    open(f"data/{datetime.date.today()}/{city_list.loc[i,"Stadt"]}.json", "a").close
    extractor = WeatherDataExtractor(city_list.loc[i,"URL"])
    extractor.save_data_to_json(f"data/{datetime.date.today()}/{city_list.loc[i,"Stadt"]}.json")