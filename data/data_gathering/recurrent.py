from data.weather_extractor.extractor import WeatherDataExtractor
from pprint import pprint
import pandas as pd
import json
import os
import time

if __name__ == "__main__":

    current_hour = int(time.strftime("%H", time.localtime()))
    rescrape_needed: bool = True if current_hour == 23 else False

    # The whole block is just about getting the latest folder in which data has been set
    dir_list = sorted(
        [dir for dir in os.listdir(f"{os.getcwd()}/data") if os.path.isdir(f"{os.getcwd()}/data/{dir}")], 
        key = lambda x: os.path.getctime(f"{os.getcwd()}/data/{x}"), 
        reverse= True
        )
    # remove directories, which are always there
    dir_list.remove("build")
    dir_list.remove("weather_extractor")
    dir_list.remove("data_gathering")
    # print(dir_list)

    # load cities dict
    df = pd.read_csv(f"{os.getcwd()}/data/cities.csv")
    # print(df)

    for file in [dir for dir in os.listdir(f"{os.getcwd()}/data/{dir_list[0]}") if "standardised" not in dir]:
        with open(f"{os.getcwd()}/data/{dir_list[0]}/{file}", "r", encoding="utf-8") as f:
            file_content = json.load(f)
            # print(file)
            if file_content["strings"]["weather_description"] == None or file_content["strings"]["further_details"] == None:
                print("missing text details: ", file)
                crawled_data = WeatherDataExtractor(df[df["Stadt"] == str(file)[:-5]]["URL"].values[0], True).return_only_data()
            else:
                continue

        if crawled_data["strings"]["weather_description"] != None or crawled_data["strings"]["further_details"] != None:
            with open(f"{os.getcwd()}/data/{dir_list[0]}/{file}", "w", encoding="utf-8") as f:  
                # print(crawled_data)             
                # if the short text has been added overwrite
                if crawled_data["strings"]["weather_description"] != None:
                    # print("weather descript found")
                    file_content["strings"]["weather_description"] = crawled_data["strings"]["weather_description"] 
                
                # if the long text has been added overwrite
                if crawled_data["strings"]["further_details"] != None:
                    # print("weather long found")
                    file_content["strings"]["further_details"] = crawled_data["strings"]["further_details"]

                # save the changes if they have happened
                json.dump(file_content, f, ensure_ascii=False, indent=4)
            
            with open(f"{os.getcwd()}/data/{dir_list[0]}/{str(file)[:-5]}_standardised.json", "r", encoding="utf-8") as f:
                file_content = json.load(f)
            with open(f"{os.getcwd()}/data/{dir_list[0]}/{str(file)[:-5]}_standardised.json", "w", encoding="utf-8") as f:
                # if the short text has been added overwrite
                if crawled_data["strings"]["weather_description"] != None:
                    file_content["report_short"] = crawled_data["strings"]["weather_description"] 
                
                # if the long text has been added overwrite
                if crawled_data["strings"]["further_details"] != None:
                    file_content["report_long"] = crawled_data["strings"]["further_details"]

                # save the changes if they have happened
                json.dump(file_content, f, ensure_ascii=False, indent=4)


    if rescrape_needed:
        files = [str(file)[:-5] for file in os.listdir(f"{os.getcwd()}/data/{dir_list[0]}") if "standardised" not in file]
        df = df[~df["Stadt"].isin(files)]
        if len(df) > 0:
            for url, city in zip(df["URL"].values, df["Stadt"].values):
                # print(url, city)
                # print(f"{os.getcwd()}/data/{dir_list[0]}/{city}.json")

                extractor = WeatherDataExtractor(str(url), False)
                extractor.save_data_to_json(f"{os.getcwd()}/data/{dir_list[0]}/{str(city)}.json")
