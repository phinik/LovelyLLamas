import argparse
import csv
import json
import matplotlib.pyplot as plt
import os


from mpl_toolkits.basemap import Basemap
from typing import Set, Dict


def load_cities(dataset: str) -> Set[str]:
    cities = []

    dirs = ["train", "eval", "test"]
    for dir in dirs:
        dir_path = os.path.join(dataset, dir)
        
        files = os.listdir(dir_path)
        for file in files:
            with open(os.path.join(dir_path, file), "r") as f:
                file_content = json.load(f)
            
            city = file_content["city"]
            
            cities.append((file, city))

    return set(cities)


def load_city_data() -> Dict:
    data = {}
    with open("./data//misc/crawled_information/city_data.csv", "r") as f:
        reader = csv.DictReader(f)
        
        for item in reader:
            data[item["True Name"].strip()] = item

    return data


def filter_city_data_based_on_cities(cities: Set, city_data: Dict) -> Dict:
    data = {}
    for file, city in cities:
        data[city] = {
            "lat": float(city_data[city]["Latitude"].strip()),
            "lon": float(city_data[city]["Longitude"].strip())
            }
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    cities = load_cities(args.dataset)
    city_data = load_city_data()

    city_lat_lon = filter_city_data_based_on_cities(cities, city_data)

    m = Basemap(projection='kav7',lon_0=0,resolution='c')
    m.drawcoastlines()
    m.fillcontinents(color='#C0C0C0',lake_color='#A4DCEE')
    m.drawmapboundary(fill_color='#A4DCEE') 

    lons = []
    lats = []
    for city, coords in city_lat_lon.items():
        lats.append(coords["lat"])
        lons.append(coords["lon"])

    x, y = m(lons,lats)
    m.scatter(x,y,3,marker='o',color='#B00046')
    plt.show()