import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import geodatasets

df = pd.read_csv("city_data.csv", delimiter=',', skiprows=0, low_memory=False)
df = df[df['Weather Report?'] == True]

geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = GeoDataFrame(df, geometry=geometry)   

#this is a simple map that goes with geopandas
# deprecated: world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = gpd.read_file(geodatasets.data.naturalearth.land['url'])
gdf.plot(ax=world.plot(figsize=(20, 12)), marker='o', color='red', markersize=2)
import matplotlib.pyplot as plt
plt.savefig('world.jpg')