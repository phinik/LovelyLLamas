import pandas as pd
import os


# USED FOR FINDING TIME DIFFERENCES
# df = pd.read_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_with_true_names_and_times.csv")
# df["Scraped Time"] = df["Time at Scrape"].apply(lambda x: int((str(x).split("|")[0]).split("-")[0]))
# df["Actual Time"] = df["Time at Scrape"].apply(lambda x: int(str(x).split("|")[1]))
# # Calculate Time Difference (row-wise)
# df["Time Difference"] = df.apply(
#     lambda x: x["Scraped Time"] - x["Actual Time"] if x["Scraped Time"] >= x["Actual Time"] else x["Scraped Time"] + 24 - x["Actual Time"],
#     axis=1
# )
# df = df.sort_values(by=["Time Difference"])
# print(df["Scraped Time"].unique())
# print(df["Actual Time"].unique())
# print(df["Time Difference"].value_counts())
# df.to_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_with_true_names_and_times.csv")


# USED FOR CHECKING IF URLs ARE DOUBLED
# # Load cities and regions data
# cities = pd.read_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_with_true_names_and_times.csv")
# regions = pd.read_csv(f"{os.getcwd()}/data/misc/crawled_information/regions.csv")

# # Remove rows from cities where URL is in regions
# filtered_cities = cities[~cities["URL"].isin(regions["URL"])]

# # Save the filtered dataset back to a file
# filtered_cities.to_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_filtered.csv", index=False)

# # Output the count of removed rows for verification
# removed_count = len(cities) - len(filtered_cities)
# print(f"Rows removed: {removed_count}")

# MERGE TWO DIFFERENT DATASETS FOR DATA REDUCTION
# Load datasets
city_data = pd.read_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data.csv")
city_data_tz = pd.read_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_tz.csv")

# Merge on 'URL', keeping the structure of city_data and adding the necessary columns
merged_data = city_data.merge(
    city_data_tz[["URL", "Latitude", "Longitude", "Timezone"]],
    on="URL",
    how="left"  # Use 'left' join to keep all rows from city_data
)

# Save the merged dataset
merged_data.to_csv(f"{os.getcwd()}/data/misc/crawled_information/city_data_with_tz.csv", index=False)

# Output some information about the merge
print(f"Merged data has {len(merged_data)} rows and {merged_data.shape[1]} columns.")
