import requests
from lxml import etree
import json
import pandas as pd
import re
from io import StringIO
import datetime

class WeatherDataExtractor:
    def __init__(self, url: str, timezone: bool = True):
        """
        Initialize the WeatherDataExtractor with the provided URL.
        Timezone parameter used to see if timezone specific data should be considered.
        """
        self._url = url
        self._timezone = timezone
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Accept-Language": "de-DE,de;q=0.9",
        }
        
        # Data structure to hold extracted data
        self._data = {
            "strings": {
                "weather_description": None,
                "further_details": None,
                "sunrise": None,
                "sundown": None,
                "sunhours": None
            },
            "tabular": {
                "weather_today": None,
                "overview": None,
                "diagram": None,
                "rainfall": None,
                "climate_table": None,
                "sub_areas_weather": None
            },
            "created_on": str(datetime.datetime.now())
        }

        # Predefined XPaths for data extraction
        self._xpaths = {
            "strings": {
                "weather_description": '//*[@id="cnt-with-location-attributes"]/div/p',
                "further_details": '//*[@id="furtherDetails"]/div/div[1]',
                "sunrise": '''//span[@title='Sonnenaufgang']/following-sibling::text()[1]''',
                "sundown": '''//span[@title='Sonnenuntergang']/following-sibling::text()[1]''',
                "sunhours": '''//span[contains(@class, 'icon-sun_hours')]/following-sibling::span/text()'''
            },
            "tabular": {
                "weather_today": '//*[@id="cnt-with-location-attributes"]/div/table',
                "overview": '//div[@id="uebersicht"]',
                "diagram": '//*[@id="vhs-detail-diagram"]',
                "rainfall": '//*[@id="nowcast-table"]',
                "climate_table": '//*[@id="furtherDetails"]/div/div[4]/div[2]/table',
                "sub_areas_weather": '//*[@id="furtherDetails"]/div/div[2]/div'
            }
        }
        
        # Fetch data from the URL and extract relevant information
        self._fetch_and_extract_data()

    def _fetch_and_extract_data(self):
        """Fetch the HTML content and extract the required data."""
        response = requests.get(self._url, headers=self._headers)
        if response.status_code == 200:
            dom = etree.HTML(response.content)
            self._extract_strings(dom)
            self._extract_tables(dom)
        else:
            print(f"Failed to fetch data: {response.status_code}")

    def _extract_strings(self, dom):
        """Extract string data based on predefined XPaths."""
        for key in self._xpaths["strings"]:
            element = dom.xpath(self._xpaths["strings"][key])
            if element:
                if key == "weather_description":
                    self._data["strings"][key] = self._extract_div_text(element)
                elif key == "further_details":
                    self._data["strings"][key] = self._extract_mix_text(element)
                else:
                    self._data["strings"][key] = self._extract_span_text(element)

    def _extract_tables(self, dom):
        """Extract table data based on predefined XPaths."""
        for key in self._xpaths["tabular"]:
            element = dom.xpath(self._xpaths["tabular"][key])
            if element:
                if key == "rainfall":
                    self._data["tabular"][key] = self._parse_rainfall_data(element)
                elif key == "overview":
                    self._data["tabular"][key] = self._parse_overview_data(element)
                    # Exit Condition based on TimeZone
                    try:
                        if self._data["tabular"][key] == None:
                            break
                    except Exception:
                        pass
                elif key == "sub_areas_weather":
                    self._data["tabular"][key] = self._parse_sub_area_data(element)
                else:
                    self._data["tabular"][key] = self._parse_table_data(element)

    def _extract_div_text(self, element):
        """Extract text content from a div based on a predefined XPath."""
        if element:
            return element[0].text.strip()  # Return stripped text content
        return None  # Return None if element is not found

    def _extract_span_text(self, element):
        """Helper method to extract text using direct XPath for text nodes."""
        if len(element) > 0:
            return element[0].strip()  # Return the direct text node if it exists
        return None  # Return None if the element doesn't exist or has no text

    def _extract_mix_text(self, element):
        """Extract and clean mixed text from an element that may contain multiple child elements."""
        # Process text content by joining all text and cleaning up
        further_details = ' '.join(element[0].itertext()).replace("\n", "").replace("\t", "").strip()
        return re.sub(r'\s+', ' ', further_details).strip()  # Replace multiple spaces with a single space
    
    def _parse_rainfall_data(self, element):
        """Convert the rainfall data string into a list of lists."""
        rainfall_text = etree.tostring(element[0], method="text", encoding="unicode").replace("\n", "").replace("\t", "").replace("Weitere Werte anzeigen", "").strip()
        
        # Use regex to find time and description pairs
        pattern = r'(\d{2}:\d{2})\s+([^0-9]*)'
        matches = re.findall(pattern, rainfall_text)

        return pd.DataFrame([[time.strip(), description.strip()] for time, description in matches], columns=["Time", "Description"])
    
    def _parse_overview_data(self, element):
        """Parse overview data from the provided element into a DataFrame."""
        # Initialize an empty list to hold the text from the filtered elements
        overview_text_list = []
        
        # Compile regex patterns once for better performance
        time_pattern = re.compile(r'\d{2} - \d{2} Uhr')
        clearness_pattern = re.compile(r'r [\w+ ]+ ')
        temp_pattern = re.compile(r'\d+°')
        rain_chance_pattern = re.compile(r'\d+ %')
        rain_amount_pattern = re.compile(r'\d+ l/m²')
        wind_dir_pattern = re.compile(r'm² \w+')
        wind_speed_pattern = re.compile(r'\d+ km/h')
        extras_pattern = re.compile(r'h[\w+ ]+$')
        
        # Iterate through each item in the element list
        for elem in element:
            # Ensure each item is processed if it's an `etree.Element`
            if isinstance(elem, etree._Element):
                # Filter to keep only elements with the desired class within each element
                filtered_elements = elem.xpath(".//*[contains(@class, 'swg-row-wrapper border--grey-next')]")
                
                # Convert each filtered element to text, clean it up, and add to list
                for filtered_elem in filtered_elements:
                    overview_text1 = re.sub(r'\s+', ' ', etree.tostring(filtered_elem, method="text", encoding="unicode").replace("\n", "").replace("\t", "").strip())
                    
                    # Extract data with null checks for each pattern
                    time_match = time_pattern.match(overview_text1)
                    clearness_matches = clearness_pattern.findall(overview_text1)
                    temp_matches = temp_pattern.findall(overview_text1)
                    rain_chance_matches = rain_chance_pattern.findall(overview_text1)
                    rain_amount_matches = rain_amount_pattern.findall(overview_text1)
                    wind_dir_matches = wind_dir_pattern.findall(overview_text1)
                    wind_speed_matches = wind_speed_pattern.findall(overview_text1)
                    extras_matches = extras_pattern.findall(overview_text1)
                    
                    # Append data with null checks
                    overview_text_list.append(
                        [
                            time_match.group(0) if time_match else "",
                            clearness_matches[0][2:-1] if clearness_matches else "",
                            temp_matches[0] if temp_matches else "",
                            rain_chance_matches[0] if rain_chance_matches else "",
                            rain_amount_matches[0] if rain_amount_matches else "",
                            wind_dir_matches[0][3:] if wind_dir_matches else "",
                            wind_speed_matches[0] if wind_speed_matches else "",
                            extras_matches[0][2:] if extras_matches else ""
                        ]
                    )
        # Join all filtered text parts together into a single string (if needed)
        overview_df = pd.DataFrame(
            overview_text_list, 
            columns=["Time", "Clearness", "Temperature", "Rain Chance", "Rain Amount", "Wind Direction", "Wind Speed", "Extras"]
        )
        # If TimeZone is not correct, stop the process
        if overview_df["Time"].loc[0] != "00 - 01 Uhr" and self._timezone:
            return None
        else:
            return overview_df

    def _parse_sub_area_data(self, element):
        """Parse sub-area weather data from the provided element into a DataFrame."""
        sub_area_text: str = etree.tostring(element[0], method="text", encoding="unicode").replace("\n", "").replace("\t", "").strip()
        return pd.DataFrame(
            self._split_list(re.sub(r' {2,}', '~', sub_area_text).strip().split("~")[1:], 2),
            columns=["Region", "Temperature"]
        )

    def _parse_table_data(self, element):
        """Parse a generic HTML table into a DataFrame."""
        # Convert the found element to a string
        table_html = etree.tostring(element[0], encoding='unicode')

        # Use StringIO to wrap the HTML string for pandas
        html_io = StringIO(table_html)

        # Use pandas to read the HTML table and convert it to a DataFrame
        df = pd.read_html(html_io)[0]  # Assuming you want the first table if there are multiple
        return df

    def _split_list(self, original_list, n):
        """Split a list into chunks of size n."""
        return [original_list[i:i + n] for i in range(0, len(original_list), n)]
    
    def return_only_data(self):
        """Used when double checking if the weather text has been added"""
        return self._data

    def save_data_to_json(self, filename="data/weather_data.json"):
        """Write the extracted data to a JSON file."""
        # TimeZone Exit Condition
        if self._data["tabular"]["overview"] is None:
            return
        # Create a standardized Dict for the AI input
        standardized_dict = {
            "city": filename.split("/")[-1][:-5],
            "created_day": str(datetime.date.today()),
            "created_time": datetime.datetime.now().strftime("%H:%M:%S"),
            "report_short": self._data["strings"]["weather_description"],
            "report_long": self._data["strings"]["further_details"],
            "sunrise": self._data["strings"]["sunrise"],
            "sundown": self._data["strings"]["sundown"],
            "sunhours": self._data["strings"]["sunhours"],
            "times": self._data["tabular"]["overview"]["Time"].to_list(),
            "clearness": self._data["tabular"]["overview"]["Clearness"].to_list(),
            "overview": self._data["tabular"]["overview"].to_csv(index=False),
            "temperatur_in_deg_C": self._data["tabular"]["diagram"].iloc[6].to_list(), 
            "niederschlagsrisiko_in_perc": self._data["tabular"]["diagram"].iloc[8].to_list(),
            "niederschlagsmenge_in_l_per_sqm": self._data["tabular"]["diagram"].iloc[10].to_list(),
            "windrichtung": self._data["tabular"]["diagram"].iloc[12].to_list(),
            "windgeschwindigkeit_in_km_per_s": self._data["tabular"]["diagram"].iloc[13].to_list(),
            "luftdruck_in_hpa": self._data["tabular"]["diagram"].iloc[15].to_list(),
            "relative_feuchte_in_perc": self._data["tabular"]["diagram"].iloc[17].to_list(),
            "bewölkungsgrad": self._data["tabular"]["diagram"].iloc[19].to_list()
        }
        
        # Convert DataFrame to a dictionary for JSON serialization if applicable
        for key in self._data["tabular"].keys():
            if isinstance(self._data["tabular"][key], pd.DataFrame):
                self._data["tabular"][key] = self._data["tabular"][key].to_csv(index=False)

        # Save Standardized and self._data
        with open(f"{filename[:-5]}_standardised.json", "w", encoding="utf-8") as file:
            json.dump(standardized_dict, file, ensure_ascii=False, indent=4)
        print(f"Organised Data saved to {filename[:-5]}_standardised.json")

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self._data, file, ensure_ascii=False, indent=4)  # Save data in JSON format
        print(f"Data saved to {filename}")


# Usage example
if __name__ == "__main__":
    url = "https://www.wetter.com/schweden/stockholm/SE0ST0012.html"
    extractor = WeatherDataExtractor(url)  # Create an instance of the extractor
    extractor.save_data_to_json("Abuja.json")  # Save the extracted data to "weather_data.json"
