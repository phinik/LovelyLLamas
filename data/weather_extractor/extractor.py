import requests
from lxml import etree
import json
import pandas as pd
import re
from io import StringIO
import datetime

class WeatherDataExtractor:
    def __init__(self, url):
        """Initialize the WeatherDataExtractor with the provided URL."""
        self._url = url
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
        overview_text: str = etree.tostring(element[0], method="text", encoding="unicode").replace("\n", "").replace("\t", "").strip()
        overview_list: list = [elem for elem in re.sub(r' {2,}', '~', overview_text).strip().split("~") if len(elem) < 16]
        return pd.DataFrame(
            self._split_list(overview_list, 7), 
            columns=["Time", "Clearness", "Temperature", "Rain Chance", "Rain Amount", "Wind Direction", "Wind Speed"]
        )

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
    
    def save_data_to_json(self, filename="data/weather_data.json"):
        """Write the extracted data to a JSON file."""
        # Convert DataFrame to a dictionary for JSON serialization if applicable
        for key in self._data["tabular"].keys():
            if isinstance(self._data["tabular"][key], pd.DataFrame):
                self._data["tabular"][key] = self._data["tabular"][key].to_csv(index=False)

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self._data, file, ensure_ascii=False, indent=4)  # Save data in JSON format
        print(f"Data saved to {filename}")

# Usage example
if __name__ == "__main__":
    url = "https://www.wetter.com/deutschland/hamburg/DE0004130.html"
    extractor = WeatherDataExtractor(url)  # Create an instance of the extractor
    extractor.save_data_to_json()  # Save the extracted data to "weather_data.json"
