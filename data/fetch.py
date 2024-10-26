import requests
from lxml import etree
import json
import pandas as pd
from bs4 import BeautifulSoup
import re

class WeatherDataExtractor:
    def __init__(self, url):
        # Initialize the WeatherDataExtractor with the provided URL
        self._url = url
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Accept-Language": "de-DE,de;q=0.9",
        }
        
        # Data structure to hold extracted data
        self._data = {
            "weather_description": None,
            "further_details": None,
            "climate_data": None,  # Placeholder for future climate data DataFrame
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
                "overview": '//*[@id="uebersicht"]',
                "diagram": '//*[@id="vhs-detail-diagram"]',
                "rainfall": '//*[@id="nowcast-table"]'
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
            # Here you can call methods to extract tabular data in the future
        else:
            print(f"Failed to fetch data: {response.status_code}")

    def _extract_strings(self, dom):
        """Extract string data based on predefined XPaths."""
        self._data["weather_description"] = self._extract_div_text(dom, self._xpaths["strings"]["weather_description"])
        self._data["further_details"] = self._extract_mix_text(dom, self._xpaths["strings"]["further_details"])
        self._data["sunrise"] = self._extract_span_text(dom, self._xpaths["strings"]["sunrise"])
        self._data["sundown"] = self._extract_span_text(dom, self._xpaths["strings"]["sundown"])
        self._data["sunhours"] = self._extract_span_text(dom, self._xpaths["strings"]["sunhours"])

    def _extract_div_text(self, dom, xpath):
        """Extract text content from a div based on a predefined XPath."""
        element = dom.xpath(xpath)
        if element:
            return element[0].text.strip()  # Return stripped text content
        return None  # Return None if element is not found

    def _extract_span_text(self, dom, xpath):
        """Helper method to extract text using direct XPath for text nodes."""
        element = dom.xpath(xpath)
        if element and len(element) > 0:
            return element[0].strip()  # Return the direct text node if it exists
        return None  # Return None if the element doesn't exist or has no text

    def _extract_mix_text(self, dom, xpath):
        """Extract and clean mixed text from an element that may contain multiple child elements."""
        element = dom.xpath(xpath)
        if element:
            # Process text content by joining all text and cleaning up
            further_details = ' '.join(element[0].itertext()).replace("\n", "").replace("\t", "").strip()
            return re.sub(r'\s+', ' ', further_details).strip()  # Replace multiple spaces with a single space
        return None

    def save_data_to_json(self, filename="data/weather_data.json"):
        """Write the extracted data to a JSON file."""
        # Convert DataFrame to a dictionary for JSON serialization if applicable
        if isinstance(self._data["climate_data"], pd.DataFrame):
            self._data["climate_data"] = self._data["climate_data"].to_dict(orient='records')

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self._data, file, ensure_ascii=False, indent=4)  # Save data in JSON format
        print(f"Data saved to {filename}")

# Usage example
url = "https://www.wetter.com/deutschland/hamburg/DE0004130.html"
extractor = WeatherDataExtractor(url)  # Create an instance of the extractor
extractor.save_data_to_json()  # Save the extracted data to "weather_data.json"
