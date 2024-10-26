# weather_extractor.pyx
# cython: language_level=3
import cython
from cpython.ref cimport PyObject
import requests
from lxml import etree
import json
import pandas as pd
import re
from io import StringIO
from datetime import datetime
from libc.string cimport strlen
from cpython cimport array
import numpy as np

cdef class WeatherDataExtractor:
    cdef str _url
    cdef dict _headers
    cdef dict _data
    cdef dict _xpaths

    def __init__(self, str url):
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
            "created_on": str(datetime.now())
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _fetch_and_extract_data(self):
        """Fetch the HTML content and extract the required data."""
        response = requests.get(self._url, headers=self._headers)
        if response.status_code == 200:
            dom = etree.HTML(response.content)
            self._extract_strings(dom)
            self._extract_tables(dom)
        else:
            print(f"Failed to fetch data: {response.status_code}")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _extract_strings(self, object dom):
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _extract_tables(self, object dom):
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

    @cython.boundscheck(False)
    cdef str _extract_div_text(self, list element):
        """Extract text content from a div based on a predefined XPath."""
        if element:
            return element[0].text.strip()
        return ""

    @cython.boundscheck(False)
    cdef str _extract_span_text(self, list element):
        """Helper method to extract text using direct XPath for text nodes."""
        if len(element) > 0:
            return element[0].strip()
        return ""

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef str _extract_mix_text(self, list element):
        """Extract and clean mixed text from an element that may contain multiple child elements."""
        cdef str further_details = ' '.join(element[0].itertext()).replace("\n", "").replace("\t", "").strip()
        return re.sub(r'\s+', ' ', further_details).strip()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef object _parse_rainfall_data(self, list element):
        """Convert the rainfall data string into a DataFrame."""
        cdef str rainfall_text = etree.tostring(element[0], method="text", encoding="unicode").replace("\n", "").replace("\t", "").replace("Weitere Werte anzeigen", "").strip()
        
        # Use regex to find time and description pairs
        pattern = r'(\d{2}:\d{2})\s+([^0-9]*)'
        matches = re.findall(pattern, rainfall_text)

        return pd.DataFrame(
            [[time.strip(), description.strip()] for time, description in matches],
            columns=["Time", "Description"]
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef object _parse_overview_data(self, list element):
        """Parse overview data from the provided element into a DataFrame."""
        cdef str overview_text = etree.tostring(element[0], method="text", encoding="unicode").replace("\n", "").replace("\t", "").strip()
        cdef list overview_list = [elem for elem in re.sub(r' {2,}', '~', overview_text).strip().split("~") if len(elem) < 16]
        return pd.DataFrame(
            self._split_list(overview_list, 7),
            columns=["Time", "Clearness", "Temperature", "Rain Chance", "Rain Amount", "Wind Direction", "Wind Speed"]
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef object _parse_sub_area_data(self, list element):
        """Parse sub-area weather data from the provided element into a DataFrame."""
        cdef str sub_area_text = etree.tostring(element[0], method="text", encoding="unicode").replace("\n", "").replace("\t", "").strip()
        return pd.DataFrame(
            self._split_list(re.sub(r' {2,}', '~', sub_area_text).strip().split("~")[1:], 2),
            columns=["Region", "Temperature"]
        )

    @cython.boundscheck(False)
    cdef object _parse_table_data(self, list element):
        """Parse a generic HTML table into a DataFrame."""
        cdef str table_html = etree.tostring(element[0], encoding='unicode')
        cdef object html_io = StringIO(table_html)
        return pd.read_html(html_io)[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _split_list(self, list original_list, int n):
        """Split a list into chunks of size n."""
        return [original_list[i:i + n] for i in range(0, len(original_list), n)]

    def save_data_to_json(self, str filename="data/weather_data.json"):
        """Write the extracted data to a JSON file."""
        # Convert DataFrame to a dictionary for JSON serialization if applicable
        cdef dict data_copy = self._data.copy()
        for key in data_copy["tabular"].keys():
            if isinstance(data_copy["tabular"][key], pd.DataFrame):
                data_copy["tabular"][key] = data_copy["tabular"][key].to_csv(index=False)

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data_copy, file, ensure_ascii=False, indent=4)
        print(f"Data saved to {filename}")