from ..weather_extractor.extractor import WeatherDataExtractor
import pandas as pd
url = "https://www.wetter.com/deutschland/hamburg/DE0004130.html"
extractor = WeatherDataExtractor(url)
extractor.save_data_to_json() 