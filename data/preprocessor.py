import re
import pandas as pd

class Preprocessor:
    def __init__(self, data: dict):
        self.data = data
        self.extra_tokens = {
            '<stop>': None,
            '<city>': self.data.get('city', '<missing>'),
            '<missing>': 'NaN',
            '<degC>': '°',
            '<l_per_sqm>': 'l/m²',
            '<kmh>': 'km/h',
            '<percent>': '%'
        }
    
    def replace_nans(self):
        # replace NaNs with <missing>
        self.data = {k: v if v is not None else '<missing>' for k, v in self.data.items()}
    
    def replace_city_name(self):
        # replace name of city in ‘report_short’ with <city>
        if 'report_short' in self.data:
            self.data['report_short'] = self.data['report_short'].replace(self.data['city'], '<city>')
    
    def clean_overview(self):
        # clean ‘overview’: turn \n into spaces and remove optional heaer
        if 'overview' in self.data:
            overview = self.data['overview']
            # remove header, if it exists
            overview = re.sub(r"^Time,Clearness,Temperature,Rain Chance,Rain Amount,Wind Direction,Wind Speed,Extras", "", overview)
            # turn \n into spaces
            overview = overview.replace('\n', ' ')
            # tokenize units
            overview = self.tokenize_units(overview)
            self.data['overview'] = overview

    def add_bewolkungsgrad_to_overview(self):
        # add cloudiness to ‘overview’
        if 'bewölkungsgrad' in self.data:
            cloudiness = ' '.join(self.data['bewölkungsgrad'])
            self.data['overview'] += f" Bewölkungsgrad: {cloudiness}"
    
    def tokenize_units(self, text):
        # tokenize units in text
        unit_map = {
            '°': '<degC>',
            'l/m²': '<l_per_sqm>',
            'km/h': '<kmh>',
            '%': '<percent>'
        }
        for unit, token in unit_map.items():
            text = text.replace(unit, token)
        return text
    
    def tokenize_all_units(self):
        # apply tokenization for units across all relevant fields
        fields_to_tokenize = ['report_short', 'report_long', 'overview']
        for field in fields_to_tokenize:
            if field in self.data:
                self.data[field] = self.tokenize_units(self.data[field])
        # tokenize lists with units (e.g., temperatures, rain chance, etc.)
        if 'temperatur_in_deg_C' in self.data:
            self.data['temperatur_in_deg_C'] = [f"{temp}<degC>" for temp in self.data['temperatur_in_deg_C']]
        if 'niederschlagsrisiko_in_perc' in self.data:
            self.data['niederschlagsrisiko_in_perc'] = [f"{risk}<percent>" for risk in self.data['niederschlagsrisiko_in_perc']]
        if 'niederschlagsmenge_in_l_per_sqm' in self.data:
            self.data['niederschlagsmenge_in_l_per_sqm'] = [f"{amount}<l_per_sqm>" if amount != 0 else "0" for amount in self.data['niederschlagsmenge_in_l_per_sqm']]
        if 'windgeschwindigkeit_in_km_per_s' in self.data:
            self.data['windgeschwindigkeit_in_km_per_s'] = [f"{speed}<kmh>" for speed in self.data['windgeschwindigkeit_in_km_per_s']]
    
    def apply_preprocessing(self):
        # apply preprocessing steps
        self.replace_nans()
        self.replace_city_name()
        self.clean_overview()
        self.add_bewolkungsgrad_to_overview()
        self.tokenize_all_units()
        return self.data

# example data

data = {
    "city": "Hamburg",
    "created_day": "2024-10-26",
    "created_time": "21:00:33",
    "report_short": "In Hamburg ist es morgens grau...",
    "overview": "Time,Clearness,Temperature,Rain Chance,Rain Amount,Wind Direction,Wind Speed\n20 - 21 Uhr,Wolkig,13°,5 %,0 l/m², SO,8 km/h\n",
    "temperatur_in_deg_C": [13, 12, 12, 12, 11],
    "niederschlagsrisiko_in_perc": [5, 25, 5, 0, 0],
    "niederschlagsmenge_in_l_per_sqm": [0, 0, 0, 0, 0],
    "windgeschwindigkeit_in_km_per_s": [8, 8, 8, 8, 7],
}

# Instanziiere Preprocessor und wende das Preprocessing an
preprocessor = Preprocessor(data)
processed_data = preprocessor.apply_preprocessing()

print(data)
print(processed_data)