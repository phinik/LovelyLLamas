from typing import Dict, List
import re

# so vom stil her wie the torchvision transforms wäre glaube ich ganz cool, also für jede einzelne Operation eine
# eigene Klasse ableiten

class ReplaceNaNs:
    # Replace NaNs with a specified token
    def __init__(self, missing_token: str = '<missing>'):
        self.missing_token = missing_token

    def __call__(self, data: Dict) -> Dict:
        data = {k: v if v is not None else self.missing_token for k, v in data.items()}
        return data


class ReplaceCityName:
    # Replace the city name in the 'report_short' field with a specified token
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        if 'report_short' in data and 'city' in data:
            data['report_short'] = data['report_short'].replace(data['city'], '<city>')
        return data


class CleanOverview:
    # Clean the 'overview' field by removing optional header and replacing line breaks with spaces
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        if 'overview' in data:
            overview = data['overview']
            # Remove optional header
            overview = re.sub(r"^Time,Clearness,Temperature,Rain Chance,Rain Amount,Wind Direction,Wind Speed,Extras", "", overview)
            # Replace line breaks with commas
            overview = overview.replace('\r\n', ',')
            overview = overview.replace('\n', ',')
            # Replace further spaces
            overview = overview.replace('\u202f', '')
            overview = overview.replace('\xa0', '')
            data['overview'] = overview
        return data


class AddBewolkungsgradToOverview:
    # add cloudiness information to the 'overview' field
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        # test if 'bewölkungsgrad' and 'overview' are in the data
        if 'bewölkungsgrad' in data and 'overview' in data:
            # Split the 'overview' field into lines at <kmh> and add the respective cloudiness information
            lines = data['overview'].split('<kmh>')
            bewölkungsgrad = data['bewölkungsgrad']
            modified_lines = []
            for i, line in enumerate(lines):
                if i < len(data['bewölkungsgrad']):
                    modified_lines.append(f"{line},{data['bewölkungsgrad'][i]}")
                else:
                    modified_lines.append(line)

            # reassemble the 'overview' field
            data['overview'] = '\n'.join(modified_lines)

            # add "Bewöllkungsgrad" to the header
            lines = data['overview'].split('Wind Speed')
            data['overview'] = ',Wind Speed,Bewölkungsgrad'.join(lines)
        
        return data


class TokenizeUnits:
    # Tokenize units in the 'report_short', 'report_long', and 'overview' fields
    def __init__(self, unit_map: Dict[str, str] = None):
        self.unit_map = unit_map or {
            '°': '<degC>',
            'l/m²': '<l_per_sqm>',
            'km/h': '<kmh>',
            '%': '<percent>'
        }

    def __call__(self, data: Dict) -> Dict:
        for key in ['report_short', 'report_long', 'overview']:
            if key in data:
                for unit, token in self.unit_map.items():
                    data[key] = data[key].replace(unit, token)

        # Tokenize units in lists (temperatures, rain chance, etc.)
        if 'temperatur_in_deg_C' in data:
            data['temperatur_in_deg_C'] = [f"{temp}{self.unit_map['°']}" for temp in data['temperatur_in_deg_C']]
        if 'niederschlagsrisiko_in_perc' in data:
            data['niederschlagsrisiko_in_perc'] = [f"{risk}{self.unit_map['%']}" for risk in data['niederschlagsrisiko_in_perc']]
        if 'niederschlagsmenge_in_l_per_sqm' in data:
            data['niederschlagsmenge_in_l_per_sqm'] = [f"{amount}{self.unit_map['l/m²']}" if amount != 0 else "0" for amount in data['niederschlagsmenge_in_l_per_sqm']]
        if 'windgeschwindigkeit_in_km_per_s' in data:
            data['windgeschwindigkeit_in_km_per_s'] = [f"{speed}{self.unit_map['km/h']}" for speed in data['windgeschwindigkeit_in_km_per_s']]

        return data


class PreprocessorPipeline:
    # Define a preprocessing pipeline
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, data: Dict) -> Dict:
        for transform in self.transforms:
            data = transform(data)
        return data
    
class ReduceKeys:
    # Reduce the set of keys in the data dictionary
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        reduced_set_of_keys = ["city", "report_short", "overview"]

        reduced_dict = {}
        for key in reduced_set_of_keys:
            reduced_dict[key] = data[key]

        return reduced_dict

# Example data
data = {
    "city": "Hamburg",
    "created_day": "2024-10-26",
    "created_time": "21:00:33",
    "report_short": "In Hamburg ist es morgens grau und es bleibt neblig und die Temperatur liegt bei 7°C. Am Mittag wechseln sich Wolken und Sonne ab bei Höchstwerten von 16°C. Abends ist es in Hamburg locker bewölkt und die Temperaturen liegen zwischen 12 und 14 Grad. Nachts gibt es nur selten Lücken in der Wolkendecke bei Tiefstwerten von 11°C.  Böen können Geschwindigkeiten zwischen 19 und 22 km/h erreichen.",
    "report_long": "Wetter heute, 26.10.2024 In Hamburg wird am Morgen die Sicht durch Nebel eingeschränkt und die Temperatur liegt bei 7°C. Mittags zeigt sich die Sonne nur vereinzelt bei sonst wolkigem Himmel und die Temperatur steigt auf 16°C. Am Abend bilden sich in Hamburg vereinzelt Wolken bei Temperaturen von 12 bis 14°C. Nachts ist ein Blick auf die Sterne nur vereinzelt bei sonst wolkigem Himmel möglich bei Tiefstwerten von 11°C. Mit Böen zwischen 19 und 22 km/h ist zu rechnen. Gefühlt liegen die Temperaturen bei 7 bis 17°C. Hamburg liegt in den Regionen Lüneburger Heide und Nordsee . Öffnen Sie eine Region um eine Wettervorhersage für die gesamte Region zu erhalten.",
    "sunrise": "08:08 Uhr",
    "sundown": "17:58 Uhr",
    "sunhours": "2 h",

    # aus "overview"
    "times": ["20-21", "21-22", "22-23", "23-00", "01-02", "02-03", "03-04", "04-05", "05-06", "06-07", "07-08", "08-09", "09-10", "10-11", "11-12", "12-13", "13-14", "14-15", "15-16", "16-17", "17-18", "18-19", "19-20"],
    "clearness": ["Wolkig", "Leicht bewölkt", "Leicht bewölkt", "Leicht bewölkt", "Wolkig", "Wolkig", "Wolkig", "Wolkig", "Bedeckt", "Bedeckt", "Wolkig", "Bedeckt", "Bedeckt", "Leichter Regen", "Wolkig", "Wolkig", "Wolkig", "Wolkig, und windig", "Wolkig", "Wolkig", "Wolkig", "Leicht bewölkt", "Klar"],
    
    # einfach den "overview" übernehmen
    "overview": "Time,Clearness,Temperature,Rain Chance,Rain Amount,Wind Direction,Wind Speed\r\n20 - 21 Uhr,Wolkig,13°,5 %,0 l/m², SO,8 km/h\r\n21 - 22 Uhr,Leicht bewölkt,12°,25 %,0 l/m², SO,8 km/h\r\n22 - 23 Uhr,Leicht bewölkt,12°,5 %,0 l/m², SO,8 km/h\r\n23 - 00 Uhr,Leicht bewölkt,12°,0 %,0 l/m², SO,8 km/h\r\n\"Sonntag, 27.10.\",01 - 02 Uhr,Wolkig,11°,0 %,0 l/m², SO\r\n7 km/h,02 - 03 Uhr,Wolkig,11°,0 %,0 l/m², SO\r\n5 km/h,03 - 04 Uhr,Wolkig,11°,0 %,0 l/m², S\r\n4 km/h,04 - 05 Uhr,Wolkig,11°,0 %,0 l/m², S\r\n5 km/h,05 - 06 Uhr,Bedeckt,11°,0 %,0 l/m², SW\r\n6 km/h,06 - 07 Uhr,Bedeckt,11°,0 %,0 l/m², SW\r\n6 km/h,07 - 08 Uhr,Wolkig,11°,0 %,0 l/m², SW\r\n8 km/h,08 - 09 Uhr,Bedeckt,12°,15 %,0 l/m², SW\r\n8 km/h,09 - 10 Uhr,Bedeckt,12°,45 %,0 l/m², W\r\n8 km/h,10 - 11 Uhr,Leichter Regen,13°,90 %,\"0,3 l/m²\", W\r\n12 km/h,11 - 12 Uhr,Wolkig,14°,45 %,0 l/m², W\r\n12 km/h,12 - 13 Uhr,Wolkig,15°,5 %,0 l/m², W\r\n15 km/h,13 - 14 Uhr,Wolkig,15°,5 %,0 l/m², W\r\n15 km/h,14 - 15 Uhr,Wolkig,und windig,15°,20 %,0 l/m²\r\n W,15 km/h,Böen 40 km/h,15 - 16 Uhr,Wolkig,14°,25 %\r\n0 l/m², W,12 km/h,16 - 17 Uhr,Wolkig,13°,20 %\r\n0 l/m², NW,10 km/h,17 - 18 Uhr,Wolkig,12°,15 %\r\n0 l/m², NW,9 km/h,18 - 19 Uhr,Leicht bewölkt,11°,25 %\r\n0 l/m², NW,4 km/h,19 - 20 Uhr,Klar,10°,5 %\r\n0 l/m², W,3 km/h,,,,\r\n",
    
    # aus "diagram"
    "temperatur_in_deg_C": [13,12,12,12,11,11,11,11,11,11,11,12,12,13,14,15,15,15,14,13,12,11,10],
    "niederschlagsrisiko_in_perc": [5,25,5,0,0,0,0,0,0,0,0,15,45,90,45,5,5,20,25,20,15,25,5],
    "niederschlagsmenge_in_l_per_sqm": [0,0,0,0,0,0,0,0,0,0,0,0,0,0.34,0,0,0,0,0,0,0,0,0],
    "windrichtung": ["SO","SO","SO","SO","SO","SO","S","S","SW","SW","SW","SW","W","W","W","W","W","W","W","NW","NW","NW","W"],
    "windgeschwindigkeit_in_km_per_s": [8,8,8,8,7,5,4,5,6,6,8,8,8,12,12,15,15,15,12,10,9,4,3],
    "luftdruck_in_hpa": [1018,1017,1017,1016,1015,1015,1016,1015,1015,1015,1016,1017,1018,1018,1019,1019,1020,1020,1021,1022,1023,1023,1024],
    "relative_feuchte_in_perc": [95,95,96,96,95,95,93,92,90,89,89,87,89,91,83,67,66,69,72,78,78,79,84],
    "bewölkungsgrad": ["4/8","4/8","4/8","4/8","3/8","7/8","6/8","7/8","8/8","8/8","5/8","8/8","8/8","7/8","5/8","6/8","6/8","7/8","7/8","6/8","6/8","3/8","1/8"]
}

# Define the preprocessing pipeline
pipeline = PreprocessorPipeline([
    ReplaceNaNs(),
    ReplaceCityName(),
    TokenizeUnits(),
    AddBewolkungsgradToOverview(),
    CleanOverview(),
    ReduceKeys()
])

# Apply preprocessing
processed_data = pipeline(data)

print(data)
print(processed_data)