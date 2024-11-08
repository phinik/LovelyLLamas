from typing import Dict, List

# so vom stil her wie the torchvision transforms wäre glaube ich ganz cool, also für jede einzelne Operation eine
# eigene Klasse ableiten

class ReplaceNaNs:
    # Replace NaNs with a specified token
    def __init__(self, missing_token: str = '<missing>'):
        self.missing_token = missing_token

    def __call__(self, data: Dict) -> Dict:
        data = {k: v if v is not None else self.missing_token for k, v in data.items()}
        return data


class TokenizeUnits:
    # Tokenize units in the 'report_short', 'report_long', and 'overview' fields
    def __init__(self, unit_map: Dict[str, str] = None):
        self.unit_map = unit_map or {
            '°C': '<degC>',
            '°': '<degC>',
            'l/m²': '<l_per_sqm>',
            'km/h': '<kmh>',
            '%': '<percent>'
        }

    def __call__(self, data: Dict) -> Dict:
        for key in ['report_short', 'report_long']:
            if key in data:
                for unit, token in self.unit_map.items():
                    data[key] = data[key].replace(unit, token)

        return data


class ReplaceCityName:
    # Replace the city name in the 'report_short' field with a specified token
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        if 'report_short' in data and 'city' in data:
            data['report_short'] = data['report_short'].replace(data['city'], '<city>')
        return data
    
class ReduceKeys:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        reduced_set_of_keys = ["city", "report_short", "overview"]

        reduced_dict = {}
        for key in reduced_set_of_keys:
            reduced_dict[key] = data[key]

        return reduced_dict
    
class AssembleCustomOverview:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        s = ""

        for time, clearness, temp, rain_risk, rain_amount, wind_direction, wind_speed, cloudiness in zip(
            data["times"], 
            data["clearness"], 
            data["temperatur_in_deg_C"], 
            data["niederschlagsrisiko_in_perc"],
            data["niederschlagsmenge_in_l_per_sqm"], 
            data["windrichtung"], 
            data["windgeschwindigkeit_in_km_per_s"],
            data["bewölkungsgrad"]
            ):\
            
            if s != "":
                s+= ","

            clearness = clearness.replace(",", "")
            
            s += f"{time},{clearness},{temp},{rain_risk},{rain_amount},{wind_direction},{wind_speed},{cloudiness}"
       
        data["overview"] = s
         
        return data
    
class ToTensor:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        pass

class PreprocessorPipeline:
    # Define a preprocessing pipeline
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, data: Dict) -> Dict:
        for transform in self.transforms:
            data = transform(data)
        return data

# Define the preprocessing pipeline
pipeline = PreprocessorPipeline([
    ReplaceNaNs(),
    TokenizeUnits(),
    ReplaceCityName(),
    AssembleCustomOverview(),
    ReduceKeys()
])