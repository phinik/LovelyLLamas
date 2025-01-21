import numpy as np

from typing import Dict, List


class ReplaceNaNs:
    # Replace NaNs with a specified token
    def __init__(self, missing_token: str = '<missing>'):
        self._missing_token = missing_token

    def __call__(self, data: Dict) -> Dict:
        keys = ["times", "clearness", "temperatur_in_deg_C", "niederschlagsrisiko_in_perc", 
                "niederschlagsmenge_in_l_per_sqm", "windrichtung", "windgeschwindigkeit_in_km_per_h", "bewölkungsgrad"]
        
        for key in keys:
            old_values = data[key]
            new_values = []
            for v in old_values:
                try:
                    if np.isnan(v):
                        new_values.append(self._missing_token)
                    else:
                        new_values.append(v)
                except TypeError:
                    new_values.append(v)
            data[key] = new_values

        #data = {k: v if v is not None else self.missing_token for k, v in data.items()}

        return data


class TokenizeUnits:
    # Tokenize units in the 'report_short', 'report_long', and 'overview' fields
    def __init__(self, unit_map: Dict[str, str] = None):
        self.unit_map = unit_map or {
            '°C': ' <degC>',
            #'°': ' <degC>',
            'l/m²': ' <l_per_sqm>',
            'km/h': ' <kmh>',
            '%': ' <percent>'
        }

    def __call__(self, data: Dict) -> Dict:
        for key in ['report_short', 'report_short_wout_boeen']:
            for unit, token in self.unit_map.items():
                data[key] = data[key].replace(unit, token)
        
        if "gpt_rewritten_cleaned" in data:
            for unit, token in self.unit_map.items():
                data["gpt_rewritten_cleaned"] = data["gpt_rewritten_cleaned"].replace(unit, token)

        return data


class ReplaceCityName:
    # Replace the city name in the 'report_short' field with a specified token
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        for key in ['report_short', 'report_short_wout_boeen']: #, "gpt_rewritten_cleaned"]:
            data[key] = data[key].replace(data['city'], '<city>')
        if "gpt_rewritten_cleaned" in data: # if the key is present
            data["gpt_rewritten_cleaned"] = data["gpt_rewritten_cleaned"].replace(data['city'], '<city>')
        return data


class ReduceKeys:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        if "gpt_rewritten_cleaned" in data:
            reduced_set_of_keys = ["city", "overview", "report_short_wout_boeen", "report_short", "gpt_rewritten_cleaned"]
        reduced_set_of_keys = ["city", "overview", "report_short_wout_boeen", "report_short"]


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
            data["windgeschwindigkeit_in_km_per_h"],
            data["bewölkungsgrad"]
            ):\
            
            if s != "":
                s+= ";"
            
            s += f"{time};{clearness};{temp};{rain_risk};{rain_amount};{wind_direction};{wind_speed};{cloudiness}"
       
        data["overview"] = s
         
        return data