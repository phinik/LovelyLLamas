import numpy as np

from typing import Dict, List


class ReplaceNaNs:
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

        return data


class TokenizeUnits:
    def __init__(self, unit_map: Dict[str, str] = None):
        self.unit_map = unit_map or {
            '°C': ' <degC>',
            'l/m²': ' <l_per_sqm>',
            'km/h': ' <kmh>',
            '%': ' <percent>'
        }

    def __call__(self, data: Dict) -> Dict:
        for key in ['report_short_wout_boeen', "gpt_rewritten_cleaned"]:
            for unit, token in self.unit_map.items():
                data[key] = data[key].replace(unit, token)

        return data


class ReplaceCityName:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        for key in ['report_short_wout_boeen', "gpt_rewritten_cleaned"]:
            data[key] = data[key].replace(data['city'], '<city>')
        return data


class ReduceKeys:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        reduced_set_of_keys = [
            "city", 
            "overview_full",
            "overview_ctpc",
            "overview_ctc",
            "overview_ct",
            "overview_tpwc", 
            "report_short_wout_boeen", 
            "gpt_rewritten_cleaned", 
            "temperatur_in_deg_C"
        ]

        reduced_dict = {}
        for key in reduced_set_of_keys:
            reduced_dict[key] = data[key]

        return reduced_dict


class AssembleFullOverview:
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
       
        data["overview_full"] = s
         
        return data
    

class AssembleOverviewCTPC:
    """
    CTPC => Clearness, Temperature, Precipitation, Cloudiness
    """
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        s = ""

        for time, clearness, temp, rain_risk, rain_amount, cloudiness in zip(
            data["times"], 
            data["clearness"], 
            data["temperatur_in_deg_C"], 
            data["niederschlagsrisiko_in_perc"],
            data["niederschlagsmenge_in_l_per_sqm"], 
            data["bewölkungsgrad"]
            ):\
            
            if s != "":
                s+= ";"
            
            s += f"{time};{clearness};{temp};{rain_risk};{rain_amount};{cloudiness}"
       
        data["overview_ctpc"] = s
         
        return data
    

class AssembleOverviewCTC:
    """
    CTC => Clearness, Temperature, Cloudiness
    """
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        s = ""

        for time, clearness, temp, cloudiness in zip(
            data["times"], 
            data["clearness"], 
            data["temperatur_in_deg_C"], 
            data["bewölkungsgrad"]
            ):\
            
            if s != "":
                s+= ";"
            
            s += f"{time};{clearness};{temp};{cloudiness}"
       
        data["overview_ctc"] = s
         
        return data
    

class AssembleOverviewCT:
    """
    CT => Clearness, Temperature
    """
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        s = ""

        for time, clearness, temp in zip(
            data["times"], 
            data["clearness"], 
            data["temperatur_in_deg_C"], 
            ):\
            
            if s != "":
                s+= ";"
            
            s += f"{time};{clearness};{temp}"
       
        data["overview_ct"] = s
         
        return data
    

class AssembleOverviewTPWC:
    """
    TPWC => Temperature, Precipitation, Wind, Cloudiness
    """
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        s = ""

        for time, temp, rain_risk, rain_amount, wind_direction, wind_speed, cloudiness in zip(
            data["times"], 
            data["temperatur_in_deg_C"], 
            data["niederschlagsrisiko_in_perc"],
            data["niederschlagsmenge_in_l_per_sqm"], 
            data["windrichtung"], 
            data["windgeschwindigkeit_in_km_per_h"],
            data["bewölkungsgrad"]
            ):\
            
            if s != "":
                s+= ";"
            
            s += f"{time};{temp};{rain_risk};{rain_amount};{wind_direction};{wind_speed};{cloudiness}"
       
        data["overview_tpwc"] = s
         
        return data