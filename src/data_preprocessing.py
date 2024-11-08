from typing import Dict

# so vom stil her wie the torchvision transforms wäre glaube ich ganz cool, also für jede einzelne Operation eine
# eigene Klasse ableiten

class ReplaceLineBreaksInOverview:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        # TODO
        return data
    

class IntroduceCustomTokens:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        # TODO
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

            s += f"{time},{clearness},{temp},{rain_risk},{rain_amount},{wind_direction},{wind_speed},{cloudiness}"
       
        data["overview"] = s
         
        return data
    
class ToTensor:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        pass