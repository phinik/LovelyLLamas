from typing import Dict, List


class ReplaceNaNs:
    """Replaces NaN values in the input dictionary with a specified token."""
    def __init__(self, missing_token: str = '<missing>'):
        self.missing_token = missing_token

    def __call__(self, data: Dict) -> Dict:
        return {k: v if v is not None else self.missing_token for k, v in data.items()}


class TokenizeUnits:
    """Tokenizes units in specific fields with corresponding mapped tokens."""
    def __init__(self, unit_map: Dict[str, str] = None):
        self.unit_map = unit_map or {
            '°C': ' <degC>',
            '°': ' <degC>',
            'l/m²': ' <l_per_sqm>',
            'km/h': ' <kmh>',
            '%': ' <percent>'
        }

    def __call__(self, data: Dict) -> Dict:
        for key in ['report_short', 'report_long']:
            if key in data:
                for unit, token in self.unit_map.items():
                    data[key] = data[key].replace(unit, token)
        return data


class ReplaceCityName:
    """Replaces occurrences of the city name in a specific field with a placeholder token."""
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        if 'report_short' in data and 'city' in data:
            data['report_short'] = data['report_short'].replace(data['city'], '<city>')
        return data


class ReduceKeys:
    """Reduces the dictionary to only the specified set of keys."""
    def __init__(self, keys_to_keep: List[str] = None):
        self.keys_to_keep = keys_to_keep or ['city', 'report_short', 'overview']

    def __call__(self, data: Dict) -> Dict:
        return {key: data.get(key) for key in self.keys_to_keep}


class AssembleCustomOverview:
    """Assembles a custom overview string based on provided time-series data."""
    def __call__(self, data: Dict) -> Dict:
        fields = ["times", "clearness", "temperatur_in_deg_C", "niederschlagsrisiko_in_perc",
                  "niederschlagsmenge_in_l_per_sqm", "windrichtung", "windgeschwindigkeit_in_km_per_s",
                  "bewölkungsgrad"]

        # validate all required fields exist
        for field in fields:
            if field not in data:
                raise ValueError(f"Field '{field}' not found in input data.")

        # generate string
        overview = []
        for time, clearness, temp, rain_risk, rain_amount, wind_dir, wind_speed, cloudiness in zip(
                data["times"], data["clearness"], data["temperatur_in_deg_C"],
                data["niederschlagsrisiko_in_perc"], data["niederschlagsmenge_in_l_per_sqm"],
                data["windrichtung"], data["windgeschwindigkeit_in_km_per_s"], data["bewölkungsgrad"]):

            clearness = clearness.replace(",", "")  # remove problematic commas
            overview.append(f"{time},{clearness},{temp},{rain_risk},{rain_amount},{wind_dir},{wind_speed},{cloudiness}")

        data["overview"] = ",".join(overview)
        return data


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        pass
