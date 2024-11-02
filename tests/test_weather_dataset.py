import unittest

from src.dataset import WeatherDataset, Split

class TestWeatherDataset(unittest.TestCase):
    def test_correct_path_results(self):
        dset = WeatherDataset("./test_weather_dataset", Split.EVAL)

        self.assertEqual("./test_weather_dataset/eval", dset._full_path)

    def test_correct_numer_of_files(self):
        dset = WeatherDataset("./test_weather_dataset", Split.EVAL)

        self.assertEqual(4, len(dset))

    def test_correct_dict_is_returned(self):
        dset = WeatherDataset("./test_weather_dataset", Split.EVAL)

        expected_keys = [
            "city",
            "created_day",
            "created_time",
            "report_short",
            "report_long",
            "sunrise",
            "sundown",
            "sunhours",
            "times",
            "clearness",
            "overview",
            "temperatur_in_deg_C",
            "niederschlagsrisiko_in_perc",
            "niederschlagsmenge_in_l_per_sqm",
            "windrichtung",
            "windgeschwindigkeit_in_km_per_s",
            "luftdruck_in_hpa",
            "relative_feuchte_in_perc",
            "bewölkungsgrad"
        ]

        for key in expected_keys:
            self.assertTrue(key in dset.__getitem__(0).keys())

if __name__ == "__main__":
    unittest.main()