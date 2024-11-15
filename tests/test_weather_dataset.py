import unittest


from src.dataset import *


class TestWeatherDataset(unittest.TestCase):
    def test_correct_path_results(self):
        dset = WeatherDataset("./test_weather_dataset", Split.EVAL, TransformationPipeline([]))

        self.assertEqual("./test_weather_dataset/eval", dset._full_path)

    def test_correct_number_of_files_if_not_cached(self):
        dset = WeatherDataset("./test_weather_dataset", Split.EVAL, TransformationPipeline([]))

        self.assertEqual(4, len(dset))

    def test_correct_number_of_files_if_cached(self):
        dset = WeatherDataset("./test_weather_dataset", Split.EVAL, TransformationPipeline([]), cached=True)

        self.assertEqual(4, len(dset))
        self.assertEqual(len(dset._cached_data), len(dset))

    def test_correct_dict_is_returned(self):
        dset = WeatherDataset("./test_weather_dataset", Split.EVAL, TransformationPipeline([]))

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

        self.assertEquals(expected_keys, list(dset.__getitem__(0).keys()))

    def test_if_cached_data_from_cache_is_returned(self):
        dset = WeatherDataset("./test_weather_dataset", Split.EVAL, TransformationPipeline([]), cached=True)
        dummy_cached_data = [f"Entry{i}" for i in range(len(dset))]

        # Set _cached_data to dummy_cached_data so that it can be verified that data from _cached_data is returned.
        dset._cached_data = dummy_cached_data

        self.assertTrue(dset.__getitem__(0) in dummy_cached_data)


if __name__ == "__main__":
    unittest.main()
