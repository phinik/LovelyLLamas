import unittest


from src.dataset import *


class TestWeatherDatasetClassifier(unittest.TestCase):
    def test_correct_path_results(self):
        dset = WeatherDatasetClassifier("./test_weather_dataset", Split.EVAL, TransformationPipeline([]))

        self.assertEqual("./test_weather_dataset/dset_eval.json", dset._full_path)

    def test_correct_number_of_files(self):
        dset = WeatherDatasetClassifier("./test_weather_dataset", Split.EVAL, TransformationPipeline([]))

        self.assertEqual(4, len(dset))
        self.assertEqual(len(dset._cached_data), len(dset))

    def test_correct_dict_is_returned(self):
        dset = WeatherDatasetClassifier("./test_weather_dataset", Split.EVAL, TransformationPipeline([]))

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
            "windgeschwindigkeit_in_km_per_h",
            "luftdruck_in_hpa",
            "relative_feuchte_in_perc",
            "bewölkungsgrad",
            "report_short_wout_boeen",
            "class_1",
            "class_0"
        ]

        self.assertEqual(sorted(expected_keys), sorted(list(dset.__getitem__(0).keys())))


if __name__ == "__main__":
    unittest.main()
