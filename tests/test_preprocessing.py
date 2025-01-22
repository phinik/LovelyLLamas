import unittest
import numpy as np
from src.data_preprocessing import *

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.data = {
            "city": "12 de Haina",
            "times": ["00 - 01 Uhr", "01 - 02 Uhr"],
            "clearness": ["Klar", "Leicht bewölkt"],
            "temperatur_in_deg_C": ["24", np.nan],
            "niederschlagsrisiko_in_perc": ["0", "NaN"],
            "niederschlagsmenge_in_l_per_sqm": ["0", np.nan],
            "windrichtung": ["N", "NO"],
            "windgeschwindigkeit_in_km_per_h": ["7", "13"],
            "bewölkungsgrad": ["2/8", "4/8"],
            "report_short": "In 12 de Haina ist es 30°C.",
            "report_short_wout_boeen": "In 12 de Haina stören am Morgen nur einzelne Wolken.",
            "gpt_rewritten_cleaned": "In 12 de Haina beträgt die Regenwahrscheinlichkeit 10 %.",
            "overview": ""
        }

    def test_replace_nans(self):
        preprocessor = ReplaceNaNs()
        result = preprocessor(self.data.copy())
        self.assertEqual(result["temperatur_in_deg_C"], ["24", "<missing>"])
        self.assertEqual(result["niederschlagsmenge_in_l_per_sqm"], ["0", "<missing>"])

    def test_tokenize_units(self):
        preprocessor = TokenizeUnits()
        result = preprocessor(self.data.copy())
        self.assertIn("<degC>", result["report_short"])
        self.assertIn("<percent>", result["gpt_rewritten_cleaned"])

    def test_replace_city_name(self):
        preprocessor = ReplaceCityName()
        result = preprocessor(self.data.copy())
        self.assertNotIn("12 de Haina", result["report_short"])
        self.assertIn("<city>", result["report_short"])
        self.assertIn("<city>", result["report_short_wout_boeen"])

    def test_reduce_keys(self):
        preprocessor = ReduceKeys()
        result = preprocessor(self.data.copy())
        self.assertIn("city", result)
        self.assertIn("report_short", result)
        self.assertNotIn("times", result)  # Should be removed

    def test_assemble_custom_overview(self):
        preprocessor = AssembleCustomOverview()
        result = preprocessor(self.data.copy())
        self.assertIn("00 - 01 Uhr", result["overview"])
        self.assertIn("Klar", result["overview"])
        self.assertIn("7", result["overview"])

if __name__ == "__main__":
    unittest.main()
