import unittest


from src.data_preprocessing import *


class TestReplaceNaNs(unittest.TestCase):
    def test_nothing_happens_if_no_nans_are_present(self):
        test_dict = {
            "times": ["1", "2", "3", "4", "5", "6"], 
            "clearness": ["1", "2", "3", "4", "5", "6"], 
            "temperatur_in_deg_C": ["1", "2", "3", "4", "5", "6"], 
            "niederschlagsrisiko_in_perc": ["1", "2", "3", "4", "5", "6"], 
            "niederschlagsmenge_in_l_per_sqm": ["1", "2", "3", "4", "5", "6"], 
            "windrichtung": ["1", "2", "3", "4", "5", "6"],  
            "windgeschwindigkeit_in_km_per_h": ["1", "2", "3", "4", "5", "6"], 
            "bewölkungsgrad": ["1", "2", "3", "4", "5", "6"]
        }

        transform = ReplaceNaNs()

        transformed_test_dict = transform(test_dict)

        self.assertEqual(list(test_dict.keys()), list(transformed_test_dict.keys()))

        for val in transformed_test_dict.values():
            self.assertEqual(["1", "2", "3", "4", "5", "6"], val)

    def test_nans_are_correctly_replaced(self):
        test_dict = {
            "times": ["1", np.nan, "3", np.nan, "5", "6"], 
            "clearness": [np.nan, "2", "3", "4", "5", "6"], 
            "temperatur_in_deg_C": ["1",np.nan, "3", "4", np.nan, "6"], 
            "niederschlagsrisiko_in_perc": ["1", "2", np.nan, np.nan, "5", "6"], 
            "niederschlagsmenge_in_l_per_sqm": ["1", "2", "3", "4", "5", np.nan], 
            "windrichtung": [np.nan, np.nan, np.nan, "4", "5", "6"], 
            "windgeschwindigkeit_in_km_per_h": ["1", "2", "3", "4", np.nan, np.nan], 
            "bewölkungsgrad": [np.nan, "2", np.nan, "4", np.nan, "6"]
        }

        transform = ReplaceNaNs()

        transformed_test_dict = transform(test_dict)

        self.assertEqual(list(test_dict.keys()), list(transformed_test_dict.keys()))

        self.assertEqual(["1", "<missing>", "3", "<missing>", "5", "6"], transformed_test_dict["times"])
        self.assertEqual(["<missing>", "2", "3", "4", "5", "6"], transformed_test_dict["clearness"])
        self.assertEqual(["1", "<missing>", "3", "4", "<missing>", "6"], transformed_test_dict["temperatur_in_deg_C"])
        self.assertEqual(["1", "2", "<missing>", "<missing>", "5", "6"], transformed_test_dict["niederschlagsrisiko_in_perc"])
        self.assertEqual(["1", "2", "3", "4", "5", "<missing>"], transformed_test_dict["niederschlagsmenge_in_l_per_sqm"])
        self.assertEqual(["<missing>", "<missing>", "<missing>", "4", "5", "6"], transformed_test_dict["windrichtung"])
        self.assertEqual(["1", "2", "3", "4", "<missing>", "<missing>"], transformed_test_dict["windgeschwindigkeit_in_km_per_h"])
        self.assertEqual(["<missing>", "2", "<missing>", "4", "<missing>", "6"], transformed_test_dict["bewölkungsgrad"])


class TestTokenizeUnits(unittest.TestCase):
    def test_units_are_correctly_tokenized(self) -> Dict:
        test_dict = {
            'report_short_wout_boeen': "5°C 10l/m² 30km/h 5% 30km/h 10l/m² 5°C 5%",
            "gpt_rewritten_cleaned": "5°C 10l/m² 30km/h 5% 30km/h 10l/m² 5°C 5%"
        }
         
        transform = TokenizeUnits()

        transformed_test_dict = transform(test_dict)

        self.assertEqual(
            "5 <degC> 10 <l_per_sqm> 30 <kmh> 5 <percent> 30 <kmh> 10 <l_per_sqm> 5 <degC> 5 <percent>",
            transformed_test_dict["report_short_wout_boeen"]
        )
        self.assertEqual(
            "5 <degC> 10 <l_per_sqm> 30 <kmh> 5 <percent> 30 <kmh> 10 <l_per_sqm> 5 <degC> 5 <percent>",
            transformed_test_dict["gpt_rewritten_cleaned"]
        )
    

class TestCityReplaceCityName(unittest.TestCase):
    def test_city_name_is_replaced_correctly(self):
        test_dict = {
            "city": "Hamburg",
            "report_short_wout_boeen": "In Hamburg schneit es. Hamburg liegt unter einer Schneedecke", 
            "gpt_rewritten_cleaned": "In Hamburg regnet es"
        }
                
        transform = ReplaceCityName()

        transformed_test_dict = transform(test_dict)

        self.assertEqual(
            "In <city> schneit es. <city> liegt unter einer Schneedecke", 
            transformed_test_dict["report_short_wout_boeen"]
        )

        self.assertEqual("In <city> regnet es", transformed_test_dict["gpt_rewritten_cleaned"])
    

class TestReduceKeys(unittest.TestCase):
    def test_correct_keys_result(self):
        test_dict = {
            "city": "",
            "overview_full": "", 
            "overview_ctpc": "", 
            "overview_ctc": "", 
            "overview_ct": "",
            "overview_tpwc": "", 
            "report_short_wout_boeen": "",
            "gpt_rewritten_cleaned": "",
            "gpt_rewritten_apo": "",
            "temperatur_in_deg_C": "",
            "key_to_remove": ""
        }
        
        transform = ReduceKeys()

        transformed_test_dict = transform(test_dict)

        self.assertEqual(
            ["city", "overview_full", "overview_ctpc", "overview_ctc", "overview_ct", "overview_tpwc",
             "report_short_wout_boeen", "gpt_rewritten_cleaned", "gpt_rewritten_apo", "temperatur_in_deg_C"],
            list(transformed_test_dict.keys())
        )

if __name__ == "__main__":
    unittest.main()
