import unittest


from src.metrics import *


class TestCityAppearanceMetric(unittest.TestCase):
    def test_correct_score_results_if_some_have_token_and_some_not(self):
        metric = CityAppearance()
        metric.update("in <city> ist es cool. <city> <city>", [], "", "", [])
        metric.update("in ... ist es cool", [], "", "", [])
        metric.update("in <city> ist es mega", [], "", "", [])

        self.assertEqual(2, metric._n_samples_with_city)
        self.assertEqual(3, metric._n_samples)
        self.assertAlmostEqual(2 / 3, metric.get()["accuracy"])

    def test_correct_score_results_if_no_city_token_at_all(self):
        metric = CityAppearance()
        metric.update("in stadt ist es cool.", [], "", "", [])
        metric.update("in ... ist es cool", [], "", "", [])
        metric.update("in city ist es mega", [], "", "", [])

        self.assertEqual(0, metric._n_samples_with_city)
        self.assertEqual(3, metric._n_samples)
        self.assertAlmostEqual(0, metric.get()["accuracy"])

    def test_correct_score_results_if_all_have_token(self):
        metric = CityAppearance()
        metric.update("in <city> ist es cool. <city> <city>", [], "", "", [])
        metric.update("in <city> ist es cool", [], "", "", [])
        metric.update("in <city> ist es mega", [], "", "", [])

        self.assertEqual(3, metric._n_samples_with_city)
        self.assertEqual(3, metric._n_samples)
        self.assertAlmostEqual(1, metric.get()["accuracy"])


class TestTemperatureCorrectness(unittest.TestCase):
    def test_correct_score_results_if_some_values_are_correct_and_some_not(self):
        metric = TemperatureCorrectness()
        metric.update("in <city> ist es 5 <degC>. Temperature über 10 <degC>", [], "", "", ["1", "2", "5", "7", "10"])
        metric.update("in <city> ist es -5 <degC>. Temperatur unter -10 <decC>", [], "", "", ["-10", "-8", "-4", "1"])

        self.assertEqual(3, metric._n_correct_temp)
        self.assertEqual(1, metric._n_incorrect_temp)
        self.assertAlmostEqual(3 / 4, metric.get()["accuracy"])
        
    def test_correct_score_results_if_all_values_are_incorrect(self):
        metric = TemperatureCorrectness()
        metric.update("in <city> ist es 5 <degC>. Temperature über 10 <degC>", [], "", "", ["1", "2", "4", "7", "11"])
        metric.update("in <city> ist es -5 <degC>. Temperatur unter -10 <decC>", [], "", "", ["-11", "-8", "-4", "1"])

        self.assertEqual(0, metric._n_correct_temp)
        self.assertEqual(4, metric._n_incorrect_temp)
        self.assertAlmostEqual(0, metric.get()["accuracy"])

    def test_correct_score_results_if_all_values_are_correct(self):
        metric = TemperatureCorrectness()
        metric.update("in <city> ist es 5 <degC>. Temperature über 10 <degC>", [], "", "", ["1", "2", "5", "7", "10"])
        metric.update("in <city> ist es -5 <degC>. Temperatur unter -10 <decC>", [], "", "", ["-10", "-8", "-5", "1"])

        self.assertEqual(4, metric._n_correct_temp)
        self.assertEqual(0, metric._n_incorrect_temp)
        self.assertAlmostEqual(1, metric.get()["accuracy"])


class TestTemperatureRange(unittest.TestCase):
    def test_correct_score_results_if_predicted_interval_is_partially_intersecting_to_the_right(self):
        metric = TemperatureRange()
        metric.update("in <city> ist es 5 <degC>. Tagsüber 7 <degC>. Temperature über 10 <degC>", [], "", "", ["7", "8", "11"])

        self.assertAlmostEqual(4 / 6, metric.get()["mean_score"])

    def test_correct_score_results_if_predicted_interval_is_partially_intersecting_to_the_left(self):
        metric = TemperatureRange()
        metric.update("in <city> ist es 5 <degC>. Tagsüber 7 <degC>. Temperature über 10 <degC>", [], "", "", ["1", "2", "5", "7"])

        self.assertAlmostEqual(3 / 6, metric.get()["mean_score"])
        
    def test_correct_score_results_if_predicted_interval_is_not_intersecting_to_the_right(self):
        metric = TemperatureRange()
        metric.update("in <city> ist es 5 <degC>. Tagsüber 7 <degC>. Temperature über 10 <degC>", [], "", "", ["11", "12", "14"])

        self.assertAlmostEqual(0, metric.get()["mean_score"])

    def test_correct_score_results_if_predicted_interval_is_not_intersecting_to_the_left(self):
        metric = TemperatureRange()
        metric.update("in <city> ist es 5 <degC>. Tagsüber 7 <degC>. Temperature über 10 <degC>", [], "", "", ["1", "2", "4"])

        self.assertAlmostEqual(0, metric.get()["mean_score"])

    def test_correct_score_results_if_predicted_interval_is_fully_contained(self):
        metric = TemperatureRange()
        metric.update("in <city> ist es 5 <degC>. Tagsüber 7 <degC>. Temperature über 10 <degC>", [], "", "", ["1", "2", "4", "7", "10"])
    
        self.assertAlmostEqual(1, metric.get()["mean_score"])

    def test_correct_score_results_if_no_temperature_values_in_prediction(self):
        metric = TemperatureRange()
        metric.update("in <city> ist es <degC>. Tagsüber <degC>. Temperature über <degC>", [], "", "", ["1", "2", "4", "7", "10"])
    
        self.assertAlmostEqual(0, metric.get()["mean_score"])

    def test_correct_score_results_if_no_temperature_values_in_temperature(self):
        metric = TemperatureRange()
        metric.update("in <city> ist es 5 <degC>. Tagsüber 7 <degC>. Temperature über 10 <degC>", [], "", "", ["<missing>", "<missing>"])
    
        self.assertAlmostEqual(0, metric.get()["mean_score"])

    def test_correct_score_results_if_no_temperatuer_values_at_all(self):
        metric = TemperatureRange()
        metric.update("in <city> ist es <degC>. Tagsüber <degC>. Temperature über <degC>", [], "", "", ["<missing>", "<missing>"])
    
        self.assertAlmostEqual(0, metric.get()["mean_score"])

    def test_returned_score_is_the_mean_of_individual_scores(self):
        metric = TemperatureRange()
        metric.update("in <city> ist es 5 <degC>. Tagsüber 7 <degC>. Temperature über 10 <degC>", [], "", "", ["7", "8", "11"])
        metric.update("in <city> ist es 5 <degC>. Tagsüber 7 <degC>. Temperature über 10 <degC>", [], "", "", ["1", "2", "5", "7"])

        self.assertAlmostEqual(7 / 12, metric.get()["mean_score"])
    

if __name__ == "__main__":
    unittest.main()
