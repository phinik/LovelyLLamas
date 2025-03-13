import unittest


from src.data_postprocessing import *

     
class TestRemoveCityToken(unittest.TestCase):
    def test_city_token_is_correctly_replaced(self):
        s_test = "In <city> ist es schön. Es wird 5 <degC> in <city>."

        transform = ReplaceCityToken()

        s_transformed = transform(s_test, "Hamburg")

        self.assertEqual("In Hamburg ist es schön. Es wird 5 <degC> in Hamburg.", s_transformed)


class TestRemovePunctuation(unittest.TestCase):
    def test_punctuation_is_correctly_removed(self):
        s_test = "In Hamburg ist es schön, die Sonne scheint. Es wird 5 <degC>; schneit es etwa? Er sagt: Halt!"

        transform = RemovePunctutation()

        s_transformed = transform(s_test)

        self.assertEqual(
            "In Hamburg ist es schön die Sonne scheint Es wird 5 <degC> schneit es etwa Er sagt Halt", 
            s_transformed
        )


class TestPostProcess(unittest.TestCase):
    def test_strings_are_correctly_transformed(self):
        s_test = " In Hamburg    schneit  es bei - 17 <degC>    "

        transform = PostProcess()

        s_transformed = transform(s_test)

        self.assertEqual("In Hamburg schneit es bei -17 <degC>", s_transformed)


class TestRemoveRepeatedSpaces(unittest.TestCase):
    def test_strings_wout_repated_spaces_are_not_changed(self):
        s_test = "In Hamburg ist es schön"

        transform = RemoveRepeatedSpaces()

        s_transformed = transform(s_test)

        self.assertEqual(s_test, s_transformed)

    def test_correct_string_results_for_repeated_spaces(self):
        s_test = "In  Hamburg    ist es      schön"

        transform = RemoveRepeatedSpaces()

        s_transformed = transform(s_test)

        self.assertEqual("In Hamburg ist es schön", s_transformed)


class TestCombineNegativeTemperatures(unittest.TestCase):
    def test_positive_temperature_are_not_affected(self):
        s_test = "In Hamburg wird es 15 <degC>."

        transform = CombineNegativeTemperatures()

        s_transformed = transform(s_test)

        self.assertEqual(s_test, s_transformed)

    def test_correct_negative_temperature_are_not_affected(self):
        s_test = "In Hamburg wird es -15 <degC>."

        transform = CombineNegativeTemperatures()

        s_transformed = transform(s_test)

        self.assertEqual(s_test, s_transformed)

    def test_incorrect_negative_temperature_are_corrected(self):
        s_test = "In Hamburg wird es - 15 <degC>."

        transform = CombineNegativeTemperatures()

        s_transformed = transform(s_test)

        self.assertEqual("In Hamburg wird es -15 <degC>.", s_transformed)


class TestStripSpaces(unittest.TestCase):
    def test_leading_spaces_are_correctly_removed(self):
        s_test = "    In Hamburg wird es 15 <degC>."

        transform = StripSpaces()

        s_transformed = transform(s_test)

        self.assertEqual("In Hamburg wird es 15 <degC>.", s_transformed)

    def test_trailing_spaces_are_correctly_removed(self):
        s_test = "In Hamburg wird es 15 <degC>.    "

        transform = StripSpaces()

        s_transformed = transform(s_test)

        self.assertEqual("In Hamburg wird es 15 <degC>.", s_transformed)

    def test_leading_and_trailing_spaces_are_correctly_removed(self):
        s_test = "    In Hamburg wird es 15 <degC>.    "

        transform = StripSpaces()

        s_transformed = transform(s_test)

        self.assertEqual("In Hamburg wird es 15 <degC>.", s_transformed)

    def test_correct_strings_are_not_affected(self):
        s_test = "In Hamburg wird es 15 <degC>."

        transform = StripSpaces()

        s_transformed = transform(s_test)

        self.assertEqual(s_test, s_transformed)


if __name__ == "__main__":
    unittest.main()
