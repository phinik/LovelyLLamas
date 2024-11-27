import torch
from models.lstm import LSTM
from tokenizer import Tokenizer
from data_preprocessing import ReplaceNaNs, ReplaceCityName, TokenizeUnits, AssembleCustomOverview, ReduceKeys
from dataset import TransformationPipeline
import json


def preprocess_input(text: str = None, city_name: str = None, json_path: str = None) -> str:
    """
    Preprocesses input text or loads and preprocesses data from a JSON file.
    :param text: Raw input text to be processed (if not using JSON).
    :param city_name: Name of the city to replace in the text (required for text input).
    :param json_path: Path to a JSON file containing input data.
    :return: Preprocessed text ready for tokenization and prediction.
    """
    pipeline = TransformationPipeline([
        ReplaceNaNs(),
        ReplaceCityName(),
        TokenizeUnits(),
        AssembleCustomOverview(),
        ReduceKeys()
    ])

    if json_path:
        # Load data from JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    else:
        # Example input structure for preprocessing
        if not text or not city_name:
            raise ValueError("Either 'text' and 'city_name' or 'json_path' must be provided.")
        input_data = {
            "report_short": text,
            "city": city_name,
            "overview": text,
            "times": [],
            "clearness": [],
            "temperatur_in_deg_C": [],
            "niederschlagsrisiko_in_perc": [],
            "niederschlagsmenge_in_l_per_sqm": [],
            "windrichtung": [],
            "windgeschwindigkeit_in_km_per_s": [],
            "bewölkungsgrad": []
        }

    preprocessed_data = pipeline(input_data)
    # Debugging Step: Print the preprocessed text
    print("Preprocessed Text:", preprocessed_data["overview"])
    return "<start> " + preprocessed_data["overview"] + " <stop>"



def load_model(model_path, vocab_size, embedding_dim, hidden_dim, output_dim, device):
    """
    Load the trained WeatherLSTM model.
    """
    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def generate_weather_report(model, tokenizer, text, device):
    """
    Generate a weather report based on input text.
    """
    # Encode the preprocessed text
    tokenized_text = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)

    # Debugging Step: Print tokenized input
    print("Tokenized Input IDs:", tokenized_text)

    with torch.no_grad():
        prediction = model(tokenized_text, tokenized_text)
        prediction_ids = prediction.argmax(dim=-1).squeeze(0).cpu().numpy()

    # Decode the prediction into text
    decoded_prediction = tokenizer.decode(prediction_ids)

    # Debugging Step: Print decoded prediction
    print("Decoded Prediction:", decoded_prediction)
    return decoded_prediction


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = Tokenizer(dataset_path='C:/Users/Agando/Desktop/Uni/Master-Projekt/debug_dataset')
tokenizer.add_custom_tokens(['<start>', '<stop>', '<degC>, <city>'])

# Ensure consistency with training tokenizer state
tokenizer_vocab_size = tokenizer.vocab_size

model = load_model(
    model_path="weather_lstm.pth",
    vocab_size=tokenizer_vocab_size,
    embedding_dim=256,
    hidden_dim=512,
    output_dim=tokenizer_vocab_size,
    device=device
)

# Input text or JSON file path
json_file_path = "C:/Users/Agando/Desktop/Uni/Master-Projekt/debug_dataset/train/2024-11-01_Addis Ababa_standardised.json"  # Set to None if not using JSON
raw_text = None  # Set to your input text if not using JSON
city_name = None  # Set to your city name if using text input

preprocessed_text = preprocess_input(text=raw_text, city_name=city_name, json_path=json_file_path)

# Generate prediction
predicted_report = generate_weather_report(model, tokenizer, preprocessed_text, device)
print("Predicted Weather Report:")
print(predicted_report)
