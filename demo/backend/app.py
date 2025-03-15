from flask import Flask, jsonify, request
from flask_cors import CORS
from generate_transformer import *
from src.models.lstm import LSTM
# Import LSTM generator if available
# from generate_lstm import LSTMGenerator
import json
import os
app = Flask(__name__)
CORS(app)

# Get List of cities
with open(os.path.join(os.getcwd(), 'assets', 'cities.json'), 'r', encoding = 'utf-8') as f:
    data = json.load(f)
    CITIES = []
    for id, vals in data.items():
        CITIES.append(f'{vals[0]} ({id})')
    # sort cities alphabetically
    CITIES.sort()

with open(os.path.join(os.getcwd(), 'data', 'dset_test.json'), 'r', encoding = 'utf-8') as f:
    dset: list = json.load(f)

# Base configuration
base_config = {
    "name": "Whatever",
    "dataset": os.path.join(os.getcwd(), 'data'),
    "cached": True,
    "block_size": 20,
    "tokenizer": 'bert',
    "target": "default",
    "overview": "full",
    "num_samples": -1
}

# Transformer specific configuration
transformer_config = base_config.copy()
transformer_config.update({
    "model": "transformer",
    "model_weights": os.path.join(os.getcwd(), "src_model", "best_model_CE_loss.pth"),
    "model_params": os.path.join(os.getcwd(), "src_model", "params.json"),
})

# LSTM specific configuration
lstm_config = base_config.copy()
lstm_config.update({
    "model": "lstm",
    "model_weights": os.path.join(os.getcwd(), "src_model", "best_model_CE_loss_lstm.pth"),
    "model_params": os.path.join(os.getcwd(), "src_model", "params_lstm.json"),
})

# Initialize model instances
transformer_model = None
lstm_model = None

def initialize_transformer():
    global transformer_model
    if transformer_model is None:
        print("Initializing Transformer model...")
        model = TransformerFactory.from_file(transformer_config["model_params"])
        model.load_weights_from(transformer_config["model_weights"])
        model.to(DEVICE)
        transformer_model = model
    return transformer_model

def initialize_lstm():
    global lstm_model
    if lstm_model is None:
        print("Initializing LSTM model...")
        # Replace with your actual LSTM initialization code
        # For example:
        # model = LSTMFactory.from_file(lstm_config["model_params"])
        # model.load_weights_from(lstm_config["model_weights"])
        # model.to(DEVICE)
        # lstm_model = model
        with open(lstm_config["model_params"], "r") as f:
            params = json.load(f)
            c = {k: v for k, v in params.items() if k in LSTM.__init__.__code__.co_varnames}
            model = LSTM(**c)
            model.load_weights_from(lstm_config["model_weights"])
            model.to(DEVICE)
        # Temporary placeholder for demonstration
        lstm_model = initialize_transformer()  # Replace with actual LSTM init
    return lstm_model

# Initialize transformer by default
transformer_generator = Generator(transformer_config)
# LSTM generator would be initialized similarly
lstm_generator = Generator(lstm_config)

@app.route('/api/mock-llm', methods=['POST'])
def mock_llm_response():
    data_json = request.json
    city = data_json.get('city', '')
    model_type = data_json.get('modelType', 'transformer')  # Default to transformer if not specified
    
    id = str(city).split('(')[-1][:-1]
    file_name = data[id][1]
    print(f"File name: {file_name}")
    idx = dset.index(f'dataset/{file_name}')
    print(f"Index: {idx}")
    
    # Select model based on the requested type
    if model_type == 'transformer':
        print("Using Transformer model")
        model = initialize_transformer()
        text = transformer_generator.get(model, int(idx))
    elif model_type == 'lstm':
        print("Using LSTM model")
        model = initialize_lstm()
        # If you have a different generator for LSTM, use that instead
        text = lstm_generator.get(model, int(idx))
        # text = transformer_generator.get(model, int(idx))  # Replace with LSTM generator
        # text = "[LSTM] " + text  # Adding a prefix to differentiate in the UI
    else:
        return jsonify({
            'error': f"Unknown model type: {model_type}"
        }), 400
    
    return jsonify({
        'response': text,
        'modelUsed': model_type
    })

@app.route('/api/cities', methods=['GET'])
def get_cities():
    query = request.args.get('query', '').lower()
    filtered_cities = [city for city in CITIES if query in city.lower()]
    return jsonify(filtered_cities)

if __name__ == '__main__':
    # Pre-initialize the transformer model
    initialize_transformer()
    app.run(host='0.0.0.0', port=5000)