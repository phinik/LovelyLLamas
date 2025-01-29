from flask import Flask, jsonify, request
from flask_cors import CORS
from generate import *
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

config = {
    "name": "Whatever",
    "dataset": os.path.join(os.getcwd(), 'data'),
    "model_weights": os.path.join(os.getcwd(), "src_model", "best_model_CE_loss.pth"),
    "model_params": os.path.join(os.getcwd(), "src_model", "params.json"),
    "cached": True,
    "model": "transformer", #args.model,
    "block_size": 20,
    "tokenizer": 'bert',
    "target": "default"
    }

# Load model
model = TransformerFactory.from_file(config["model_params"])
model.load_weights_from(config["model_weights"])
model.to(DEVICE)
generator = Generator(config)


@app.route('/api/mock-llm', methods=['POST'])
def mock_llm_response():
    city = request.json.get('city', '')
    id = str(city).split('(')[-1][:-1]
    file_name = data[id][1]
    print(file_name)
    idx = dset.index(f'dataset/{file_name}')
    print(idx)
    text = generator.get(model, int(idx))
    return jsonify({
        # 'response': f"Based on my analysis, {city} is a fascinating city with a rich cultural heritage. "
        #            f"The city offers numerous attractions, diverse cuisine, and unique architectural styles. "
        #            f"Visitors to {city} often praise its blend of historical landmarks and modern amenities. "
        #            f"The local transportation system is well-developed, making it easy to explore various neighborhoods. "
        #            f"Would you like to know more about any specific aspect of {city}?"
        'response': text
    })

@app.route('/api/cities', methods=['GET'])
def get_cities():
    query = request.args.get('query', '').lower()
    filtered_cities = [city for city in CITIES if query in city.lower()]
    return jsonify(filtered_cities)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)