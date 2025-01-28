from flask import Flask, jsonify, request
from flask_cors import CORS
import json
app = Flask(__name__)
CORS(app)

with open('files.json', 'r') as f:
    files = json.load(f)
    # concatenate city with id
    CITIES = [f"{value['city']} ({value['id']})" for key, value in files.items()]

@app.route('/api/mock-llm', methods=['POST'])
def mock_llm_response():
    city = request.json.get('city', '')
    return jsonify({
        'response': f"Based on my analysis, {city} is a fascinating city with a rich cultural heritage. "
                   f"The city offers numerous attractions, diverse cuisine, and unique architectural styles. "
                   f"Visitors to {city} often praise its blend of historical landmarks and modern amenities. "
                   f"The local transportation system is well-developed, making it easy to explore various neighborhoods. "
                   f"Would you like to know more about any specific aspect of {city}?"
    })

@app.route('/api/cities', methods=['GET'])
def get_cities():
    query = request.args.get('query', '').lower()
    filtered_cities = [city for city in CITIES if query in city.lower()]
    return jsonify(filtered_cities)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)