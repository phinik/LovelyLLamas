from airllm import AutoModel
import time

# Load model with 4-bit quantization for efficiency
model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B", compression="4bit")  

# Ensure the tokenizer has a pad_token
if model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token  # Set pad_token to eos_token
city = 'Abeïd'
weather_report = "In Abeïd strahlt am Tag die Sonne und die Temperaturen liegen zwischen 16 und 26°C. In der Nacht ist der Himmel bedeckt bei Werten von 16°C. Böen können Geschwindigkeiten zwischen 22 und 42 km/h erreichen."
# Improved input prompt
input_text = [
f'''Du bist ein professioneller Autor, welcher Wetterberichte und Flachwitze schreibt. Nun möchtest du beides kombinieren und deine Wetterberichte in lustige, und durchaus
etwas unseriös klingende Wetterberichte verwandeln. Schreibe bitte folgenden Wetterbericht um:
{weather_report}'''
]
# , der eine fesselnde Erzählung über die Wetterbedingungen in {city} gestaltet.

# Aufgabe: Schreibe eine lebendige, konversative Beschreibung, die das Wesen des Wetters einfängt. Beinhalte:

# Atmosphärische Beschreibung
# Temperatur- und Sonnenlichtdetails
# Windcharakteristiken
# Vorgeschlagene Aktivitäten im Freien
# Stimmung/Gefühl des Tages

# Stil: Lokaler Nachrichtenblog-Beitrag mit einem freundlichen, beschreibenden Ton. Priorisiere Immersion und Nachvollziehbarkeit.
# Ursprünglicher Wetterbericht:
# {weather_report}
# Deine Erzählung:'''
# ]

# Set the length dynamically for input and output
MAX_LENGTH = len(input_text[0])  # Adjust based on typical input length
MAX_NEW_TOKENS = 200  # Allow for extended output

# Start the timer
start_time = time.time()

# Tokenize input with attention mask
input_tokens = model.tokenizer(
    input_text,      
    return_tensors="pt",       
    truncation=True,       
    max_length=MAX_LENGTH,       
    padding=True  # Ensure padding is applied
)  

# Ensure pad_token_id is set in the model config
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.tokenizer.pad_token_id

# Generate text
generation_output = model.generate(      
    input_tokens['input_ids'].cuda(),       
    attention_mask=input_tokens['attention_mask'].cuda(),  # Pass attention mask
    max_new_tokens=MAX_NEW_TOKENS,      
    use_cache=True,      
    temperature=1.0,      
    top_p=0.9,      
    return_dict_in_generate=True
)  

# Stop the timer
end_time = time.time()

# Decode and print output
output = model.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
print(output)

# Print elapsed time
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time:.2f} seconds")
