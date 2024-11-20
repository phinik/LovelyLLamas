from tokenizer import Tokenizer
from dataloader import get_train_dataloader_weather_dataset, get_eval_dataloader_weather_dataset
from dataset import WeatherDataset, Split, TransformationPipeline
from data_preprocessing import *

# Initialisierung der Transformationen
transformations = TransformationPipeline([
    ReplaceNaNs(),
    ReplaceCityName(),
    TokenizeUnits(),
    AssembleCustomOverview(),
    ReduceKeys()
])

# Dataloader für das Training
train_dataloader = get_train_dataloader_weather_dataset(
    path='C:/Users/Agando/Desktop/Uni/Master-Projekt/debug_dataset', 
    batch_size=32,
    cached=True
)

# Dataloader für die Evaluation
eval_dataloader = get_eval_dataloader_weather_dataset(
    path='C:/Users/Agando/Desktop/Uni/Master-Projekt/debug_dataset', 
    batch_size=32, 
    cached=True
)

# Initialisiere den Tokenizer
tokenizer = Tokenizer(dataset_path='path_to_vocab')
# add tokens to the tokenizer
tokenizer.add_custom_tokens(['<start>', '<stop>'])

# Beispiel für die Tokenisierung einer Eingabe
input_text = "Berlin hat eine Temperatur von 20°C und es regnet."
token_ids = tokenizer.stoi_context(input_text)

import torch
import torch.nn as nn
import torch.optim as optim

# WeatherLSTM Model
class WeatherLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(WeatherLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        output = self.fc(lstm_out)  # Only use the last LSTM output
        return output

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        inputs = batch["overview"]  # Example input text
        targets = batch["report_short"]  # Example target text
        
        # Tokenize input and target data with padding and truncation
        input_encodings = tokenizer.encode_plus_context(inputs)
        target_encodings = tokenizer.encode_plus_target(targets)
        
        # Extract input IDs and attention masks
        input_ids = input_encodings['input_ids'].to(device)
        target_ids = target_encodings['input_ids'].to(device)

        # Ensure that input and target lengths match
        assert input_ids.size(1) == target_ids.size(1), f"Input length {input_ids.size(1)} != Target length {target_ids.size(1)}"

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids)
        
        # Ensure outputs shape is (batch_size * seq_len, num_classes)
        outputs = outputs.view(-1, outputs.shape[-1])  # Flatten for cross-entropy
        target_ids = target_ids.view(-1)  # Flatten target

        # Calculate loss)
        loss = criterion(outputs, target_ids)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in dataloader:
            inputs = batch["overview"]
            targets = batch["report_short"]
            
            input_encodings = tokenizer.encode_plus_context(inputs)
            target_encodings = tokenizer.encode_plus_target(targets)
            
            input_ids = input_encodings['input_ids'].to(device)
            target_ids = target_encodings['input_ids'].to(device)

            # Ensure lengths match
            assert input_ids.size(1) == target_ids.size(1), f"Input length {input_ids.size(1)} != Target length {target_ids.size(1)}"

            # Forward pass
            outputs = model(input_ids)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.shape[-1])
            target_ids = target_ids.view(-1)

            # Calculate loss
            loss = criterion(outputs, target_ids)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Hyperparameters
embedding_dim = 256
hidden_dim = 512
output_dim = tokenizer.size_target_vocab
vocab_size = tokenizer.size_context_vocab

# Initialize model, optimizer, and criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WeatherLSTM(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduction="sum")

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion, device)
    eval_loss = evaluate(model, eval_dataloader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Eval Loss: {eval_loss:.4f}")

print("Finished training.")

# save model
torch.save(model.state_dict(), 'weather_lstm.pth')

# Load model
#model.load_state_dict(torch.load('weather_lstm.pth'))

# Let model predict
input_text = "In Addis Abeba ist es am Morgen überwiegend dicht bewölkt und die Temperatur liegt bei 14°C. Im Laufe des Mittags kommt es zu Regenschauern und das Thermometer klettert auf 20°C. Abends ist es regnerisch bei Werten von 16 bis zu 17°C. Nachts sind anhaltende Regen-Schauer zu erwarten bei Tiefsttemperaturen von 14°C."
input_encodings = tokenizer.encode_plus_context(input_text)
input_ids = input_encodings['input_ids'].to(device)
output = model(input_ids)

# Apply argmax to get the predicted token IDs
predicted_ids = torch.argmax(output, dim=-1)  # Get most probable token for each position

# Flatten the predicted_ids tensor to a 1D list
predicted_ids = predicted_ids.squeeze().cpu().numpy()  # Move to CPU and convert to numpy

# Convert output token IDs to text
output_text = tokenizer.itos_targets(predicted_ids)

print(output_text)  # Print the predicted text
