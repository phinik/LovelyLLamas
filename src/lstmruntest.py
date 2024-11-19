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
input_text = "Berlin hat eine Temperatur von 20°C"
token_ids = tokenizer.stoi_context(input_text)
print(token_ids)  # Gibt die Token-IDs aus

import torch
import torch.nn as nn
import torch.optim as optim

class WeatherLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(WeatherLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])  # Nur das letzte LSTM-Ausgabe verwenden
        return output

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Angenommen, batch enthält 'overview' und 'report_short' (oder das, was du vorhersagen möchtest)
        inputs = batch["overview"]  # Beispiel
        targets = batch["report_short"]  # Beispiel
        
        # Tokenisiere die Eingabedaten (hier: Übersicht)
        input_ids = tokenizer.stoi_context(inputs)
        input_ids = torch.tensor(input_ids).to(device)
        target_ids = tokenizer.stoi_targets(targets)
        target_ids = torch.tensor(target_ids).to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Modellvorhersage
        outputs = model(input_ids)

        # Verlustberechnung
        loss = criterion(outputs, target_ids)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

# Hyperparameter
embedding_dim = 256
hidden_dim = 512
output_dim = len(tokenizer.stoi_targets("<start>")) # Beispiel: für die Zieltoken-IDs
vocab_size = tokenizer.size_context_vocab

# Modell und Optimierer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WeatherLSTM(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Trainingsloop
for epoch in range(10):  # Beispiel: 10 Epochen
    train_loss = train(model, train_dataloader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["overview"]
            targets = batch["report_short"]
            
            input_ids = tokenizer.stoi_context(inputs)
            input_ids = torch.tensor(input_ids).to(device)
            target_ids = tokenizer.stoi_targets(targets)
            target_ids = torch.tensor(target_ids).to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, target_ids)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Beispiel für Evaluierung
eval_loss = evaluate(model, eval_dataloader, device)
print(f"Eval Loss: {eval_loss}")
