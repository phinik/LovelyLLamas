from tokenizer import Tokenizer
from dataloader import get_train_dataloader_weather_dataset, get_eval_dataloader_weather_dataset
from dataset import WeatherDataset, Split, TransformationPipeline
from data_preprocessing import *

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
tokenizer = Tokenizer(dataset_path='C:/Users/Agando/Desktop/Uni/Master-Projekt/debug_dataset')
# add tokens to the tokenizer
tokenizer.add_custom_tokens(['<start>', '<stop>', '<degC>', '<l_per_sqm>', '<kmh>', '<percent>'])

# Tokenize the sample text
#tokenized_output = tokenizer.stoi_context(sample_text)

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
    
    for i, batch in enumerate(dataloader):
        context = batch["overview"]  # Example input text
        targets = batch["report_short"]  # Example target text
        
        # Tokenize input and target data with padding and truncation
        for j in range(len(context)):
            print(context[j])
            context[j] = torch.tensor(tokenizer.stoi_context(context[j])).unsqueeze(0)
            targets[j] = torch.tensor(tokenizer.stoi_targets("<start> " + targets[j] + " <stop>"))
        
        targets = nn.utils.rnn.pad_sequence(targets, padding_value=tokenizer.padding_idx_target, batch_first=True)
        context = torch.cat(context)

        targets = targets.to(device)
        context = context.to(device)

        optimizer.zero_grad()

        for j in range(0, targets.shape[1] - 1):
            inputs = targets[:, j:j+1]
            labels = targets[:, j+1:j+2]

            prediction = model(context, inputs)

            n_total_loss_values += torch.sum(torch.where(labels != tokenizer.padding_idx_target, 1, 0))
            labels = labels.reshape(labels.shape[0] * labels.shape[1])  # B * T
            total_loss += criterion(prediction, labels)

        total_loss /= n_total_loss_values
        total_loss.backward()
        optimizer.step()
        print(f"Batch {i+1}/{len(dataloader)} - Loss: {total_loss.item():.4f}")

    return total_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    total_loss_values = 0

    for i, batch in enumerate(dataloader):
        context = batch["overview"]
        targets = batch["report_short"]

        for j in range(len(context)):
            context[j] = torch.tensor(tokenizer.stoi_context(context[j])).unsqueeze(0)
            targets[j] = torch.tensor(tokenizer.stoi_targets("<start> " + targets[j] + " <stop>"))

        targets = nn.utils.rnn.pad_sequence(targets, padding_value=tokenizer.padding_idx_target, batch_first=True)
        context = torch.cat(context)

        targets = targets.to(device)
        context = context.to(device)

        for j in range(0, targets.shape[1] - 1):
            inputs = targets[:, j:j+1]
            labels = targets[:, j+1:j+2]

            prediction = model(context, inputs)

            total_loss_values += torch.sum(torch.where(labels != tokenizer.padding_idx_target, 1, 0))
            labels = labels.reshape(labels.shape[0] * labels.shape[1])
            total_loss += criterion(prediction, labels)

    return total_loss / total_loss_values

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