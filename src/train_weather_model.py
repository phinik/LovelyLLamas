import torch
import torch.optim as optim
import torch.nn as nn

from dataloader import get_train_dataloader_weather_dataset, get_eval_dataloader_weather_dataset
from tokenizer import Tokenizer
from models.lstm import LSTM
from training_utils import train, evaluate

# Paths and parameters
DATASET_PATH = 'C:/Users/Agando/Desktop/Uni/Master-Projekt/debug_dataset'
BATCH_SIZE = 32
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
MODEL_PATH = "weather_lstm.pth"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer
tokenizer = Tokenizer(DATASET_PATH)
tokenizer.add_custom_tokens(['<start>', '<stop>', '<degC>, <city>'])

# Create DataLoaders
train_dataloader = get_train_dataloader_weather_dataset(DATASET_PATH, BATCH_SIZE, cached=True)
eval_dataloader = get_eval_dataloader_weather_dataset(DATASET_PATH, BATCH_SIZE, cached=True)

# Initialize model
vocab_size = tokenizer.vocab_size
output_dim = vocab_size
model = LSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training Loop
print("Training started...")
for epoch in range(NUM_EPOCHS):
    train_loss = train(model, train_dataloader, tokenizer, optimizer, criterion, device)
    eval_loss = evaluate(model, eval_dataloader, tokenizer, criterion, device)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Eval Loss: {eval_loss:.4f}")

# Save Model
torch.save(model.state_dict(), MODEL_PATH)
print("Training complete. Model saved.")