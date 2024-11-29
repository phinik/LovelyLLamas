import torch
import torch.optim as optim
import torch.nn as nn
import csv

from dataloader import get_train_dataloader_weather_dataset, get_eval_dataloader_weather_dataset
from tokenizer import Tokenizer
from models.lstm import LSTM
from training_utils import train, evaluate

# Paths and parameters
DATASET_PATH = 'C:/Users/Agando/Desktop/Uni/Master-Projekt/dataset2'
#DATASET_PATH = 'C:/Users/Niels/Desktop/Uni/WS25/Master-Projekt/debug_dataset'
BATCH_SIZE = 64
EMBEDDING_DIM = 64 #128
HIDDEN_DIM = 128 #256
LEARNING_RATE = 0.005
NUM_EPOCHS = 18
GRADIENT_CLIP = 1.0
MODEL_PATH = "weather_lstm.pth"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer
tokenizer = Tokenizer(DATASET_PATH)
tokenizer.add_custom_tokens(['<start>', '<stop>', '<degC>', '<city>', '<pad>'])

# Create DataLoaders
train_dataloader = get_train_dataloader_weather_dataset(DATASET_PATH, BATCH_SIZE, cached=True)
eval_dataloader = get_eval_dataloader_weather_dataset(DATASET_PATH, BATCH_SIZE, cached=True)

# Initialize model
vocab_size = tokenizer.vocab_size
output_dim = vocab_size
model = LSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=tokenizer.padding_idx)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Load weights if available
try:
    model.load_state_dict(torch.load("checkpoint.pth"))
    print("Loaded model weights.")
except FileNotFoundError:
    print("No model weights found.")

# Training Log Header
log_file = "training_log.csv"
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Eval Loss"])  # Header

best_loss = float('inf')

# Training Loop
print(f"Training started on model with {NUM_EPOCHS} epochs.")
print(f"Parameters: {model.num_parameters()}")
for epoch in range(NUM_EPOCHS):
    train_loss = train(model, train_dataloader, tokenizer, optimizer, criterion, device, gradient_clip=1.0)
    eval_loss = evaluate(model, eval_dataloader, tokenizer, criterion, device)

    scheduler.step(eval_loss)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Eval Loss: {eval_loss:.4f}")

    # Save the best model
    if eval_loss < best_loss:
        best_loss = eval_loss
        torch.save(model.state_dict(), f"best_model_loss.pth")
        print("Saved new best model (loss).")

    # Log results
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, eval_loss])