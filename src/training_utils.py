import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def prepare_batch(batch, tokenizer, device):
    """
    Prepares tokenized context and target data for training or evaluation.
    """
    context = batch["overview"]
    targets = batch["report_short"]

    tokenized_context = []
    tokenized_targets = []

    for i in range(len(context)):
        tokenized_ctx = torch.tensor(tokenizer.encode(context[i]))
        tokenized_tgt = torch.tensor(tokenizer.encode("<start> " + targets[i] + " <stop>"))

        tokenized_context.append(tokenized_ctx)
        tokenized_targets.append(tokenized_tgt)

    padded_context = nn.utils.rnn.pad_sequence(tokenized_context, padding_value=tokenizer.padding_idx, batch_first=True)
    padded_targets = nn.utils.rnn.pad_sequence(tokenized_targets, padding_value=tokenizer.padding_idx, batch_first=True)

    return padded_context.to(device), padded_targets.to(device)

def train(model, dataloader, tokenizer, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        context, targets = prepare_batch(batch, tokenizer, device)

        # Reset hidden state for each batch
        hidden_state = None

        optimizer.zero_grad()
        loss = 0

        for t in range(targets.shape[1] - 1):
            inputs = targets[:, t:t+1]
            labels = targets[:, t+1:t+2]

            predictions, hidden_state = model(context, inputs, hidden_state) # use reset hidden state
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach()) # detach hidden state from computation graph

            loss += criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, tokenizer, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            context, targets = prepare_batch(batch, tokenizer, device)
            loss = 0

            for t in range(targets.shape[1] - 1):
                inputs = targets[:, t:t+1]
                labels = targets[:, t+1:t+2]

                predictions = model(context, inputs)
                loss += criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))

            total_loss += loss.item()

    return total_loss / len(dataloader)