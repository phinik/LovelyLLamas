import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # Progress bar for training and evaluation

def pad_to_max_length(tensor, max_length, padding_idx):
    """
    Pads a tensor to the specified max_length with padding_idx.
    """
    padding_size = max_length - tensor.size(1)
    if padding_size > 0:
        padding = torch.full(
            (tensor.size(0), padding_size), 
            padding_idx, dtype=torch.long, device=tensor.device
        )
        return torch.cat([tensor, padding], dim=1)
    else:
        return tensor

def prepare_batch(batch, tokenizer, device):
    """
    Prepares tokenized context and target data for training or evaluation using PackedSequence.
    Handles sorting, padding, and packing.
    """
    tokenized_context = [torch.tensor(tokenizer.encode(text)) for text in batch["overview"]]
    tokenized_targets = [torch.tensor(tokenizer.encode("<start> " + text + " <stop>")) for text in batch["report_short"]]

    context_lengths = [len(seq) for seq in tokenized_context]
    target_lengths = [len(seq) for seq in tokenized_targets]

    # Sort by context lengths for packing
    data = sorted(zip(tokenized_context, tokenized_targets, context_lengths, target_lengths), key=lambda x: -x[2])
    sorted_context, sorted_targets, sorted_context_lengths, sorted_target_lengths = zip(*data)

    # Pad and pack the context and targets
    padded_context = nn.utils.rnn.pad_sequence(sorted_context, batch_first=True, padding_value=tokenizer.padding_idx)
    packed_context = nn.utils.rnn.pack_padded_sequence(padded_context, sorted_context_lengths, batch_first=True, enforce_sorted=True)

    padded_targets = nn.utils.rnn.pad_sequence(sorted_targets, batch_first=True, padding_value=tokenizer.padding_idx)
    packed_targets = nn.utils.rnn.pack_padded_sequence(padded_targets, sorted_target_lengths, batch_first=True, enforce_sorted=False) # no ONNX support for enforce_sorted=False

    return packed_context.to(device), packed_targets.to(device), sorted_target_lengths

def train(model, dataloader, tokenizer, optimizer, criterion, device, gradient_clip=1.0, teacher_forcing_ratio=0.5):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for batch in dataloader:
            packed_context, packed_targets, target_lengths = prepare_batch(batch, tokenizer, device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(packed_context, packed_targets, teacher_forcing_ratio=teacher_forcing_ratio)  # predictions: [batch_size, max_seq_length, vocab_size]

            # Unpack targets
            padded_targets, _ = nn.utils.rnn.pad_packed_sequence(
                packed_targets, batch_first=True, padding_value=tokenizer.padding_idx
            )
            padded_targets = nn.functional.pad(padded_targets, (0, predictions.size(1) - padded_targets.size(1)), value=tokenizer.padding_idx)

            flattened_predictions = predictions.view(-1, predictions.size(-1))
            flattened_targets = padded_targets.view(-1)

            # Compute loss
            weights = torch.tensor([1.0 if t != tokenizer.padding_idx else 0.1 for t in flattened_targets], device=device)
            loss = (criterion(flattened_predictions, flattened_targets) * weights).mean()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({"Batch Loss": loss.item()})
            pbar.update(1)

    return total_loss / len(dataloader)

def evaluate(model, dataloader, tokenizer, criterion, device, teacher_forcing_ratio=0):
    """
    Evaluates the model on the validation dataset.
    """
    model.eval()
    total_loss = 0

    with tqdm(total=len(dataloader), desc="Evaluating", unit="batch") as pbar:
        with torch.no_grad():
            for batch in dataloader:
                packed_context, packed_targets, target_lengths = prepare_batch(batch, tokenizer, device)

                # Forward pass
                predictions = model(packed_context, packed_targets, teacher_forcing_ratio)  # predictions: [batch_size, max_seq_length, vocab_size]

                # Unpack targets
                padded_targets, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_targets, batch_first=True, padding_value=tokenizer.padding_idx
                )
                padded_targets = nn.functional.pad(padded_targets, (0, predictions.size(1) - padded_targets.size(1)), value=tokenizer.padding_idx)

                flattened_predictions = predictions.view(-1, predictions.size(-1))
                flattened_targets = padded_targets.view(-1)

                # Compute loss
                weights = torch.tensor([1.0 if t != tokenizer.padding_idx else 0.1 for t in flattened_targets], device=device)
                loss = (criterion(flattened_predictions, flattened_targets) * weights).mean()
                total_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.update(1)

    return total_loss / len(dataloader)
