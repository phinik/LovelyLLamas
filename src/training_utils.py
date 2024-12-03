import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # Fortschrittsanzeige

def unpack_packed_sequence(packed_sequence, padding_idx, device):
    """
    Converts a PackedSequence back to a padded tensor for loss calculation.
    """
    target_data = packed_sequence.data
    batch_sizes = packed_sequence.batch_sizes

    num_sequences = batch_sizes[0].item()
    sequence_lengths = torch.zeros(num_sequences, dtype=torch.long, device=device)

    for i, batch_size in enumerate(batch_sizes):
        sequence_lengths[:batch_size] += 1
    max_seq_length = sequence_lengths.max().item()
    padded_targets = torch.full((num_sequences, max_seq_length), padding_idx, dtype=torch.long, device=device)

    data_idx = 0
    for timestep, batch_size in enumerate(batch_sizes):
        for seq_idx in range(batch_size):
            if data_idx >= len(target_data):
                break
            padded_targets[seq_idx, timestep] = target_data[data_idx]
            data_idx += 1

    flattened_targets = padded_targets.view(-1)
    return flattened_targets
    
def prepare_batch(batch, tokenizer, device, max_length=512):
    """
    Prepares tokenized context and target data for training or evaluation using PackedSequence.
    Handles sequences longer than max_length by splitting into chunks.
    """
    context = batch["overview"]
    targets = batch["report_short"]

    tokenized_context = []
    tokenized_targets = []
    context_lengths = []

    for i in range(len(context)):
        tokenized_ctx = torch.tensor(tokenizer.encode(context[i]))
        tokenized_tgt = torch.tensor(tokenizer.encode("<start> " + targets[i] + " <stop>"))

        tokenized_context.append(tokenized_ctx)
        tokenized_targets.append(tokenized_tgt)
        context_lengths.append(len(tokenized_ctx))

    data = list(zip(tokenized_context, tokenized_targets, context_lengths))
    data.sort(key=lambda x: -x[2])

    sorted_context, sorted_targets, sorted_context_lengths = zip(*data)

    padded_context = nn.utils.rnn.pad_sequence(sorted_context, batch_first=True)
    packed_context = nn.utils.rnn.pack_padded_sequence(padded_context, sorted_context_lengths, batch_first=True, enforce_sorted=True)

    padded_targets = nn.utils.rnn.pad_sequence(sorted_targets, batch_first=True)
    packed_targets = nn.utils.rnn.pack_padded_sequence(padded_targets, sorted_context_lengths, batch_first=True, enforce_sorted=True)
    
    return packed_context.to(device), packed_targets.to(device)

def train(model, dataloader, tokenizer, optimizer, criterion, device, gradient_clip=1.0):
    model.train()
    total_loss = 0

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for batch in dataloader:
            packed_context, packed_targets = prepare_batch(batch, tokenizer, device)

            optimizer.zero_grad()

            loss = 0

            predictions = model(packed_context, packed_targets) # predictions = [batch_size, max_seq_length, vocab_size]
            flattened_predictions = predictions.view(-1, predictions.shape[2]) # [batch_size * max_seq_length, vocab_size]
            flattened_targets = unpack_packed_sequence(packed_targets, tokenizer.padding_idx, device=device)

            loss = criterion(flattened_predictions, flattened_targets)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            total_loss += loss.item()

            # progress indicator
            pbar.set_postfix({"Batch Loss": loss.item()})
            pbar.update(1)

    return total_loss / len(dataloader)

def evaluate(model, dataloader, tokenizer, criterion, device):
    model.eval()
    total_loss = 0

    with tqdm(total=len(dataloader), desc="Evaluating", unit="batch") as pbar:
        with torch.no_grad():
            for batch in dataloader:
                packed_context, packed_targets = prepare_batch(batch, tokenizer, device)

                loss = 0

                predictions = model(packed_context, packed_targets) # predictions = [batch_size, max_seq_length, vocab_size]
                flattened_predictions = predictions.view(-1, predictions.shape[2]) # [batch_size * max_seq_length, vocab_size]
                flattened_targets = unpack_packed_sequence(packed_targets, tokenizer.padding_idx, device=device)

                loss = criterion(flattened_predictions, flattened_targets)
                total_loss += loss.item()

                # progress indicator
                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.update(1)

    return total_loss / len(dataloader)