import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # Fortschrittsanzeige

def pad_to_max_length(tensor, max_length, padding_idx):
    padding_size = max_length - tensor.size(1)
    if padding_size > 0:
        padding = torch.full((tensor.size(0), padding_size), padding_idx, dtype=torch.long, device=tensor.device)
        return torch.cat([tensor, padding], dim=1)
    else:
        return tensor

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
    tokenized_context = [torch.tensor(tokenizer.encode(text)) for text in batch["overview"]]
    tokenized_targets = [torch.tensor(tokenizer.encode("<start> " + text + " <stop>")) for text in batch["report_short"]]

    context_lengths = [len(seq) for seq in tokenized_context]
    target_lengths = [len(seq) for seq in tokenized_targets]

    data = sorted(zip(tokenized_context, tokenized_targets, context_lengths, target_lengths), key=lambda x: -x[2])
    sorted_context, sorted_targets, sorted_context_lengths, sorted_target_lengths = zip(*data)

    padded_context = nn.utils.rnn.pad_sequence(sorted_context, batch_first=True, padding_value=tokenizer.padding_idx)
    packed_context = nn.utils.rnn.pack_padded_sequence(padded_context, sorted_context_lengths, batch_first=True, enforce_sorted=True)

    padded_targets = nn.utils.rnn.pad_sequence(sorted_targets, batch_first=True, padding_value=tokenizer.padding_idx)
    packed_targets = nn.utils.rnn.pack_padded_sequence(padded_targets, sorted_target_lengths, batch_first=True, enforce_sorted=False)
    
    return packed_context.to(device), packed_targets.to(device), target_lengths

def train(model, dataloader, tokenizer, optimizer, criterion, device, gradient_clip=1.0):
    model.train()
    total_loss = 0

    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for batch in dataloader:
            packed_context, packed_targets, target_lengths = prepare_batch(batch, tokenizer, device)

            optimizer.zero_grad()

            loss = 0

            predictions = model(packed_context, packed_targets) # predictions = [batch_size, max_seq_length, vocab_size]
            targets = packed_targets.data

            batch_size, max_seq_length, vocab_size = predictions.size()
            flattened_predictions = predictions.reshape(-1, vocab_size) # [batch_size * max_seq_length, vocab_size]

            padded_targets = torch.full((batch_size, max_seq_length), tokenizer.padding_idx, dtype=torch.long, device=device)
            for i, length in enumerate(target_lengths):
                padded_targets[i, :length] = targets[:length]
            flattened_targets = padded_targets.view(-1)

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
                packed_context, packed_targets, target_lengths = prepare_batch(batch, tokenizer, device)

                loss = 0

                predictions = model(packed_context, packed_targets) # predictions = [batch_size, max_seq_length, vocab_size]
                targets = packed_targets.data
                
                batch_size, max_seq_length, vocab_size = predictions.size()
                flattened_predictions = predictions.reshape(-1, vocab_size) # [batch_size * max_seq_length, vocab_size]

                padded_targets = torch.full((batch_size, max_seq_length), tokenizer.padding_idx, dtype=torch.long, device=device)
                for i, length in enumerate(target_lengths):
                    padded_targets[i, :length] = targets[:length]
                flattened_targets = padded_targets.view(-1)

                loss = criterion(flattened_predictions, flattened_targets)
                total_loss += loss.item()

                # progress indicator
                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.update(1)

    return total_loss / len(dataloader)