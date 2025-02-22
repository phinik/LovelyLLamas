import torch 

from typing import Dict


class OverviewSelector:
    def __init__(self):
        pass
        
    @staticmethod
    def select(overview_type: str):
        assert overview_type in ["full", "ctpc", "ctc", "ct", "tpwc"], f"Unknown overview type {overview_type}"

        return f"overview_{overview_type}"

class TargetSelector:
    def __init__(self):
        pass
        
    @staticmethod
    def select(target_type: str):
        assert target_type in ["gpt", "default"], f"Unknown overview type {target_type}"

        if target_type == "gpt":
            return "gpt_rewritten_cleaned"
        elif target_type == "default":
            return "report_short_wout_boeen"
        else:
            raise KeyError(f"Unknown overview type {target_type}")
        

def batchify(context: torch.tensor, targets: torch.tensor, block_size: int, device: torch.device) -> Dict:
        input_seqs = []
        label_seqs = []
        context_seqs = []
        
        # Create the maximum amount of sequences from the data. Each sequence has length 'block_size' and sequences
        # are shifted by one token each.
        for j in range(0, targets.shape[1] - block_size):
            input_seqs.append(targets[:, j:j+block_size])
            label_seqs.append(targets[:, j+1:j+1+block_size])
            context_seqs.append(context)

        #print(targets[:, ::40].shape)
        # Get tensors from the list of sequences
        context = torch.concat(context_seqs)
        inputs = torch.concat(input_seqs)
        labels = torch.concat(label_seqs)

        max_batch_size = 512
        n_batches = context.shape[0] // max_batch_size + 1
        actual_batch_size = context.shape[0] // n_batches
        
        # Permute sequences randomly
        perm = torch.randperm(context.shape[0])
        context_perm = context[perm, ...]
        inputs_perm = inputs[perm, ...]
        labels_perm = labels[perm, ...]

        context_perm = context_perm.to(device)
        inputs_perm = inputs_perm.to(device)
        labels_perm = labels_perm.to(device)
            
        # Create batches of size 'actual_batch_size' from the permuted tensor of sequences
        batched_contexts = []
        batched_inputs = []
        batched_labels = []
        for i in range(n_batches-1):
            batched_contexts.append(context_perm[i*actual_batch_size : (i+1)*actual_batch_size, ...])
            batched_inputs.append(inputs_perm[i*actual_batch_size : (i+1)*actual_batch_size, ...])
            batched_labels.append(labels_perm[i*actual_batch_size : (i+1)*actual_batch_size, ...])

        # If context.shape[0] is not divisible by n_batches, the last batch will have size 'actual_batch_size' + 1
        batched_contexts.append(context_perm[(n_batches-1)*actual_batch_size : , ...])
        batched_inputs.append(inputs_perm[(n_batches-1)*actual_batch_size : , ...])
        batched_labels.append(labels_perm[(n_batches-1)*actual_batch_size : , ...])

        batch = {
            "context": batched_contexts,
            "inputs": batched_inputs,
            "labels": batched_labels
        }
        
        return batch


def batchify_classifier(
          context: torch.tensor, 
          targets_class_0: torch.tensor, 
          targets_class_1: torch.tensor, 
          block_size: int, 
          pad_idx: int, 
          device: torch.device
        ) -> Dict:
        
        input_seqs = []
        label_seqs = []
        context_seqs = []
        
        # Create the maximum amount of sequences from the data. Each sequence has length 'block_size' and sequences
        # are shifted by one token each.
        for j in range(0, targets_class_0.shape[1] - block_size):
            input_seqs.append(targets_class_0[:, j:j+block_size])
            label_seqs.append(torch.zeros((targets_class_0.shape[0], 1)))
            context_seqs.append(context)

        for j in range(0, targets_class_1.shape[1] - block_size):
            input_seqs.append(targets_class_1[:, j:j+block_size])
            label_seqs.append(torch.ones((targets_class_1.shape[0], 1)))
            context_seqs.append(context)

        # Get tensors from the list of sequences
        context = torch.concat(context_seqs)
        inputs = torch.concat(input_seqs)
        labels = torch.concat(label_seqs)

        # Remove sequences that only contain the padding idx
        mask = torch.any(inputs != pad_idx, 1)
        context = context[mask]
        labels = labels[mask]
        inputs = inputs[mask]

        # Calcualte batch values
        max_batch_size = 512
        n_batches = context.shape[0] // max_batch_size + 1
        actual_batch_size = context.shape[0] // n_batches
        
        # Permute sequences randomly
        perm = torch.randperm(context.shape[0])
        context_perm = context[perm, ...]
        inputs_perm = inputs[perm, ...]
        labels_perm = labels[perm, ...]

        context_perm = context_perm.to(device)
        inputs_perm = inputs_perm.to(device)
        labels_perm = labels_perm.to(device)
            
        # Create batches of size 'actual_batch_size' from the permuted tensor of sequences
        batched_contexts = []
        batched_inputs = []
        batched_labels = []
        for i in range(n_batches-1):
            batched_contexts.append(context_perm[i*actual_batch_size : (i+1)*actual_batch_size, ...])
            batched_inputs.append(inputs_perm[i*actual_batch_size : (i+1)*actual_batch_size, ...])
            batched_labels.append(labels_perm[i*actual_batch_size : (i+1)*actual_batch_size, ...])

        # If context.shape[0] is not divisible by n_batches, the last batch will have size 'actual_batch_size' + 1
        batched_contexts.append(context_perm[(n_batches-1)*actual_batch_size : , ...])
        batched_inputs.append(inputs_perm[(n_batches-1)*actual_batch_size : , ...])
        batched_labels.append(labels_perm[(n_batches-1)*actual_batch_size : , ...])

        batch = {
            "context": batched_contexts,
            "inputs": batched_inputs,
            "labels": batched_labels
        }
        
        return batch
