import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM model for generating weather reports."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.initialize_weights()

    def forward(self, packed_context, packed_targets):
        """
        Forward pass for the LSTM model.

        Args:
            packed_context (PackedSequence): Packed input context.
            packed_targets (PackedSequence): Packed input targets.

        Returns:
            Tensor: Predictions for the input targets.
        """
        # Embedded inputs for PackedSequence
        embedded_context = nn.utils.rnn.PackedSequence(
            self.embedding(packed_context.data), 
            packed_context.batch_sizes,
            packed_context.sorted_indices,
            packed_context.unsorted_indices
        )

        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded_context) # expects a PackedSequence

        # Unpack the output for the Linear Layer
        padded_output, output_lengths = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        predictions = self.fc(padded_output)

        return predictions
    
    def initialize_weights(self):
        """
        Initialize model weights.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

                if "bias_ih" in name:
                    n = param.size(0)
                    param.data[n // 4: n // 2].fill_(1.0) # set forget gate bias to 1.0
    
    def num_parameters(self):
        """
        Returns the number of trainable parameters in the model.
        :return: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict(self, inputs):
        """
        Predict weather report for a given input without packed sequences.
        """
        embedded_inputs = self.embedding(inputs)
        lstm_out, _ = self.lstm(embedded_inputs)
        output = self.fc(lstm_out)
        return output