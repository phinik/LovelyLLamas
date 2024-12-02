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

    def forward(self, packed_context, packed_targets, hidden_state=None):
        """
        Forward pass for the LSTM model.

        Args:
            packed_context (PackedSequence): Packed input context.
            packed_targets (PackedSequence): Packed input targets.
            hidden_state (tuple): Optional initial hidden state for the LSTM.

        Returns:
            PackedSequence: Predictions in packed sequence format.
            tuple: Updated hidden state.
        """
        # Embedded inputs for PackedSequence
        embedded_context = nn.utils.rnn.PackedSequence(
            self.embedding(packed_context.data), 
            packed_context.batch_sizes,
            packed_context.sorted_indices,
            packed_context.unsorted_indices
        )

        # LSTM forward pass
        lstm_out, hidden_state = self.lstm(embedded_context, hidden_state) # expects a PackedSequence

        # Unpack the output for the Linear Layer
        padded_output, output_lengths = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        predictions = self.fc(padded_output)

        return predictions, hidden_state
    
    def num_parameters(self):
        """
        Returns the number of trainable parameters in the model.
        :return: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict(self, inputs, hidden_state=None):
        """
        Predict weather report for a given input without packed sequences.
        """
        embedded_inputs = self.embedding(inputs)
        lstm_out, hidden_state = self.lstm(embedded_inputs, hidden_state)
        output = self.fc(lstm_out)
        return output, hidden_state