import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM model for generating weather reports."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2, bidirectional=False):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Linear(hidden_dim, output_dim)

        #self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_normal_(param.data)
            else:
                nn.init.constant_(param.data, 0)

    def forward(self, packed_context, packed_targets, teacher_forcing_ratio=0.5):
        """
        Forward pass for the LSTM model with teacher forcing.

        Args:
            packed_context (PackedSequence): Packed input context.
            packed_targets (PackedSequence): Packed input targets.
            teacher_forcing_ratio (float): Probability of using true target as next input during training.

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

        # Unpack targets for processing
        targets, targets_lengths = nn.utils.rnn.pad_packed_sequence(packed_targets, batch_first=True)
        targets = targets.long()

        # Initialize LSTM state
        batch_size, max_seq_len = targets.size()
        device = targets.device

        # Prepare input and hidden state for LSTM
        outputs = torch.zeros(batch_size, max_seq_len, self.fc.out_features, device=device)
        lstm_hidden = None  # initialize hidden state to None or provide initial hidden states if needed

        # Pass context through LSTM
        packed_output, lstm_hidden = self.lstm(embedded_context, lstm_hidden)

        # Unpack LSTM outputs to process step-by-step
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Process each timestep
        for t in range(max_seq_len):
            if t == 0 or torch.rand(1).item() < teacher_forcing_ratio:
                # Use the true target at timestep `t` (teacher forcing)
                input_step = targets[:, t]
            else:
                # Use the model's previous prediction
                input_step = predictions.argmax(dim=1)

            # Embed the input step
            embedded_input = self.embedding(input_step.long())
            embedded_input = embedded_input.unsqueeze(1)

            # Pass through LSTM one step
            lstm_output, lstm_hidden = self.lstm(embedded_input, lstm_hidden)

            # Generate prediction
            predictions = self.fc(lstm_output.squeeze(1))

            # Store predictions
            outputs[:, t] = predictions

        return outputs

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