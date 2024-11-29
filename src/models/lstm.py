import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM model for generating weather reports."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, context, inputs, hidden_state=None):
        embedded_inputs = self.embedding(inputs)

        lstm_out, hidden_state = self.lstm(embedded_inputs, hidden_state)
        output = self.fc(lstm_out)
        return output, hidden_state
    
    def num_parameters(self):
        """
        Returns the number of trainable parameters in the model.
        :return: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)