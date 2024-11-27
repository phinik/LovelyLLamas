import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM model for generating weather reports."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, context, inputs):
        embedded_context = self.embedding(context)
        embedded_inputs = self.embedding(inputs)

        lstm_out, _ = self.lstm(embedded_inputs)
        output = self.fc(lstm_out)
        return output