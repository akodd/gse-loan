import torch as torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BaselM1Model(nn.Module):
    r"""
    Very simple model where we don't use any embedding or upb rescaling.
    We expect 4 features
    """
    def __init__(self, n_features, lstm_size, linear_size):
        super(BaselM1Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size= n_features,
            hidden_size=lstm_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        self.linear = nn.Linear(lstm_size, linear_size)
        self.linear_out = nn.Linear(linear_size, 1)


    def forward(self, seq_padded, acq, input_lengths, default_1y):
        #self.lstm.flatten_parameters()
        total_length = seq_padded.size(1)
        packed_input = pack_padded_sequence(seq_padded, input_lengths, 
                                            batch_first=True)
        packed_output, _  = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                total_length=total_length)
        #default_1y_hat = torch.zeros((output.size(0), total_length))
        default_1y_hat = torch.zeros_like(default_1y)
        for t in range(total_length):
            l_output = relu(self.linear(output[:, t, :]))
            default_1y_hat[:, t] = self.linear_out(l_output).view(-1)
        default_1y_hat = torch.sigmoid(default_1y_hat)
        return default_1y_hat