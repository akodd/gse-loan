
import torch as torch
import torch.nn as nn
import collections

from torch.nn.utils.rnn import pack_padded_sequence

class LinearBlock(nn.Module):
    def __init__(self, input_size, linear_conf):
        super(LinearBlock, self).__init__()
        dc = collections.OrderedDict()
        prev_s = input_size
        for name, (s, r) in linear_conf.items():
            dc['linear'+name]=nn.Linear(prev_s, s)
            dc['tanh'  +name] = nn.Tanh()
            dc['batchn'+name] = nn.BatchNorm1d(s)
            dc['drpout'+name] = nn.Dropout(r)
            prev_s = s
        self.linblock = nn.Sequential(dc)

    def forward(self, x):
        return self.linblock(x)

    def out_features(self):
        return self.linblock[-4].out_features

class LSTMEmbeddingEncoder(nn.Module):
    def __init__(self, seq_n_features, lstm_conf, emb_acq_dims, emb_seq_dims):
        super(LSTMEmbeddingEncoder, self).__init__()
        emb_acq_dict = collections.OrderedDict()
        emb_acq_dim_sum = 0
        for name, (c, d) in emb_acq_dims.items():
            emb_acq_dict[name] = nn.Embedding(c, d)
            emb_acq_dim_sum += d
        self.emb_acq = nn.ModuleDict(emb_acq_dict)

        emb_seq_dict = collections.OrderedDict()
        emb_seq_dim_sum = 0
        for name, (c, d) in emb_seq_dims.items():
            emb_seq_dict[name] = nn.Embedding(c, d)
            emb_seq_dim_sum += d
        self.emb_seq = nn.ModuleDict(emb_seq_dict)
       
        emb_dim_total = emb_acq_dim_sum + emb_seq_dim_sum

        lstm_size = lstm_conf['lstm_size']
        lstm_layers = lstm_conf['lstm_layers']
        lstm_dropout = lstm_conf['lstm_dropout']

        self.lstm = nn.LSTM(    
            input_size  = seq_n_features + emb_dim_total,
            hidden_size = lstm_size,
            num_layers  = lstm_layers,
            dropout     = lstm_dropout,
            batch_first  = True,
            bidirectional = True
        )

        #self.linblock = LinearBlock(self.lstm.hidden_size, linear_conf)

    def forward(self, seq, seq_len, ymd, acq):
        ea = [x(acq[:,i]) for i, (a, x) in enumerate(self.emb_acq.items())]
        ea = torch.cat(ea,  1)
        ea = ea.reshape(seq.shape[0], -1, ea.shape[1])\
            .expand(-1, seq.shape[1], -1)   

        ey = [x(ymd[:,:,i]) for i, (a, x) in enumerate(self.emb_seq.items())]
        ey = torch.cat(ey,  2)

        s = torch.cat([seq, ea, ey], 2)

        self.lstm.flatten_parameters()
        packed_input = pack_padded_sequence(s, seq_len, batch_first=True)
        _, (ht, _)  = self.lstm(packed_input)

        # move batch from dim 1 to 0
        #out = ht.permute(1, 0, 2).view(-1, 
        #    2 * self.lstm.num_layers * self.lstm.hidden_size)

        out = ht.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
        out = out[-1, 0, :, :] + out[-1, 1, :, :]

        return out

def lastDLQ(seq, seq_len, dlq_dim=9):
    batch_idx = torch.arange(seq.shape[0], device=seq.device)
    dlq = seq[batch_idx, seq_len.long()-1, -dlq_dim:]
    return dlq

