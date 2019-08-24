import torch as torch
import torch.nn as nn
import torch.nn.functional as functional

import glob
import numpy as np
import tqdm
from os import path

class PitM12Model(nn.Module):
    r"""
     Fill up
    """
    def __init__(self, seq_n_features, lstm_size, lstm_layers, lstm_dropout, 
                    embed_dims):
        """
        Parameters:
            seq_n_features: number of features inside sequence. note that first
            8 (0+7) represent one hot encoding for DLQ
            lstm_size: the size of hidden unit in LSTM 
            linear_size: the number of of output units for the first linear
            embed_dims: the list of tuples, where the first element of tuple
                represents the number of categories and second element of the
                tuple if the embedding size
            embed_drp: drop out percentage after embedding layer
        """
        super(PitM12Model, self).__init__()
        self.embed_layers = nn.ModuleList(\
            [nn.Embedding(c, d) for c, d in embed_dims])

        # total dim from embeddings
        self.acq_bn_dim = sum(x[1] for x in embed_dims)
        self.embedding_bn = nn.BatchNorm1d(self.acq_bn_dim)
        lstm_input_dim = seq_n_features + self.acq_bn_dim
        self.lstm = nn.LSTM(
            input_size  = lstm_input_dim,
            hidden_size = lstm_size,
            num_layers  = lstm_layers,
            dropout     = lstm_dropout,
            batch_first  = True
        )

        self.linear1 = nn.Linear(lstm_size, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.dpout1 = nn.Dropout(0.3)

        self.linear2 = nn.Linear(100, 250)
        self.bn2 = nn.BatchNorm1d(250)
        self.dpout2 = nn.Dropout(0.3)
        
        self.linear3 = nn.Linear(250, 19*12)
        self.bn3 = nn.BatchNorm1d(19*12)
        #self.dpout1 = nn.Dropout(0.1)

        #self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, seq_padded, input_lengths, acq):
        # embed acq
        acq = acq.long()
        e = [embed(acq[:,i]) for i, embed in enumerate(self.embed_layers)]
        e = torch.cat(e,  1)
        e = functional.relu(e)
        e = self.embedding_bn(e)
        # reshape and replicate e to attach to each time index
        e = e.reshape(seq_padded.shape[0], -1, e.shape[1])\
            .expand(-1, seq_padded.shape[1], -1)
        # glue embeddings to each time index of the seq
        s = torch.cat([seq_padded, e], 2)

        # process sequence
        self.lstm.flatten_parameters()
        #total_length = seq_padded.size(1)
        packed_input = pack_padded_sequence(s, input_lengths, 
                                            batch_first=True)
        packed_output, (ht, _)  = self.lstm(packed_input)

        # ht is the final element of the sequence shape (hidden_size, 1)
        # we are predicting 19 DLQ x 12 months
        out = ht[-1, :, :]
        out = self.bn1(functional.relu(self.linear1(out)))
        out = self.dpout1(out)

        out = self.bn2(functional.relu(self.linear2(out)))
        out = self.dpout2(out)

        out = self.bn3(functional.relu(self.linear3(out)))
        #out = self.dpout2(out)

        out = out.reshape(-1, 19, 12) 
        #out = self.logsoftmax(out)
        return out

def makeEmeddings(acq, embeding_dim=2):
    # add 1 since NULLs are extracted as 0
    acq_cat = np.max(acq[:, 1:], axis=0) + 1
    emb = [(acq_cat[i], embeding_dim) for i in range(acq_cat.shape[0])]
    return emb

class TrainingContext:
    def __init__(self, model_path):
        self.train_losses = []
        self.valid_losses = []
        self.model_path = model_path

        # model path does not exists then make it and save it

    def saveModel(self):
        if path.exists(MODEL_SAVED):
            print('Loading model checkpoint: {}'.format(MODEL_SAVED))
            checkpoint = torch.load(MODEL_SAVED)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            checkpoint_epoch = checkpoint['epoch']
            train_losses = checkpoint['train_losses']
            valid_losses = checkpoint['valid_losses']


if __name__ == "__main__":
    TRAIN_PATH = '/home/user/notebooks/data/train'
    VALID_PATH = '/home/user/notebooks/data/valid'
    MODEL_PATH = '/home/user/notebooks/data/model/baselS12'
    SUMMARY_PATH = '/home/user/notebooks/runs/baselS12'
    MODEL_SAVED = MODEL_PATH + '/baselS12.pth'

    if DEBUG:
        BATCH_SIZE = 4
        NUM_WORKERS = 0 
    else:
        BATCH_SIZE = 512 #12*2**10
        NUM_WORKERS = 16
    #BATCH_STOP = 100
    NUM_EPOCHS = 90
    LOSS_EVERY_N_BATCHES=200
    SAVE_EVERY_N_BATCHES=1000

    train_acq, train_idx_to_seq, train_seq = load_data(TRAIN_PATH)
    valid_acq, valid_idx_to_seq, valid_seq = load_data(VALID_PATH)

    train_ds = FNMDatasetS12(train_acq, train_idx_to_seq, train_seq, 10)
    valid_ds = FNMDatasetS12(valid_acq, valid_idx_to_seq, valid_seq, 10)

    print("Number of train acq: {:,}".format(len(train_ds)))
    print("Number of valid acq: {:,}".format(len(valid_ds)))
    
    trainDL = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)
    validDL = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle = True, \
        collate_fn=paddingCollator, num_workers=NUM_WORKERS)

    if DEBUG:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    acq_embs = makeEmeddings(train_acq, embeding_dim=2)

    model = BaselS12Model(\
        seq_n_features=22, lstm_size=100, lstm_layers=3, lstm_dropout=0.3, 
            embed_dims = acq_embs)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(params=model.parameters())
    checkpoint_epoch = 0
    train_losses = []
    valid_losses = []

    writer = SummaryWriter(SUMMARY_PATH, comment='fixed_seq_ind_2')
   #  dataiter = iter(trainDL)
   #  seq, seq_len, acq, target_dlq = dataiter.next()
   #  writer.add_graph(model, (seq, seq_len, acq), False)
   #  writer.flush()
    
    model = model.to(device)
    loss_function = loss_function.to(device)

    if path.exists(MODEL_SAVED):
        print('Loading model checkpoint: {}'.format(MODEL_SAVED))
        checkpoint = torch.load(MODEL_SAVED)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']

    if not DEBUG:
        if torch.cuda.device_count() > 1:
            print("Training on", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model)

    training_iter = 0
    validation_iter = 0

    try:
        with tqdm.trange(checkpoint_epoch, checkpoint_epoch + NUM_EPOCHS) as t:
            for epoch in t:
                train_losses_epoch = []
                t.set_description('Epoch: %i' % epoch)
                model.train()
                tqdm_inner = tqdm.tqdm(trainDL)
                for bidx, (seq, seq_len, acq, target) in enumerate(tqdm_inner):
                    tqdm_inner.set_description('Train: %i' % bidx)
                    model.zero_grad()
                    seq = seq.to(device)
                    acq = acq.to(device)
                    seq_len = seq_len.to(device)
                    target = target.to(device)

                    target_hat = model(seq, seq_len, acq)
                    loss = loss_function(target_hat, target)
                    train_losses_epoch.append(loss.item())
                    loss.backward()
                    optimizer.step()

                    if bidx % LOSS_EVERY_N_BATCHES  == 0:
                        mean_loss = np.sum(train_losses_epoch)/len(train_losses_epoch)/BATCH_SIZE
                        tqdm_inner.set_postfix(trainLoss = "{:.12f}".format(mean_loss))
                        if not DEBUG:
                            writer.add_scalar('loss/training', mean_loss, training_iter)

                    if bidx % SAVE_EVERY_N_BATCHES == SAVE_EVERY_N_BATCHES - 1:
                        if not DEBUG:
                            torch.save({ \
                                'epoch': epoch,
                                'model_state_dict': model.module.state_dict(), \
                                'optimizer_state_dict': optimizer.state_dict(), \
                                'train_losses' : train_losses, \
                                'valid_losses' : valid_losses \
                                }, MODEL_SAVED \
                            )
                    training_iter += 1

                model.eval()
                with torch.no_grad():
                    valid_losses_epoch = []
                    tqdm_inner = tqdm.tqdm(validDL)
                    for bidx, (seq, seq_len, acq, target) in enumerate(tqdm_inner):
                        tqdm_inner.set_description('Valid: %i' % bidx)
                        seq = seq.to(device)
                        acq = acq.to(device)
                        seq_len = seq_len.to(device)
                        target = target.to(device)
                        target_hat = model(seq, seq_len, acq)
                        loss = loss_function(target_hat, target)
                        valid_losses_epoch.append(loss.item())

                        if bidx % LOSS_EVERY_N_BATCHES  == 0:
                            mean_loss = np.sum(valid_losses_epoch)/len(valid_losses_epoch)/BATCH_SIZE
                            tqdm_inner.set_postfix( \
                                validLoss = "{:.12f}".format(mean_loss))
                            if not DEBUG:
                                writer.add_scalar(\
                                        'loss/validation',
                                        mean_loss, validation_iter
                                    )
                        validation_iter += 1

                t.set_postfix(\
                    trainLoss = "{:.12f}".format(np.sum(train_losses_epoch)/len(train_losses_epoch)/BATCH_SIZE), \
                    validLoss = "{:.12f}".format(np.sum(valid_losses_epoch)/len(valid_losses_epoch)/BATCH_SIZE))

                # save losses and model
                if not DEBUG:
                    torch.save({ \
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(), \
                        'optimizer_state_dict': optimizer.state_dict(), \
                        'train_losses' : train_losses, \
                        'valid_losses' : valid_losses \
                        }, MODEL_SAVED \
                    )

                train_losses.append(np.sum(train_losses_epoch)/len(train_losses_epoch)/BATCH_SIZE)
                valid_losses.append(np.sum(valid_losses_epoch)/len(valid_losses_epoch)/BATCH_SIZE)

    except KeyboardInterrupt:
        print ('Saving the model state before exiting')
        if not DEBUG:
            torch.save({ \
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(), \
                'optimizer_state_dict': optimizer.state_dict(), \
                'train_losses' : train_losses, \
                'valid_losses' : valid_losses \
                }, MODEL_SAVED \
            )
    except RuntimeError as e:
        print(e)
        print("####### bad: {}, {}, {}".format(bad_idx, bad_chunk, bad_id_in_chunk))
    writer.close()
