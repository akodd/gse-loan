import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import collections
import glob
import numpy as np
import tqdm
import os
import time
import datetime
from pytz import timezone
class TrainingContext:
    def __init__(self, PRINT_EVERY=25):
        self.train_losses = []
        self.valid_losses = []
        self.checkpoint_epoch = 0

        self.PRINT_EVERY = PRINT_EVERY
        
        self.device = torch.device("cpu")

    def _trainStep(self, batch):
        self.model.zero_grad()

        loss = self.applyModel(batch)

        loss_item = loss.item()
        loss.backward()
        self.optimizer.step()

        return loss_item

    def applyModel(self, batch):
        raise RuntimeError('not implemented')

    def trainLoop(self, epoch):
        self.model.train()
        tq = tqdm.tqdm(self.trainDL)
        losses = []
        for bidx, batch in enumerate(tq):
            tq.set_description('Train: %i' % bidx)
            
            loss_item = self._trainStep(batch)
            losses.append(loss_item)

            if bidx % self.PRINT_EVERY  == 0:
                mean_loss = np.mean(losses)/self.trainDL.batch_size
                tq.set_postfix(trainLoss = "{:.8f}".format(mean_loss))
                #writer.add_scalar('loss/training', mean_loss, epoch*bidx)

        mean_loss = np.mean(losses)/self.trainDL.batch_size
        self.train_losses.append(mean_loss)
        return mean_loss

    def validStep(self, batch):
        loss = self.applyModel(batch)
        return loss.item()

    def validLoop(self, epoch):
        self.model.eval()
        losses = []
        with torch.no_grad():
            tq = tqdm.tqdm(self.validDL)
            for bidx, batch in enumerate(tq):
                tq.set_description('Valid: %i' % bidx)
                
                loss_item = self.validStep(batch)
                losses.append(loss_item)

                if bidx % self.PRINT_EVERY  == 0:
                    mean_loss = np.mean(losses)/self.validDL.batch_size
                    tq.set_postfix(validLoss = "{:.8f}".format(mean_loss))
                    #writer.add_scalar('loss/validation', mean_loss, epoch*bidx)

        mean_loss = np.mean(losses)/self.validDL.batch_size
        self.valid_losses.append(mean_loss)
        return mean_loss
        

    def dataLoaderTrain(self):
        raise RuntimeError('not implemented')

    def dataLoaderValid(self):
        raise RuntimeError('not implemented')

    def loadModel(self, model_path):
        raise RuntimeError('not implemented')
 
    def saveModel(self, model_path, epoch):
        raise RuntimeError('not implemented')


    def useGPU(self, use=False, verbose=True):
        if use:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        if (verbose):
            print('Using device: {}'.format(self.device.type))
        self.model.to(self.device)
        self.loss_function.to(self.device)

    def makeParallel(self, use=False, verbose=True):
        if use and torch.cuda.device_count() > 1:
            if verbose:
                print("Training on", torch.cuda.device_count(), "GPUs")
            self.model = nn.DataParallel(self.model)

    def fit(self, NUM_EPOCHS, save_model=False):
        def lprint(x):
            return "|".join(map(lambda x: "{:.4f}".format(x), x))

        ch = self.checkpoint_epoch
        try:
            with tqdm.trange(ch, ch + NUM_EPOCHS) as t:
                for epoch in t:
                    t.set_description('Epoch: %i' % epoch)
                    train_loss = self.trainLoop(epoch)
                    valid_loss = self.validLoop(epoch)

                    t.set_postfix(
                        TL = "{:.4f}".format(train_loss),
                        MVL = "{:.4f}".format(np.mean(self.valid_losses[-10:])),
                        VL = lprint(self.valid_losses[-4:])
                    )

                    if len(self.valid_losses) > 10 \
                        and valid_loss > np.mean(self.valid_losses[-10:]):
                        print('Validation loss is increasing quit before saving')
                        break

                    if save_model:
                        self.saveModel(epoch, "")
        except KeyboardInterrupt:
            if save_model:
                print ('Saving the model state before exiting')
                self.saveModel(epoch, "")
            t.close()
        t.close()

        return self.valid_losses[-1]




def adjustModelPath(model_path, restart=True):
    if not restart:
        est = timezone('EST')
        model_path = model_path + '/' + datetime.datetime.now(est)\
            .strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


