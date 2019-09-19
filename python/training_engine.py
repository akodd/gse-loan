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
        self.model = None
        self.loss_function = None
        self.optimizer = None
        self.trainDL = None
        self.validDL = None

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

    def useGPU(self, use=False):
        if use:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        print('Using device: {}'.format(self.device.type))
        self.model.to(self.device)


    # def loadModel(self):
    #     if os.path.exists(self.model_path):
    #         print('Loading model checkpoint: {}'.format(self.model_path))
    #         checkpoint = torch.load(self.model_path)
    #         self.model.load_state_dict(checkpoint['model_state_dict'])
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         self.train_losses = checkpoint['train_losses']
    #         self.valid_losses = checkpoint['valid_losses']
    #         self.checkpoint_epoch = checkpoint['epoch']
    #     else:
    #         self.checkpoint_epoch = 0
# 
    # def saveModel(self, epoch):
    #     torch.save({ 
    #         'epoch': epoch,
    #         'model_state_dict': self.model.module.state_dict(), 
    #         'model_config' : self.model.module.model_config,
    #         'optimizer_state_dict': self.optimizer.state_dict(), 
    #         'train_losses' : self.train_losses, 
    #         'valid_losses' : self.valid_losses 
    #         }, self.model_path 
    #     ) 

    def makeParallel(self, use=False):
        if use and torch.cuda.device_count() > 1:
            print("Training on", torch.cuda.device_count(), "GPUs")
            self.model = nn.DataParallel(self.model)


def adjustModelPath(model_path, restart=True):
    if not restart:
        est = timezone('EST')
        model_path = model_path + '/' + datetime.datetime.now(est)\
            .strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


