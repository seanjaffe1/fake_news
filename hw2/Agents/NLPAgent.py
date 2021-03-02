import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import h5py
import json
from tqdm import tqdm

from Data import *
from Model import *

class NLPAgent():

    def __init__(self, config):
        with open("words2idx.json", 'r') as f:
            word2id = json.load(f)

        self.vocab_len = len(word2id.keys())-1

        with open("speaker2idx.json", 'r') as f:
            self.speaker2idx = json.load(f)
        self.speaker2idx["<UNK>"] = len(self.speaker2idx.keys())

        with open("party2idx.json", 'r') as f:
            self.party2idx = json.load(f)
        self.party2idx["<UNK>"] = len(self.party2idx.keys())
            
        with open("state2idx.json", 'r') as f:
            self.state2idx = json.load(f)
        self.state2idx["<UNK>"] = len(self.state2idx.keys())

        with h5py.File("embedding_data.hdf5", "r") as f:
            weights = f['.']['embeddings'].value
        #print(word2id['the'], weights[word2id['the']])
        #print(word2id['<UNK>'], weights[word2id['<UNK>']])

        self.poss_labels = ['true', 'mostly-true', 'half-true', 
                        'barely-true', 'false', 'pants-fire']
        self.labels2id = {}
        for i, l in enumerate(self.poss_labels):
            self.labels2id[l] = i


        self.batch_size = config.batch_size
        # GPU assign
        # TOD
        self.has_cuda = torch.cuda.is_available()
        self.cuda = self.has_cuda and config.cuda
        config.cuda = self.cuda
        if self.cuda:
            self.device = torch.device("cuda:0")
            # torch.cuda.set_device(self.config.gpu_device)
        else:
            self.device = torch.device("cpu")

        if config.mode == 'train':
            self.train_dataset = LIARDataset(config.train_file,  word2id, self.speaker2idx, self.state2idx, self.party2idx, mode= 'train')
            

            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                        batch_size=self.batch_size,
                        pin_memory=self.cuda,
                        shuffle=True,
                        #num_workers=self.config.num_workers
                        )

            if config.valid_file is not None:
                self.valid_dataset = LIARDataset(config.valid_file, word2id, self.speaker2idx, self.state2idx, self.party2idx, mode='validate')
                self.val_dataloader = DataLoader(dataset=self.valid_dataset,
                                     batch_size=1,
                                     pin_memory=self.cuda,
                                     shuffle=True,
                                     #num_workers=self.config.num_workers
                                     )
        
        else:
            self.test_dataset = LIARDataset(config.test_file, word2id, self.speaker2idx, self.state2idx, self.party2idx, mode='test')
            self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                     batch_size=1,
                                     pin_memory=self.cuda,
                                     shuffle=False,
                                     #num_workers=self.config.num_workers
                                     )


        self.model = MixedLSTMModel(self.vocab_len, 2917, 87, 24, weights, self.device).to(device=self.device)

        self.loss = nn.CrossEntropyLoss()

        self.config = config

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr = config.lr,
                                        weight_decay = .0001)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    patience=10,
                                                                    verbose=True)
        
        self.load_checkpoint(self.config.load_checkpoint_file)
        
        
        # Initialize counter
        self.current_epoch = 0
        self.current_iteration = 0

        self.loss = nn.CrossEntropyLoss()


    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename)) 
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])                                               

            print("Successfully loaded checkkpoint from '{}'\n"
                    .format(self.config.checkpoint_dir))
        except OSError as e:
            print("NO checkpoint found at {}".format(self.config.checkpoint_dir))
            print("Training for first time")

    def save_checkpoint(self, filename):
        if filename == "None":
            print("'None' is filename")
            return

        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        print("Saving checkpoint at" + str(self.config.checkpoint_dir + filename))


    def train(self):
        
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch

            # TODO 2 Check where scheduler update should go
            self.scheduler.step(epoch)

            self.train_one_epoch()

            #TODO 3 validate here if you want

            if epoch % 20 == 0:  
                self.save_checkpoint(self.config.save_checkpoint_file)
            
            
            
            
        self.save_checkpoint(self.config.save_checkpoint_file)

    
    
    def train_one_epoch(self):
        i = 0
        for x ,t_lens, y in tqdm(self.train_dataloader, desc = "Epoch: {}".format(self.current_epoch)):
            self.optimizer.zero_grad()
            # print("X:", x)
            # print("tlens:", t_lens)
            # print("Y:", y)
            #TODO cuda? async?
            #if self.cuda:
            #    # TODO 2 do I need to do pin memory?
            #    x = x.pin_memory().cuda(non_blocking=self.config.async_loading)
            #    y = y.pin_memory().cuda(non_blocking=self.config.async_loading)
            # TODO 2 put x and y on cuda here?
            
            #print("Statement:", x[0])
            # print("Speaker:", x[1])
            # print("state:", x[2])
            # print("party", x[3])
            # print("Y:",  y)
            # print("Y", [self.labels2id[yi] for yi in y])
            
            x = [xi.to(dtype=torch.int64, device=self.device, non_blocking=True) for xi in x]
            # TODO 1 change x so it can be passed into model

            y = [[self.labels2id[yi]] for yi in y]
            y = torch.Tensor(y).to(dtype=torch.int64, device=self.device, non_blocking=True).squeeze(dim=1)

            out = self.model(x, t_lens).squeeze(dim=1)
            #print("label:", y, y.dim())
            #print("out:", out, out.size(), y)
            loss = self.loss(out, y)

            
            loss.backward()
            self.optimizer.step()
            i+=1
            if i ==1000: 
                 self.validate()
        print("CURR Loss:", loss)


    def validate(self):
        correct = 0
        datapoints = 0
        # TODO 1 change x so it can be passed into model
        for x, t_lens, y in tqdm(self.val_dataloader):


            x = [xi.to(dtype=torch.int64, device=self.device, non_blocking=True) for xi in x]
            # TODO 1 change x so it can be passed into model
            


            y = [[self.labels2id[yi]] for yi in y]
            
            #print("X:", x)
            out = self.model(x, t_lens).squeeze(dim=1)
            #print("OUT:", out,  y[0][0])
            #print("label:", y[0][0])
            # print(torch.argmax(out).item())
            if torch.argmax(out).item() == y[0][0]:
                correct +=1 # TODO 1
            datapoints += 1.0
            if datapoints > 1400:
                break
        print("Validation Accuracy = ", str(correct / datapoints))
    
    
    def test(self, outfile):
        "Outputs predictions to outfile"
        preds = []
        # TODO 1 change x so it can be passed into model
        for x, t_lens in tqdm(self.test_dataloader):
            # TODO 1 change x so it can be passed into model

            x = [xi.to(dtype=torch.int64, device=self.device, non_blocking=True) for xi in x] 
            out = self.model(x, t_lens).squeeze(dim=1)
            # print("OUT:", out)
            # print("label:", y[0][0])
            # print(torch.argmax(out).item())
            argmax =  torch.argmax(out).item()           
            preds.append(self.poss_labels[argmax]) # TODO 1
    
        with open(outfile, "w") as out:
            out.write('\n'.join(preds))
