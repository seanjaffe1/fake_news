import numpy as np
import torch
import torch.nn as nn
import h5py

from Data import *
from Model import *

class NLPAgent():

    def __init__(self, config):
        with open("../words2idx.json", 'r') as f:
            word2id = json.load(f)
            
        with open("../speaker2idx.json", 'r') as f:
            speaker2idx = json.load(f)
            
            
        with open("../party2idx.json", 'r') as f:
            party2idx = json.load(f)
            
            
        with open("../state2idx.json", 'r') as f:
            state2idx = json.load(f)

        with h5py.File("embeddig_data.hdf5", "r") as f:
            weights = f['.']['embeddings'].value

        self.poss_labels = ['true', 'mostly-true', 'half-true', 
                        'barely-true', 'false', 'pants-fire']
        self.labels2id = {}
        for i, l in enumerate(self.poss_labels):
            self.labels2id[l] = i

        # GPU assign
        # TOD
        self.has_cuda = torch.cuda.is_available()
        self.cuda = self.has_cuda and self.config.cuda
        config.cuda = self.cuda
        if self.cuda:
            self.device = torch.device("cuda:0")
            # torch.cuda.set_device(self.config.gpu_device)
        else:
            self.device = torch.device("cpu")

        if mode == 'train':
            self.train_dataset = LIARDataPoint(config.train_dataset,  word2id, speaker2idx, state2idx, party2idx, mode= 'train')
            
            self.vocab = self.train_dataset.vocab


            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                        batch_size=self.batch_size,
                        pin_memory=self.cuda,
                        shuffle=True,
                        #num_workers=self.config.num_workers
                        )
            if self.config.valid_file is not None:
                self.valid_dataset = LIARDataset(config.valid_dataset, word2id, speaker2idx, state2idx, party2idx, mode='validate')
                self.val_dataloader = DataLoader(dataset=self.valid_dataset,
                                     batch_size=1,
                                     pin_memory=self.cuda,
                                     shuffle=True,
                                     #num_workers=self.config.num_workers
                                     )
        
        else:
            self.test_dataset = LIARDataset(config.test_dataset, word2id, speaker2idx, state2idx, party2idx, mode='test')



        self.model = MixedLSTMModel(len(vocab), 2916, 86, 23, weights)

        self.loss = NN.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr = self.config.lr,
                                        weight_decay = .0001)

        self.scheduler = torch.optim.lr_scheduler.ReduceROnPlateau(self.optimzer,
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
            
            
            
            self.validate(dataloader=self.valid_dataloader)
        self.save_checkpoint(self.config.save_checkpoint_file)

    
    
    def train_one_epoch(self):
        for x, y in tqdm(self.train_dataloader, desc = "Epoch: {}".format(self.current_epoch)):
            #TODO cuda? async?
            #if self.cuda:
            #    # TODO 2 do I need to do pin memory?
            #    x = x.pin_memory().cuda(non_blocking=self.config.async_loading)
            #    y = y.pin_memory().cuda(non_blocking=self.config.async_loading)
            # TODO 2 put x and y on cuda here?

            x = x.to(dtype=torch.float32, device=self.device, non_blocking=True)
            # TODO 1 change x so it can be passed into model

            y = y.to(dtype=torch.float32, device=self.device, non_blocking=True)


            out = self.model(x)

            loss = self.loss(torch.max(out), self.labels2id(y))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("CURR Loss:", loss)


    def validate(self):
        correct = 0
        datapoints = 0
        # TODO 1 change x so it can be passed into model
        for x, y in (self.val_dataloader):
            print(x)
            # TODO 1 change x so it can be passed into model

            
            
    
            correct +=1 # TODO 1
            datapoints += 1.0
        print("Validation Accuracy = ", str(correct / datapoints))
    
    
    def test(self, outfile):
        "Outputs predictions to outfile"
        preds = []
        # TODO 1 change x so it can be passed into model
        for x, y in (self.test_dataloader):
            # TODO 1 change x so it can be passed into model

            
            preds.append() # TODO 1
    
        with(open(outfile), "w") as out:
            for p in preds:
                out.write(p)
