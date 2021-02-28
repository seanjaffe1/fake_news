import torch
import pandas as pandas
import numpy as numpy
from torch.utils.data import Dataset, DataLoader
import string

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")




class LIARDataPoint():

    def __init__(self, args, word2id, speaker2idx, state2idx, party2idx, label=None):
        self.features = args
        self.label = label

        # remove punctuation from statement and split
        statement = self.features[0].translate(str.maketrans(
            '', '', string.punctuation)).replace("-"," ").lower().split(" ")
        
        self.features[0] = []
        for word in statement:
            if word in word2id.keys():
                self.features[0].append(word2id[word])
            else:
                self.features[0].append(word2id["<UNK>"])
        
        self.text_len = len(self.features[0])
        # pad with zeros
        while len(self.features[0]) < 68:
            self.features[0].append(word2id["<PAD>"])

        speaker = self.features[2]
        if speaker in speaker2idx.keys():
                self.features[2] = speaker2idx[speaker]
        else:
            self.features[2] = len(speaker2idx.keys())

        state = self.features[4]
        if state in state2idx.keys():
                self.features[4] = state2idx[state]
        else:
            self.features[4] = len(state2idx.keys())

        party = self.features[5]
        if party in party2idx.keys():
                self.features[5] = party2idx[party]
        else:
            self.features[5] = len(party2idx.keys())    


           
    def statement(self):
        return self.features[0]

    def subject(self):
        return self.features[1]   
 
    def speaker(self):
        return self.features[2]

    def title(self):
        return self.features[3]

    def state(self):
        return self.features[4]

    def party(self):
        return self.features[5]

    def context(self):
        return self.features[6]

    def __len__():
        return self.text_len
        

class LIARDataset(Dataset):
    
    def __init__(self, input_file, word2id, speaker2idx, state2idx, party2idx, mode='train'):
        """
        Args:
            input_file (string): path to csv file
        "

        """

        self.data = []
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                l = line.rstrip('\n').split('\t')
                if mode == 'train' or mode == 'validate':
                    label = l[0]
                    features = l[1:]
                else:
                    features = l
                    label = None
                self.data.append(LIARDataPoint(features, word2id, speaker2idx, state2idx, party2idx, label=label))
                


    def __len__(self):
        return(len(self.labels))

    def __getitem__(self, idx):
        # TODO make sure this is correct
        return self.data[idx].features, self.data[idx].label