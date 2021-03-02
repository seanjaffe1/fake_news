
import torch
import torch.nn as nn
import numpy as np




class BiLSTM(nn.Module):

    def __init__(self,  weights, hidden_size, output_size, device):
        super(BiLSTM, self).__init__()
        #print("Weights", weights.shape, len(weights))
        self.embedding_dim = 50
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            num_embeddings=len(weights),
            embedding_dim = self.embedding_dim,
            padding_idx = 0,
        )
        self.embedding.load_state_dict({'weight': torch.Tensor(weights)})
        #TODO 1 check batch_first is correct
        # TODO 1 put make the embeddings be the weight matrix we found
        
        #print("Test:", self.embedding(torch.LongTensor([401318])))
        
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size= self.hidden_size,
            bidirectional=False,
            batch_first=True
        )

        #self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, t_lens):

        #print("aaa")
        #print(self.embedding(torch.LongTensor([0])))
        try:
            inp = self.embedding(input)
        except IndexError:
            print("ERROR OCCURED")
            print(input)
            raise IndexError()

        #statement = torch.nn.utils.rnn.pack_padded_sequence(inp, t_lens, batch_first=True, enforce_sorted=False)
        # TODO set h0?

        hidden = (torch.randn(1, 1, self.hidden_size).zero_().to(self.device),
          torch.randn(1, 1, self.hidden_size).zero_().to(self.device))
        output, (final_h, final_c) = self.lstm(inp, hidden)
        return final_h

    
class MixedLSTMModel(nn.Module):
    def __init__(self, vocab_size, speakers, states, parties, weights, device):
        super(MixedLSTMModel, self).__init__()
        self.lstm = BiLSTM(weights, 20, 20, device)

        self.speaker_embedding = nn.Embedding(
            num_embeddings = speakers+1,
            embedding_dim = 20
        )
        
        self.states_embedding = nn.Embedding(
            num_embeddings = states+1,
            embedding_dim = 8
        )

        self.parties_embedding = nn.Embedding(
            num_embeddings = parties+1,
            embedding_dim = 4
        )

        self.meta = nn.Sequential(nn.Linear(20 + 8 + 4, 16),
                                nn.ReLU(),
                                nn.Linear(16, 16),
                                nn.ReLU())
    
        self.conj = nn.Sequential(nn.Linear(20 + 16, 16),
                                nn.ReLU(),
                                nn.Linear(16, 6))

        self.pred = nn.Softmax(dim = 2)

    def forward(self, X, t_lens):

        statement_h = self.lstm(X[0], t_lens)
        #print(statement_h)
        #print(statement_h)
        #statement_h[:] = 0

        speaker = self.speaker_embedding(X[1])
        state  = self.states_embedding(X[2])
        party = self.parties_embedding(X[3])

        #print(speaker)
        x = torch.cat((speaker, state, party), 2)
        
        out1 = self.meta(x)

        #out1[:] = 0
        out2 = torch.cat((statement_h, out1), 2)
        out3 = self.conj(out2)

        return self.pred(out3)
    
