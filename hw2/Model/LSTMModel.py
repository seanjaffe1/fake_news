
import torch
import torch.nn as nn
import numpy as np




class BiLSTM(nn.Module):

    def __init__(self, vocab_size, weights, hidden_size, output_size):

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim = 50,
            padding_idx = -1,
        )
        #TODO 1 check batch_first is correct
        # TODO 1 put make the embeddings be the weight matrix we found

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size= self.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        #self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        inp = self.embedding(input)

        # TODO set h0?

        output, (final_h, final_c) = self.lstm(inp)
        return final_h

    
class MixedLSTMModel(nn.Module):
    def __init__(self, vocab_size, speakers, states, parties, weights):
        
        self.lstm = BiLSTM(vocab_size, weights, 32, 32)

        self.speaker_embedding = nn.Embedding(
            num_embeddings = speakers,
            embedding_dim = 20
        )
        
        self.states_embedding = nn.Embedding(
            num_embeddings = states,
            embedding_dim = 8
        )

        self.parties_embedding = nn.Embedding(
            num_embeddings = parties,
            embedding_dim = 4
        )

        self.meta = nn.Sequential(nn.linear(20 + 8 + 4, 32),
                                nn.ReLu(),
                                nn.linear(32, 32),
                                nn.ReLu())
    
        self.conj = nn.Sequential(nn.linear(32 + 32, 48),
                                nn.ReLu(),
                                nn.linear(48, 6),
                                nn.Softmax())

    def forward(self, X):
        statement = X[:,0]

        statement_h = self.lstm(statement)

        speaker = self.speaker_embedding(X[:, 1])
        state  = self.state_embedding(X[:, 2])
        party = self.parties_embedding(X[:,3])

        x = torch.cat((spealer, state, party), 1)

        out1 = self.meta(x)

        out2 = torch.cat((statement_h, out1), 1)

        return self.conj(out2)
    
