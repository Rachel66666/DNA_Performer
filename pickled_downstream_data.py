import pickle
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

class SeqDataset(Dataset):

    def __init__(self, filename, numlines):
        self.filename = filename
        self.numlines = numlines
        self.df = pd.read_pickle(filename).iloc[0:numlines, :]

    def __len__(self):
        return self.numlines

    def __getitem__(self):
        dna = self.df.iloc[0, 0]#, self.df.iloc[0, 1]
        seq = np.zeros((1, 100000))
        print(seq.shape)
        for i in range(len(dna)):
            if dna[i] == 'A':
                seq[0, i] = 0.0
            elif dna[i] == 'C':
                seq[0, i] = 1.0
            elif dna[i] == 'G':
                seq[0, i] = 2.0
            elif dna[i] == 'T':
                seq[0, i] == 3.0

        seq = torch.from_numpy(seq)
        item = seq, self.df.iloc[0, 1]
        self.df = self.df.iloc[1:, :]
        return item

    def restart(self):
        self.df = pd.read_pickle(filename).iloc[0:numlines, :]

