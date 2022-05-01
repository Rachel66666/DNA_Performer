import random
from torch.utils.data import Dataset
import io
import pickle


class SeqDataset(Dataset):

    def __init__(self, input_file_name,numlines):
        self.numlines= numlines
        self.input_file_name = input_file_name
        self.pickle_file = open(input_file_name, "rb")
        
    def __len__(self):
        
        #self.numlines
        return self.numlines#1000

    def __getitem__(self,idx):
        
        try:
            one_seq = pickle.load(self.pickle_file)
            
        except EOFError:
            #print("starting from ")
            #self.pickle_file = open(self.input_file_name, "rb")

            raise StopIteration
            #print("end of file")
        return one_seq
    def restart(self):
        
        self.pickle_file = open(self.input_file_name, "rb")
        
