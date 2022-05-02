from torch.utils.data import Dataset
import pickle


class SeqDataset(Dataset):
    """
    A class that is used to generate pickle data from the .bed file for training purpose
    """
    def __init__(self, input_file_name,numlines):
        self.numlines= numlines
        self.input_file_name = input_file_name
        self.pickle_file = open(input_file_name, "rb")
        
    def __len__(self):
        return self.numlines#1000

    def __getitem__(self,idx):
        
        try:
            one_seq = pickle.load(self.pickle_file)
            
        except EOFError:
            raise StopIteration
        return one_seq

    def restart(self):
        self.pickle_file = open(self.input_file_name, "rb")
        
