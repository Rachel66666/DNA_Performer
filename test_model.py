import torch
import pickle
#import base_model as model
import model_with_performer as model

atensor=torch.ones(8, 1, 100000, dtype=torch.float32)

'''
input_file_name = '100k_line_100k_len_Mar_25.pickle'
pickle_file = open(input_file_name, "rb")
try:
    atensor = pickle.load(pickle_file)
        
except EOFError:
    #print("starting from ")
    #self.pickle_file = open(self.input_file_name, "rb")

    raise StopIteration
'''



aconfig=model.DNA_Performer_Config(1,1, seq_len=100000)


amodel=model.DNA_Performer(aconfig)



output=amodel(atensor)




