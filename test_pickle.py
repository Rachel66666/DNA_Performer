import torch
import pickle





with (open('100klen_64lines_unshuffled.pickle', "rb")) as f:
    n=1
    while n>0:

        try:
            data = pickle.load(f)
            n=n+1
            print("n ",n)
            if (n==64):
                print("data", data)

        except EOFError:
            raise StopIteration