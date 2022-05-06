import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import loader.pickledataset as dataset


class Tester:
    def __init__(self, PATH, test_dataset):
        self.model = torch.load(PATH, map_location=torch.device('cpu'))#.state
        self.test_loader = DataLoader(test_dataset, batch_size=2, num_workers=0)
        self.device = 'cpu'# if no GPU then it is cpu 

        
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.device = torch.cuda.current_device() # if there is GPU then use GPU 
            self.model = torch.nn.DataParallel(self.model).to(self.device) 

    def test_run(self):
        test_accuracy = 0.0
        total = 0
        correct = 0
        for it, example in enumerate(self.test_loader):
            # testing inputs
            x = example[0].long()
            x = x.to(self.device)

            targets = torch.tensor(example[1])
            targets = targets.to(self.device)

            # apply the model
            output = self.model(x)

            # calculate accuracy
            _, predictions = torch.max(output.view(-1, output.size(-1)), 1)
            total += torch.sum(targets.view(-1) != -100)
            correct += (predictions == targets.view(-1)).sum()
            test_accuracy = correct/(total+0.0000001)
            print("Testing accuracy at", it, "sequnce is", test_accuracy)
        
        test_accuracy = correct/(total+0.0000001)
        return test_accuracy

# model path
print("Loading model...")
#PATH = "./runs_performer/performer_lr3e-4/lrdecay1/bs8/n_att_layer3/head2/n_embd2000/seq_len100000/em_dr0/ff_dr0/att_dr0/epoch20/nlines100000/04-16-2022_23-12-53/model"
#PATH = "./runs_performer/classic_transformer_4embd_lr3e-4/lrdecay1/bs8/n_att_layer5/head4/n_embd3500/seq_len100000/epoch20/nlines100000/04-23-2022_13-33-43/model"
#PATH = "./runs_performer/performer/nb_features5/interval100000000/lr3e-4/lrdecay1/bs8/n_att_layer5/head4/n_embd3500/seq_len100000/att_dr0/epoch20/nlines100000/04-23-2022_23-23-44/model"
PATH = "./runs_performer/cnn_v1_lr3e-4/lrdecay1/bs8/n_cnn_layer2/n_embd3500/seq_len100000/epoch20/nlines100000/04-24-2022_19-35-53/model"

# GPU setting: Define the device
device = 'cpu'# if no GPU then it is cpu 

# load testing dataset
print("Loading DataSet...")
pickle_path = 'Downstream_Data.pkl'
test_dataset = dataset.SeqDataset(pickle_path,numlines=1000)

print("Start Testing...")
tester = Tester(PATH, test_dataset)
acc = tester.test_run()
print("Testing Accuracy:", acc)