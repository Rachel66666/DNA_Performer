import torch
import tensorflow 
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import time #Speed
import math
from model.helpers import cparam


class TrainerConfig:
    # optimization parameters
    epochs = 1000
    learning_rate = 3e-4 #default: 1e-3
    betas = (0.9, 0.95) #default (0.9, 0.999)
    grad_norm_clip = 1.0
    optimizer = 'adamw'
    weight_decay = 0.1 # only applied on matmul weights
    logdir = './runs/'
    model_name = 'trained_model/test_model_1000line.model'

#For DATALOADER
    batch_size = 16
    num_workers = 0 # for DataLoader
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)



    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k,v)


class Trainer:
    def __init__(self, model, train_dataset, config):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.reg_or_classify = 'classify'
        self.tokens = 0

        #Check Points Code go here - NOT IMPLEMENTED

        self.device = 'cpu'# if no GPU then it is cpu 
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.device = torch.cuda.current_device() # if there is GPU then use GPU 
            # self.model = torch.nn.DataParallel(self.model).to(self.device) 
            self.model = self.model.to(self.device) 
                            # I think this is a combination of 2 lines of code: 
                            # 1. self.model = torch.nn.DataParallel(self.model)
                            # 2. self.model.to(device)
                        
    def train(self):
        model, config = self.model, self.config

        total_params = cparam(model)
        print("Total_Params:", total_params)    
        #print("")    
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas)

        if self.reg_or_classify=='classify':
            loss_function= torch.nn.CrossEntropyLoss()
        else :
            loss_function= torch.nn.MSELoss()
        
        writer = SummaryWriter(config.logdir)

        
        def run_epoch(it_total=0):
            model.train
            data = self.train_dataset
            
            loader = DataLoader(data, batch_size=config.batch_size, num_workers=config.num_workers)

            #print("it_total", it_total)
            #Assuming progress bar is always TRUE
            pbar = tqdm(enumerate(loader), total=len(loader))
            #print("len(loader) ", len(loader))
            running_loss = 0.0
            last_time = time.monotonic()

            printed = False
            
            for it , example in pbar:
                #print("it", it)
                # X -> Mask sequence
                x = example[0].long()
                x = x.to(self.device)
                # Targets -> Original sequence
                targets = torch.tensor(example[1])
                #targets = torch.tensor(example[1])
                targets = targets.to(self.device)
                # mask -> Indicates which position is mutated 
                # mask = example[2].float()
                # mask = mask.to(self.device)
                
                # forward the model
                with torch.set_grad_enabled(True):
                    
                    if config.lr_decay:
                        self.tokens += (targets >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            if not printed:
                                #print("finished warmup!")
                                printed = True
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    
                    logits = model(x)
                    
                    #If we are given some desired targets also calculate the loss
                    loss = None
                    accuracy = None
                    if targets is not None:
                        #Accuracy -> The predictions right now are the logits (output of the mode)
                        #Convert the logits to actual predictions
                        # _, predictions = torch.max(logits.view(-1, logits.size(-1)), 1)
                        # total = torch.sum(targets.view(-1) != -100)
                        _, predictions = torch.max(logits,1)
                        correct = (predictions == targets.view(-1)).sum()
                        accuracy = correct/(config.batch_size+0.0000001)
                        
                        # loss = F.cross_entropy(
                        #     logits.view(-1, logits.size(-1)), targets.view(-1))
                        # loss = (loss * mask).mean()
                        
                        #print("logits", logits.size())
                        #
                        #print("prediction",predictions.size())
                        #print("target", targets.size())

                        loss = F.cross_entropy(logits, targets)
                        #     logits.view(-1, logits.size(-1)), targets.view(-1))
                        

                    
                    #Backprop and update the parameters
                    #Zero all of the gradients for the variables it will update
                    optimizer.zero_grad() #Should this be like this or model.zero_grad()

                    #BackwardPass: compute gradient of the loss with respect to model
                    loss.backward()

                    #gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

                    optimizer.step()  #update the parameters


                    #Speed 
                    current_time = time.monotonic()
                    delta_time = current_time - last_time
                    last_time = current_time

                    
                    #TensorBoard
                    
                    writer.add_scalar('Accuracy', accuracy, it_total)
                    writer.add_scalar('Loss', loss, it_total)
                    writer.add_scalar('Learning rate',lr, it_total)
                    writer.add_scalar('Iteration/second', 1.0/delta_time, it_total)
                    it_total = it_total+1
                    #print("it total", it_total)

                    #Returns the loses
                    running_loss += (loss.item() - running_loss)/min(it+1.0, 1000.0)
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. Accuracy {accuracy}, running loss {running_loss:0.5f}, Iteration/second {1.0/delta_time}")
            return it_total

        it_total = 0
        for epoch in range(config.epochs):
            #print("config.epochs: ", config.epochs)
            it_total= run_epoch(it_total)
            self.train_dataset.restart()