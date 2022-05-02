import torch

def cparam(model):
    """
    Count the number of parameters in a model
    """
    t_params = 0
    params = list(model.parameters())
    for i in range(len(params)):
        t_params += torch.numel(params[i])
    return t_params


def model_summary(model):
    """
    Summary of a model in terms of layers
    """
    print()
    print("Layer_name"+"\t"*7+"Number of Parameters")
    print("="*100)
    model_parameters = [layer for layer in model.parameters()]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    for i in layer_name:
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False  
        if not bias:
            param =model_parameters[j].numel()
            j = j+2
        else:
            param =model_parameters[j].numel()
            j = j+1
            print(str(i)+"\t"*3+str(param))
            total_params+=param
        print()
    print("="*100)
    print(f"Total Params:{total_params}")   

def model_summary2(model):
    """
    Alternative version of model summary function
    """
    print()
    print("Layer_name"+"\t"*7+"Number of Parameters")
    print("="*100)
    model_parameters = [layer for layer in model.parameters()]
    layer_name = [child for child in model.children()]
    total_params = 0
    for i in range(len(layer_name)):
        param = 0
        param = torch.numel(model_parameters[i])
        print(str(layer_name[i])+"\t"*3+str(param))
        total_params+=param
    print("="*100)
    print(f"Total Params:{total_params}")   