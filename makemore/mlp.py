import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def build_dataset(block_size, words, stoi, itos):
    #x~neuron inputs, y~labels for each example in x
    x, y = [], []
    for word in words:       
        context = [0]*block_size
        for character in word + '.':
            ix = stoi[character]
            x.append(context)
            y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(x),torch.tensor(y)

"""Import all words"""
#list of strings
names  = open('Micrograd/makemore/names.txt','r').read().splitlines()
#alphabet lookup a=1,b=2,...
chars =  sorted(list(set(''.join(names))))

"""Mapping of chars->strings & strings->chars"""
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

"""Build the dataset"""
#block_size = How many chars do we take in to predict the next one?
block_size = 3
x,y = build_dataset(block_size, names, stoi, itos)

"""Forward Pass"""
g = torch.Generator().manual_seed(2147483647)
#c~lookup table
c = torch.randn((27,2), generator=g)

"""1.hidden layer"""
#n_neurons = 100, n_inputs = 6
#embed.shape = 18,3,2 so no imputs = 3*2
#100~ we determine that
weight_1 = torch.randn((6,100),generator=g)
biases_1 = torch.randn(100,generator=g)



"""2.Final layer"""
#n_neurons = 27, n_inputs = 100
weight_2 = torch.randn((100,27),generator=g)
biases_2 = torch.randn(27,generator=g)
parameters = [c,weight_1,biases_1,weight_2,biases_2]
total_params = sum( param.nelement() for param in parameters)



"""4.Get counts + Normalize"""
#counts = logits.exp()
#probability = counts/counts.sum(1, keepdim=True)
#since this is classification we can use cross entropy for loss

"""5.index the current prob given to the correct next char in sequence"""
#y = the actual next char in sequence
#we want the -log mean to create the negative log likelyhood loss
#-probability[torch.arange(probability.shape[0]),y].log().mean()
#Embed our charcters into a lower dimensional space 27dim->2dim

#set gradients to zero
for param in parameters:
    param.requires_grad = True
    
#learning rate
learn_exp = torch.linspace(-40,0,1000)
learning_rate = 10**learn_exp    

lri, lossi = [], []
"training"
for i in range (100):
    #minibatch 
    ix = torch.randint(0,x.shape[0], (32,))
    
    """Forward Pass"""
    embed = c[x[ix]]
    # 18,3,2=>18,6
    f =  torch.tanh(embed.view(-1,6)@weight_1 + biases_1)
    #NN ouput
    logits = f@weight_2 + biases_2
    loss = F.cross_entropy(logits,y[ix])
    print(loss.item())
    """Backward Pass"""
    #set gradients to zero
    for param in parameters:
        param.grad = None
    loss.backward()

    #update we want to nudge our params
    l = learning_rate[i]
    for params in parameters:
        param.data += -l*param.grad
        
    #track stats
    lri.append(l)
    lossi.append(loss.item())

#print(loss.item())
plt.plot(lri,lossi)  
plt.show() 