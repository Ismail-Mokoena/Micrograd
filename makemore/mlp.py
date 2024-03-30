import torch
import torch.nn.functional as F

def build_dataset(block_size, words, stoi, itos):
    #x~neuron inputs, y~labels for each example in x
    x, y = [], []
    for word in words[:6]:
        print(word)
        context = [0]*block_size
        for character in word + '.':
            ix = stoi[character]
            x.append(context)
            y.append(ix)
            print(''.join(itos[i] for i in context), '---->', itos[ix])
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

"""Embedding lookup tabel"""
#Embed our charcters into a lower dimensional space 27dim->2dim
#c~lookup table
c = torch.randn(27,2)
embed = c[x]

"""hidden layer"""
#n_neurons = 100, n_inputs = 6
#embed.shape = 18,3,2 so no imputs = 3*2
#100~ we determine that
weight_1 = torch.randn((6,100))
biases_1 = torch.randn(100)
# 18,3,2=>18,6
f =  torch.tanh(embed.view(-1,6)@weight_1 + biases_1)
#f~shape = 18,100
"""Final layer"""
#n_neurons = 27, n_inputs = 100
weight_2 = torch.randn((100,27))
biases_2 = torch.randn(27)
parameters = [c,weight_1,biases_1,weight_2,biases_2]
total_params = sum( param.nelement() for param in parameters)
"""NN ouput"""
logits  = f@weight_2 + biases_2

"""Get counts + Normalize"""
#counts = logits.exp()
#probability = counts/counts.sum(1, keepdim=True)
#since this is classification we can use cross entropy for loss
"""index the current prob given to the correct next char in sequence"""
#y = the actual next char in sequence
#we want the -log mean to create the negative log likelyhood loss
#-probability[torch.arange(probability.shape[0]),y].log().mean()
 
loss = F.cross_entropy(logits,y)
print(loss)

    