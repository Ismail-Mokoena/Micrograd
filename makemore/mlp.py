import torch
import torch.nn.functional as F

def build_dataset(block_size, words, stoi, itos):
    #x~neuron inputs, y~labels for each example in x
    x, y = [], []
    for word in words[:3]:
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
c = torch.randn(27,2)

print(x.shape)

    