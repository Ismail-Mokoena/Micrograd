import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

def build_dataset(words):
    #x~neuron inputs, y~labels for each example in x
    x, y = [], []
    #block_size = How many chars do we take in to predict the next one?
    block_size = 3
    for word in words:       
        context = [0]*block_size
        for character in word + '.':
            ix = stoi[character]
            x.append(context)
            y.append(ix)
            context = context[1:] + [ix]
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x,y


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
x,y = build_dataset(names)

#training=80%, dev=10%, test=10%
random.seed(42)
random.shuffle(names)
n1 = int(0.8*len(names)) #80% of words
n2 = int(0.9*len(names)) #90% of words
#splits
x_train, y_train = build_dataset(names[:n1])#80%
x_dev, y_dev = build_dataset(names[n1:n2])#10%
x_test, y_test = build_dataset(names[n2:])#10%



"""Forward Pass"""
g = torch.Generator().manual_seed(2147483647)
#c~lookup table
c = torch.randn((27,2), generator=g)

"""1.hidden layer"""
#n_neurons = 100, n_inputs = 6
#embed.shape = 18,3,2 so no imputs = 3*2
#100~ we determine that
weight_1 = torch.randn((6,300),generator=g)*0.1
biases_1 = torch.randn(300,generator=g)*0.01



"""2.Final layer"""
#n_neurons = 27, n_inputs = 100
weight_2 = torch.randn((300,27),generator=g)*0.01
biases_2 = torch.randn(27,generator=g)*0
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
learn_exp = torch.linspace(0.001,0.1,10000)
learning_rate = 10**learn_exp    

lri, lossi = [], []
"training"
for i in range (10000):
    #minibatch 
    ix = torch.randint(0,x_train.shape[0], (32,))
    
    """Forward Pass"""
    embed = c[x_train[ix]]
    # 18,3,2=>18,6
    f =  torch.tanh(embed.view(-1,6)@weight_1 + biases_1)
    #NN ouput
    logits = f@weight_2 + biases_2
    loss = F.cross_entropy(logits,y_train[ix])
    #print(loss.item())
    """Backward Pass"""
    #set gradients to zero
    for param in parameters:
        param.grad = None
    loss.backward() 

    #update we want to nudge our params
    #l = learning_rate[i]
    l=0.01
    for params in parameters:
        param.data += -l*param.grad
        
    #track stats
    lri.append(l)
    lossi.append(loss.item())

#plt.figure(figsize=(20,10))
#plt.imshow(f.abs()>0.99, cmap='gray', interpolation='nearest')
print(loss.item())
#plt.plot(lri,lossi)  
#plt.show() 

"""Sample from model


r = torch.Generator().manual_seed(2147483647+10)
block_size = 3
for _ in range (10):
    out = []
    context = [0]*block_size
    while True:
        emb = c[torch.tensor([context])]
        h = torch.tanh(emb.view(1,-1)@weight_1 +biases_1)
        logits = h@weight_2 + biases_2
        prob = F.softmax(logits,dim=1)
        ix = torch.multinomial(prob, num_samples=1, generator=r).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))

"""
