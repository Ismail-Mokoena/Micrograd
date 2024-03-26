import torch
import torch.nn.functional as F

def create_bi(words:list, letters)->torch.int32:
    N=torch.zeros((27,27), dtype=torch.int32)
    stoi = {s:i+1 for i,s in enumerate(letters)}
    #special token
    stoi['.'] = 0
    #zip takes two iteraters & returns a tupple
    for word in words:
        #special start token ane end token
        tokenized = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(tokenized, tokenized[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
            #print(f'{ch1}{ch2}: {prob:}')

    return N
 
def predictions(weights, gen_obj, letters)->None:
    token_index = 0
    for _ in range(30):
        text = []
        itos = {s+1:i for s,i in enumerate(letters)}
        itos[0]='.'
        while True:
            x_encoded = F.one_hot(torch.tensor([token_index]),num_classes=27).float()
            logits  = x_encoded@weights
            #Normalize
            counts = (logits).exp() #x_encoded@weights log-counts
            char_prob = counts/counts.sum(1, keepdims=True)
            token_index = torch.multinomial(char_prob,num_samples=1,replacement=True,generator=gen_obj).item()
            text.append(itos[token_index])
            if token_index == 0:
                break
        print(''.join(text))
        

         

#list of strings
words  = open('Micrograd/makemore/names.txt','r').read().splitlines()
#alphabet lookup a=1,b=2,...
letters =  sorted(list(set(''.join(words))))


counts_array = create_bi(words, letters)

#Probability of any single character being the first word
#counts_array[0]=gives the counts of how often a character starts a word
#first_row = counts_array[0].float()

#probabilities = counts_array.float()
#obeys  broadcasting rules
#probabilities /= probabilities.sum(1,keepdim=True)

#predictions(probabilities,g,letters)

#cast into Neuralnet
#We'll give our NN first char it will predict the next most likely char
#1. create training set
x_set, y_set = [], []
#This iterates over all bigrams
stoi = {s:i+1 for i,s in enumerate(letters)}
#special token deliniate start/end
stoi['.'] = 0
for word in words[:1]:
    print(word)
    #special start token ane end token
    tokenized = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(tokenized, tokenized[1:]):
        print(ch1,':',ch2)
        x_set.append(stoi[ch1])
        y_set.append(stoi[ch2])
x_set = torch.tensor(x_set)
y_set = torch.tensor(y_set)


#weights
#make deterministic generator object
g = torch.Generator().manual_seed(2147483647)
weights = torch.randn((27,27), generator=g, requires_grad=True)



##Gradient Descent
for i in range(10):
    ##Forward Pass
    
    #encoding using 1-hot-encoding
    #then cast our vector to float~we feed floats into nn
    x_encoded = F.one_hot(x_set,num_classes=27).float()

    #@~ matrix multiplication opperator
    logits  = x_encoded@weights

    #counts equivalent to N
    counts = (logits).exp() #x_encoded@weights log-counts
    prob = counts/counts.sum(1, keepdims=True)

    # Loss Function
    #probability the model assignes to the next correct char
    #neg log loss
    #implement regularization loss
    loss = -prob[torch.arange(6), y_set].log().mean() + 0.01*(weights**2).mean()
    print(f'loss={loss}')
    
    ##Backward Pass
    weights.grad = None
    loss.backward()
    #update tensor
    weights.data += -10 * weights.grad

predictions(weights,g,letters)