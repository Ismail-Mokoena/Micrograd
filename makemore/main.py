import torch

 

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

    return N
 
def predictions(n, gen_obj, letters)->None:
    token_index = 0
    for i in range(20):
        text = []
        itos = {s+1:i for s,i in enumerate(letters)}
        itos[0]='.'
        while True:
            token_array = n[token_index]
            #Normalize
            token_index = torch.multinomial(token_array,num_samples=1,replacement=True,generator=gen_obj).item()
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
#make deterministic generator object
g = torch.Generator().manual_seed(2147483647)
probabilities = counts_array.float()
#obeys  broadcasting rules
probabilities /= probabilities.sum(1,keepdim=True)

predictions(probabilities,g,letters)


