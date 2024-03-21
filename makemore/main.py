#import torch

def create_bi(words:list)->None:
    #zip takes two iteraters & returns a tupple
    for word in words:
        #special start token ane end token
        tokenized = ['<S>'] + list(word) + ['<E>']
        for ch1, ch2 in zip(tokenized, tokenized[1:]):
            key = (ch1, ch2)
            #bigram[key] = bigram.get(key,0) + 1


#list of strings
words  = open('Micrograd/makemore/names.txt','r').read().splitlines()
#alphabet lookup a=1,b=2,...
letters =  sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(letters,1)} 
print(stoi)
#bigram
#bigram = {}

#create_bi(words)
#26-letters + 2-special chars <S>, <E> = 28
#a = torch.zeros((3,5), dtype=torch.int32)
#print(lookup)

