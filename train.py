import requests
from torch import tensor

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
output_file = "input.txt"
response = requests.get(url)
response.raise_for_status()  # Check for any errors during the request
with open(output_file, "wb") as file:
    file.write(response.content)
print("File downloaded successfully.")

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('length of dataset in characters:', len(text))
# taking a look at first 1000 characters
print(text[:1000])

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# this above is our vocabulary

# next we would like to create strategy to tokenize the input text, converting text to sequence of integers
# as this is a character based language model we will be simply turning characters into integers
# create mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take string and output list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of ints and outputs a string

print(encode("Hii There"))
print(decode(encode("Hii There")))

# now let's encode the entire text dataset and store it into a torch.tensor

import torch  # we use pytorch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])  # the 1000 characters we looked at earlier will to the gpt look like this

# let's now split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% is for training and the final 10% is for validation
train_data = data[:n]
val_data = data[n:]

# we don't feed all the text into transformer at once, computationally expensive & prohibitive
# instead when training a transformer on these datasets we only work with chunks of the dataset
# we sample ramdom chunks from the training set and train them chunks at a time

block_size = 8
train_data[:block_size + 1]

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

# batch dimension
torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?


def get_batch(split):
    # generate a small batch of data of inputs x and y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # stacking in a 4 x 8 row
    return x, y


xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimension
        contet = xb[b, :t + 1]
        target = yb[b, t]
        print(f"when input is {contet.tolist()} the target: {target}")

print(xb)  # our input to the transformer

# now that the batch of input is done that we want to feed into a transformer
# we can now feed the data into neural networks - simplest one bigram language model
# importing pytorch NN module for reproducibility
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):  # creating a bigram language model which is class of nn.module

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel(vocab_size)  # calling the model
logits, loss = m(xb, yb)  # passing in the inputs and the targets
print(logits.shape)
print(loss)

print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
