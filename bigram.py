import torch
import torch.nn as nn  # importing pytorch NN module for reproducibility
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------


torch.manual_seed(1337)

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(
    set(text)))  # TEXT is a sequence of characters in python, when you call set constructor on it, you get set of all characters that occur in the text, then you call list on that, so you have a list of charcaters with an ordering, then you sort that
vocab_size = len(chars)  # no of characters is our vocab size, possible elements of our sequences
# this above is our vocabulary

# next we would like to create strategy to tokenize the input text, converting text to sequence of integers
# as this is a character based language model we will be simply turning characters into integers
# create mapping from characters to integers
# when people say tokenize, they mean convert the raw text as a string to some sequence of integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take string and output list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of ints and outputs a string
# we are keeping the tokenizer very simple here and thus are using a character level tokenizer
# very simple code books, encoder and decoder, but we do get a long sequence as a result e.g. hii there [46, 47, 47, 1, etc etc]

# now let's encode the entire text dataset and store it into a torch.tensor
# we use pytorch
data = torch.tensor(encode(text),
                    dtype=torch.long)  # take all text in tini shakespeare, encode it and wrap it in torch.tensor to get the data tensor.

# let's now split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% is for training and the final 10% is for validation
train_data = data[:n]
val_data = data[
           n:]  # This will prevent perfect memorisation of the exact shakespeare, instead our model should be able to produce shakespeare like text


# we don't feed all the text into transformer at once, computationally expensive & prohibitive
# instead when training a transformer on these datasets we only work with chunks of the dataset
# we sample random chunks from the training set and train them chunks at a time
# these chunks have a maximum length (block_size)
# EXAMPLE: block_size = 8
#          train_data[:block_size+1]
# Output:  tensor([18, 47, 56, 57, 58, 1, 15, 47, 58]) # What this means is:
# In the context of 18, 47 comes next. In the context of 18 and 47, 56 comes next. AND SO ON.

# get batch for any arbitrary split
def get_batch(split):
    # generate a small batch of data of inputs x and y
    data = train_data if split == 'train' else val_data  # if split is training split then look at train_data otherwise val_data. That get sus the data array
    ix = torch.randint(len(data) - block_size, (batch_size,))  # ix is going to be 4 random generated numbers
    x = torch.stack([data[i:i + block_size] for i in ix])  # first block_size characters, starting at i
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # stacking in a 4 x 8 row
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# now that the batch of input is done that we want to feed into a neural network. In this case we will use the simplest language model being the Bigram language model
# we can now feed the data into neural networks - simplest one bigram language model
# super simple bigram language model
class BigramLanguageModel(nn.Module):  # here we are constructing a bigram language model, which is a class of nn.module

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # here in the constructor: we are creating a token embedding table of size vocabsize * vocabsize
        self.token_embedding_table = nn.Embedding(vocab_size,
                                                  vocab_size)  # here we are using a nn.embedding which is basically a very thin wrapper around a tensor of a shape vocabsize * vocabsize

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


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# The Mathematical trick in self-Attention
torch.manual_seed(1337)
B, T, C = 4, 8, 2  # BATCH, TIME, CHANNEL
x = torch.randn(B, T, C)
x.shape

# we want x[b,t] = mean_{i<=t} x[b, i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]  # (t,C)
        xbow[b, t] = torch.mean(xprev, 0)  # XBOW = x bag of words
