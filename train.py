import requests

#
# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# output_file = "input.txt"
# response = requests.get(url)
# response.raise_for_status()  # Check for any errors during the request
# with open(output_file, "wb") as file:
#     file.write(response.content)
# print("File downloaded successfully.")

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
print(data[:1000]) # the 1000 characters we looked at earlier will to the gpt look like this
