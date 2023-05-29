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