import re
from data_preprocessing.tokenizer import *

with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
#print("Total number of character:", len(raw_text))
#print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(len(preprocessed))

all_tokens = sorted(set(preprocessed))

vocab_size = len(all_tokens)
#print(vocab_size)
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i > 50:
#         break

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

tokenizer = SimpleTokenizerV2(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))


text = "Hello, do you like tea?"
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))