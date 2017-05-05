from nltk.tokenize import sent_tokenize, word_tokenize

# a word tokenizer separates sentences by words
# a sentence tokenizer separates paragraphs by sentences
# Corpora - body of text, ie: medical journals, presidential speeches, English language
# Lexicon - words and their means

example_text = 'Hello there, how are you doing today? The weather is great and python is awesome.'

# prints a list of each sentence
#print(sent_tokenize(example_text))

# print a list of each word
#print(word_tokenize(example_text))

# prints a list of each word in a column
for i in word_tokenize(example_text):
    print(i)
