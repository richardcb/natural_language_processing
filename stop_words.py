from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = 'This is an example showing off stop word filtration.'
stop_words = set(stopwords.words('english'))

# print(stop_words)

words = word_tokenize(example_sentence)

filtered_sentence = []

# iterates through each word and print it if not a stop word
# for loop can be written like this -
# for w in words:
#     if w not in stop_words:
#         filtered_sentence.append(w)

#print(filtered_sentence)

# for loop can also be written on one line like this -
filtered_sentence = [w for w in words if not w in stop_words]

print(filtered_sentence)