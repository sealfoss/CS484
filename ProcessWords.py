import re
from nltk.corpus import stopwords

subreddit_name = "the_donald"
in_filename = "donald_garbage.txt"
in_file = open(in_filename, encoding="utf8")

words = []
lines = in_file.read().splitlines()
lines_count = len(lines)
stop_words = set(stopwords.words('english'))
print("Parsing words from lines...")
for line in lines:
    reduced = re.sub(r'[^a-zA-Z\']', ' ', line)
    split = list(reduced.split(' '))
    split = list(dict.fromkeys(split))
    for word in split:
        if len(word) > 1 and word not in stop_words:
            lower = word.lower()
            words.append(lower)
print("Removing duplicate words...")
words = list(dict.fromkeys(words))

out_filename = subreddit_name + "_words.txt"
out_file = open(out_filename, "w")

print("Writing words to file...")
for word in words:
    out_file.write(word + "\n")
print("All DONE!")

