import re
from nltk.corpus import stopwords

min_line_len = 5

# subreddit_name = "The_Donald"
# subreddit_name = "JoeBiden"
subreddit_name = "ElizabethWarren"-35624
in_filename = subreddit_name + "-posts200-comments100.dat"
in_file = open(in_filename, encoding="utf8")

words = []
lines = in_file.read().splitlines()
lines_count = len(lines)
lines_reduced = []
stop_words = set(stopwords.words('english'))
custom_stop_words = ["[deleted]", "[removed]"]
stop_words = stop_words.union(custom_stop_words)
print("Reducing lines down to identifiable words...")
for line in lines:
    line_reduced = []
    reduced = re.sub(r'[^a-zA-Z\']', ' ', line)
    split = list(reduced.split(' '))
    split = list(dict.fromkeys(split))
    for word in split:
        if len(word) > 1 and word not in stop_words:
            lower = word.lower()
            line_reduced.append(lower)
    if len(line_reduced) >= min_line_len:
        lines_reduced.append(line_reduced)

words = set()
print("Getting list of words used in lines...")
for line in lines_reduced:
    for word in line:
        words.add(word)
words = list(words)
print("Removing duplicate words...")
words = list(dict.fromkeys(words))

print("Writing words to file...")
out_filename = subreddit_name + "_words.txt"
out_file = open(out_filename, "w")
for word in words:
    out_file.write(word + "\n")
out_file.close()

print("Writing reduced lines to file...")
out_filename = subreddit_name + "_lines.txt"
out_file = open(out_filename, "w")
for line in lines_reduced:
    for word in line:
        out_file.write(word + " ")
    out_file.write("\n")

print("All DONE!")

