in_filename = "donald_garbage.txt"
in_file = open(in_filename, "r")

words = []
splits = []
lines = in_file.read().splitlines()
lines_count = len(lines)
for line in lines:
    split = list(line.split(' '))
    split = list(dict.fromkeys(split))
    splits.append(split)
    for word in split:
        words.append(word)

words = list(dict.fromkeys(words))
