def words_to_file(word_list, filename):
    out_file = open(filename, "w")
    for word in word_list:
        out_file.write(word + "\n")


def get_word_set(lines):
    words = set()
    for line in lines:
        for word in line:
            if len(word) > 0:
                words.add(word)
    return words


def read_lines(sub_name):
    filename = sub_name + "_lines.txt"
    file = open(filename, "r")
    string_lines = file.read().splitlines()
    file.close()

    list_lines = []
    for line in string_lines:
        split = line.split(" ")
        new_line = []
        for word in split:
            new_line.append(word)
        list_lines.append(new_line)

    return list_lines


def check_word_usage(words, lines):
    words_to_remove = set()
    word_len = len(words)
    current_word = 1
    print("Checking word usage for " + str(word_len) + " words...")
    for word in words:
        if current_word % 1000 == 0 or current_word == 1:
            print("Checking word: \"" + word + "\" (" + str(current_word)
                  + " of " + str(word_len) + ")\n")
        count = 0
        for line in lines:
            if word in line:
                count += 1
        if count < 2 or count == word_len:
            words_to_remove.add(word)
        current_word += 1
    print("Removing " + str(len(words_to_remove)) + " words from word bag.")
    words = words - words_to_remove
    return words


def filter_lines(lines, words):
    new_lines = []
    for line in lines:
        new_line = []
        for word in line:
            if word in words:
                new_line.append(word)
        if len(new_line) > 0:
            new_lines.append(new_line)
    return new_lines


donald_lines = read_lines("The_Donald")
joe_lines = read_lines("JoeBiden")
liz_lines = read_lines("ElizabethWarren")
all_lines = donald_lines + joe_lines + liz_lines
word_set = get_word_set(all_lines)
word_bag = check_word_usage(word_set, all_lines)
words_to_file(word_bag, "word_bag.txt")

print("Donald Lines old count: " + str(len(donald_lines)))
donald_lines = filter_lines(donald_lines, word_bag)
print("Donald Lines new count: " + str(len(donald_lines)))
joe_lines = filter_lines(joe_lines, word_bag)
liz_lines = filter_lines(liz_lines, word_bag)

