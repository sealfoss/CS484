import praw
from praw import exceptions
import prawcore
import re
from nltk.corpus import stopwords
import numpy as np
import pdb
class Analyzer:
    def __init__(self):
        self.running = True

        self.c_zero_lines = None
        self.c_zero_name = None
        self.c_zero_matrix = None

        self.c_one_lines = None
        self.c_one_name = None
        self.c_one_matrix = None

        self.word_bag = None
        self.test_lines = None
        self.greeting = "\nPlease choose from the following options ('q' to quit):\n"
        self.options = ["1)\tChoose subreddit representing classification 0.\n",
                        "2)\tChoose subreddit representing classification 1.\n",
                        "3)\tBuild word bag and training matrices.\n",
                        "4)\tTest a reddit post against training data.\n",
                        "5)\tTest a reddit comment against training data.\n"]

    @staticmethod
    def build_matrix(lines, word_bag):

        row_count = len(lines)
        col_count = len(word_bag)
        matrix = np.zeros((row_count, col_count))

        for i in range(0, row_count):
            line = lines[i]
            for j in range(0, len(line)):
                word = line[j]
                index = word_bag.index(word) #What is this used for than?
                matrix[i,index] += 1

        return matrix
	
	@staticmethod
	def knnAlgo(matrix1,matrix2):
		for matrix1Line in matrix1:
			for matrix2Line in matrix2:
				print("pizza")

    @staticmethod
    def reduce_lines(lines):
        # reddit has some stop words that are unique to it as a platform. this can probably be expanded
        stop_words = set(stopwords.words('english'))
        custom_stop_words = ["[deleted]", "[removed]"]

        # this is arbitrary, and can probably be adjusted for better results
        min_line_len = 1

        # this is the regex used to reduce each line, can be modified or changed
        # this iteration includes the ' symbol
        reduction_regex = r'[^a-zA-Z\']'
        # this iteration includes only letters
        # reduction_regex = r'[^a-zA-Z]'

        lines_reduced = []
        for line in lines:
            line_reduced = []
            reduced = re.sub(reduction_regex, ' ', line)
            split = list(reduced.split(' '))
            split = list(dict.fromkeys(split))
            for word in split:
                if len(word) > 1 and word not in stop_words:
                    lower = word.lower()
                    line_reduced.append(lower)
            if len(line_reduced) >= min_line_len:
                lines_reduced.append(line_reduced)
        words = set()
        for line in lines_reduced:
            for word in line:
                words.add(word)

        return [lines_reduced, words]

    def parse_lines(self):
        if self.c_zero_lines is not None and self.c_one_lines is not None:
            reduced_c_zero_lines = None
            reduced_c_zero_words = None
            reduced_c_one_lines = None
            reduced_c_one_words = None
            keep_going = True

            print("Reducing lines in classification 0...")
            reduced_c_zero = self.reduce_lines(self.c_zero_lines)
            reduced_c_zero_lines = reduced_c_zero[0]
            reduced_c_zero_words = reduced_c_zero[1]
            print("Classification 0 reduced.")

            if len(reduced_c_zero_lines) is 0:
                keep_going = False
                print("No usable lines found in classification 0!")
                print("Please pull more or different comments from reddit for classification 0 and try again.")
            else:
                print("Found " + str(len(reduced_c_zero_lines)) + " usable lines for classification 0.")

            if len(reduced_c_zero_words) is 0:
                keep_going = False
                print("No usable words found in classification 0!")
                print("Please pull more or different comments from reddit for classification 0 and try again.")
            else:
                print("Found " + str(len(reduced_c_zero_words)) + " usable words for classification 0.")

            if keep_going:
                print("Reducing lines in classification 1...")
                reduced_c_one = self.reduce_lines(self.c_one_lines)
                reduced_c_one_lines = reduced_c_one[0]
                reduced_c_one_words = reduced_c_one[1]
                print("Classification 1 reduced")

                if len(reduced_c_one_lines) is 0:
                    keep_going = False
                    print("No usable lines found in classification 1!")
                    print("Please pull more or different comments from reddit for classification 1 and try again.")
                else:
                    print("Found " + str(len(reduced_c_one_lines)) + " usable lines for classification 1.")

                if len(reduced_c_one_words) is 0:
                    keep_going = False
                    print("No usable words found in classification 1!")
                    print("Please pull more or different comments from reddit for classification 1 and try again.")
                else:
                    print("Found " + str(len(reduced_c_one_words)) + " usable words for classification 1.")

            if keep_going:
                self.word_bag = list(reduced_c_one_words.union(reduced_c_zero_words))
                print("Both classification 1 and 2 have been reduced and a bag of words has been defined.")
                print("Building matrices from training data...")
                self.c_zero_matrix = self.build_matrix(reduced_c_zero_lines, self.word_bag)
                self.c_one_matrix = self.build_matrix(reduced_c_one_lines, self.word_bag)
                print("Matrices completed successfully.")
                self.options[2] = self.options[2] + "(Matrices present for existing classification data.)"

        else:
            print("Please select subreddits and associated comments for classifications "
                  "0 and 1 BEFORE attempting to build bag of words.")

    def error_quit(self, error):
        print("ERROR DETECTED: " + error)
        print("Exiting program...")
        self.running = False
        pass

    @staticmethod
    def select_sub():
        sub_name = None
        max_posts = 0
        max_comments = 0

        while sub_name is None or max_posts is 0 or max_comments is 0:
            while sub_name is None:
                print("What is the name of the subreddit you would like to pull comments from?")
                name = str(input())
                if name is not None and 0 < len(name) < 20:
                    sub_name = name
                else:
                    print("Invalid subreddit name, please try again.")
            print("You have chosen /r/" + sub_name + " as your subreddit.")

            while max_posts is 0:
                print(
                    "How many of the top posts from /r/" + sub_name + " would you like to pull from reddit? (n>0)")
                m = int(input())
                if m is not None and m > 0:
                    max_posts = m
                else:
                    print("Please enter an integer value greater than zero.")
            print("You have chosen to pull the top " + str(max_posts) + " posts from /r/" + sub_name + ".")

            while max_comments is 0:
                print("How many of the top comments from these posts in /r/"
                      + sub_name + " would you like to pull from reddit? (n>0)")
                m = int(input())
                if m is not None and m > 0:
                    max_comments = m
                else:
                    print("Please enter an integer value greater than zero.")

            answer = None
            while answer is None:
                print("You have chosen to pull the top " + str(max_comments) + " comments from the top "
                      + str(max_posts) + " posts in /r/" + sub_name + ".")
                print("Is this correct?")
                answer = str(input()).lower()
                if answer == 'y' or answer == 'n':
                    if answer == 'n':
                        sub_name = None
                        max_posts = 0
                        max_comments = 0
                else:
                    print("Invalid input, please answer 'y' or 'n'.")
                    answer = None

        return [sub_name, max_posts, max_comments]

    def pull_comments(self, c0c1):
        selected_sub = self.select_sub()
        sub_name = selected_sub[0]
        max_posts = selected_sub[1]
        max_comments = selected_sub[2]
        pulled = None
        filename = sub_name + ".dat"
        found = True
        in_file = None
        try:
            in_file = open(filename, "r")
        except FileNotFoundError:
            found = False
        if found:
            print("A local file containing previously read comments from /r/" + sub_name + " has been detected.")
            print("Would you like to read comments from file instead of reddit? (y/n)")
            yn = None
            while yn is None:
                answer = str(input())
                if answer is not None and 0 < len(answer) < 2 and (answer == "y" or answer == "n"):
                    yn = answer.lower()
                else:
                    print("Invalid answer, please try again.")
            if yn == "y":
                lines = in_file.read().splitlines()
                sub_name = lines[0]
                pulled = lines[1:]

        if pulled is None:
            try:
                pulled = self.get_from_reddit(sub_name, max_posts, max_comments)
            except praw.exceptions.PRAWException:
                self.error_quit("Reddit doesn't like your sub name.")
                return
            except prawcore.PrawcoreException:
                self.error_quit("Reddit doesn't like your sub name.")
                return

            print("Successfully pulled " + str(len(pulled)) + " comments from /r/" + sub_name + ".\n")
            print("Writing comments to file... ")
            out_file = open(filename, "w")
            if out_file.mode != "w":
                self.error_quit("Cannot write to file \"" + filename + "\".")
                return
            out_file.write(sub_name + "\n")
            for comment in pulled:
                out_file.write(comment + "\n")

        if c0c1:
            self.c_zero_name = sub_name
            self.c_zero_lines = pulled
            self.options[0] = "1)\tChoose subreddit representing classification 0. (Set to " + str(len(pulled)) \
                              + " comments from /r/" + sub_name + ".)\n"
        else:
            self.c_one_name = sub_name
            self.c_one_lines = pulled
            self.options[1] = "2)\tChoose subreddit representing classification 1. (Set to " + str(len(pulled)) \
                              + " comments from /r/" + sub_name + ".)\n"

    def get_from_reddit(self, sub_name, max_top_posts, max_top_comments):
        print("Pulling the top " + str(max_top_comments) + " comments from the top "
              + str(max_top_posts) + " posts in /r/" + sub_name + ".\n")

        c_id = 'R1_WMZ0uXs7OZA'
        c_secret = 'ggr71VKdpYQlL7cYwyT_QAtT4VI'
        p_word = 'mason4lyfe'
        u_agent = 'Comment Pull by Reed'
        u_name = 'cs484_bot'
        reddit = praw.Reddit(client_id=c_id,
                             client_secret=c_secret,
                             password=p_word,
                             user_agent=u_agent,
                             username=u_name)
        subreddit = reddit.subreddit(sub_name)

        out_filename = sub_name + "_p" + str(max_top_posts) + "_c" + str(max_top_comments) + ".dat"
        out_file = open(out_filename, "w")
        if out_file.mode != "w":
            print("Error, unable to write to file \"" + out_filename + "\". Exiting program.")
            self.running = False
            return

        comment_list = []
        count = 1
        for submission in subreddit.top(limit=max_top_posts):
            print("Reading comments from post \"" + submission.title + "\" (" + str(count)
                  + "/" + str(max_top_posts) + "), score: " + str(submission.score))
            submission.comments.replace_more(limit=0)
            submission.comment_sort = 'top'
            comments = submission.comments.list()
            com_count = 0
            for comment in comments:
                body = str(comment.body)
                if body is not None:
                    comment_list.append(body)
                    com_count += 1
                    if com_count >= max_top_comments:
                        break
            count += 1

        return comment_list

    def is_running(self):
        return self.running

    def select_menu_option(self, option):
        print("You have selected menu option \"" + str(option) + "\".\n")
        op_str = str(option).lower()
        op_int = None
        try:
            op_int = int(option)
        except ValueError:
            op_int = None

        if op_str is not None and op_str == 'q':
            self.running = False
        elif op_int is not None:
            if op_int == 1:
                self.pull_comments(True)
            elif op_int == 2:
                self.pull_comments(False)
            elif op_int == 3:
                self.parse_lines()
            elif op_int == 4:
                pass
            elif op_int == 5:
                pass
        else:
            print("Invalid input, please try again.\n")

    def print_menu(self):
        print(self.greeting)
        for o in self.options:
            print(o)


print("\nWelcome to the subreddit analysis program.")
a = Analyzer()
while a.is_running():
    a.print_menu()
    option = input()
    a.select_menu_option(option)
	pdb.set_trace()
print("Thanks for using the reddit analysis program!\n")
quit(0)
