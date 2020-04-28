import praw
from praw import exceptions
import prawcore
import re
from nltk.corpus import stopwords
import numpy as np
import math
from random import randrange as rr


class Analyzer:
    def __init__(self):
        self.c_id = 'R1_WMZ0uXs7OZA'
        self.c_secret = 'ggr71VKdpYQlL7cYwyT_QAtT4VI'
        self.p_word = 'mason4lyfe'
        self.u_agent = 'Comment Pull by Reed'
        self.u_name = 'cs484_bot'
        self.running = True
        self.initial_k = 7

        self.test_url = None
        self.test_lines = None
        self.test_post_comments_count = 0
        self.test_post_line_count = 0
        self.test_matrix = None
        self.min_post_line_len = 5

        self.train_matrix = None
        self.train_matrix_len = 0

        self.tfidf = None
        self.tfidf_normalized = None

        self.c_zero_lines = None
        self.c_zero_name = None
        self.c_zero_matrix = None

        self.c_one_lines = None
        self.c_one_name = None
        self.c_one_matrix = None

        self.word_bag = None
        self.word_bag_len = 0
        self.test_lines = None
        self.greeting = "\nPlease choose from the following options ('q' to quit):\n"
        self.options = ["1)\tChoose subreddit representing classification 0.\n",
                        "2)\tChoose subreddit representing classification 1.\n",
                        "3)\tBuild word bag and training matrices.\n",
                        "4)\tChoose a reddit post to pull comments from for testing.\n",
                        "5)\tClassify comments from chosen reddit post based on training data.\n"]

    def scale_by_tfidf(self):
        self.train_matrix = np.append(self.c_zero_matrix, self.c_one_matrix, axis=0)
        self.train_matrix_len = len(self.train_matrix)
        all_words_count = 0
        for i in range(0, self.train_matrix_len):
            all_words_count += int(np.sum(self.train_matrix[i]))
        self.tfidf = np.zeros(self.word_bag_len)
        total_times_used = np.sum(self.train_matrix, axis=0)
        for i in range(0, self.word_bag_len):
            tf = total_times_used[i] / float(all_words_count)
            lines_with_word = 0
            for j in range(0, self.train_matrix_len):
                if self.train_matrix[j, i] > 0:
                    lines_with_word += 1
            idf = 0
            if lines_with_word > 0:
                idf = math.log(self.train_matrix_len / float(lines_with_word))
            self.tfidf[i] = tf * idf
        tfidf_mag = np.linalg.norm(self.tfidf)
        self.tfidf_normalized = self.tfidf / tfidf_mag
        self.train_matrix *= self.tfidf_normalized
        self.test_matrix *= self.tfidf_normalized

    # this is literally copied and pasted.
    # try to keep in mind that this function was written as a way to spread the work load over multiple processes
    # kayy is your value k, given a dumb name to differentiate between other instances of that value in the program
    # training_m_proc is the training data matrix
    # test_slice is the slice of test matrix given to this process to work on.
    # grades_copy are the grades for each line of the training matrix, -1 or 1.
    # results is an array for storing test results in and sending back to the main process.
    # conn is a device for synchronizing processes
    @staticmethod
    def run_comparison_cos(kay, train_m, test_m, grades):
        results = []
        percent_done = 0
        test_len = test_m.shape[0]
        print("Comparing " + str(test_len) + " vectors to training matrix of shape "
              + str(train_m.shape) + " by cosine,  k = " + str(kay))
        test_count = 0
        for test_vec in test_m:
            new_percent = (100 * test_count) / test_len
            if (new_percent - percent_done) > 1:
                percent_done = int(new_percent)
                print("Progress: " + str(percent_done) + "% finished (" + str(test_count) + "/" + str(test_len) + ")")
            k_grades = np.zeros((kay,), dtype=int)
            all_cosines = list()
            test_vec_mag = np.linalg.norm(test_vec)
            for training_vec in train_m:
                dot = np.dot(test_vec, training_vec)
                training_vec_mag = np.linalg.norm(training_vec)
                mag_prod = test_vec_mag * training_vec_mag
                cos = 0
                if dot != 0 and mag_prod != 0:
                    cos = dot / mag_prod
                all_cosines.append(cos)
            all_cosines_vec = np.asarray(all_cosines)
            for i in range(0, kay):
                max_index = np.argmax(all_cosines_vec)
                k_grades[i] = grades[max_index]
                all_cosines_vec[max_index] = 0

            k_sum = np.sum(k_grades)
            if k_sum >= 0:
                results.append("1")
            else:
                results.append("-1")
            test_count += 1
        return results

    @staticmethod
    def build_matrix(lines, word_bag):

        row_count = len(lines)
        col_count = len(word_bag)
        matrix = np.zeros((row_count, col_count))

        for i in range(0, row_count):
            line = lines[i]
            for j in range(0, len(line)):
                word = line[j]
                index = word_bag.index(word)
                matrix[i, index] += 1

        return matrix

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
        if self.c_zero_lines is None or self.c_one_lines is None:
            print("You must choose subreddit comments for classifications 0 and 1 before building "
                  "a word bag and training matrices from them.")
            return

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
                self.word_bag_len = len(self.word_bag)
                print("Both classification 1 and 2 have been reduced and a bag of words (size " + str(self.word_bag_len)
                      + ") has been defined.")
                print("Building matrices from training data...")
                self.c_zero_matrix = self.build_matrix(reduced_c_zero_lines, self.word_bag)
                self.c_one_matrix = self.build_matrix(reduced_c_one_lines, self.word_bag)
                print("Matrices completed successfully.")
                self.options[2] = "3)\tBuild word bag and training matrices. (Training matrices built successfully.)\n"

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
        filename = sub_name + "_p" + str(max_posts) + "_c" + str(max_comments) + ".dat"
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
                pulled = self.get_top_comments(sub_name, max_posts, max_comments)
            except praw.exceptions.PRAWException as err:
                self.error_quit("Reddit doesn't like your sub name: " + str(err))
                return
            except prawcore.PrawcoreException as err:
                self.error_quit("Reddit doesn't like your sub name: " + str(err))
                return

            print("Successfully pulled " + str(len(pulled)) + " comments from /r/" + sub_name + ".\n")
            print("Writing comments to file... ")
            out_file = open(filename, "w")
            if out_file.mode != "w":
                self.error_quit("Cannot write to file \"" + filename + "\".")
                return
            out_file.write(sub_name + "\n")
            for comment in pulled:
                no_newline = comment.replace('\n', ' ')
                out_file.write(no_newline + "\n")

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

    def get_reddit(self):
        reddit = praw.Reddit(client_id=self.c_id,
                             client_secret=self.c_secret,
                             password=self.p_word,
                             user_agent=self.u_agent,
                             username=self.u_name)
        return reddit

    def get_top_comments(self, sub_name, max_top_posts, max_top_comments):
        print("Pulling the top " + str(max_top_comments) + " comments from the top "
              + str(max_top_posts) + " posts in /r/" + sub_name + ".\n")

        reddit = self.get_reddit()
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

    def get_post_comments(self):
        if self.c_zero_matrix is None or self.c_one_matrix is None:
            print("A word bag and training matrix must be built for classifications 0 and 1 before choosing "
                  "a reddit post comments section to classify.")
            return
        # first, get comments from a given post
        reddit = self.get_reddit()
        max_comments = None
        input_url = None
        comment_list = []
        post = None
        while post is None:
            while input_url is None:
                input_url = input("What is the URL of the post you would like to pull comments from?")
            while max_comments is None:
                num_in = input("What is the max amount of comments you want to pull from this post?")
                try:
                    max_comments = int(num_in)
                except ValueError:
                    print("Please enter a valid integer value.")
            try:
                post = reddit.submission(url=input_url)
            except praw.exceptions.PRAWException as err:
                print("Reddit doesn't like the URL you gave it: " + str(err))
                print("Please try again.")
                post = None
                input_url = None
                max_comments = None
            except prawcore.PrawcoreException as err:
                print("Reddit doesn't like the URL you gave it: " + str(err))
                print("Please try again.")
                post = None
                input_url = None
                max_comments = None
        self.test_url = input_url
        post.comments.replace_more(limit=0)
        post.comment_sort = 'top'
        comments = post.comments.list()
        com_count = 0
        for comment in comments:
            body = str(comment.body)
            if body is not None:
                comment_list.append(body)
                com_count += 1
                if com_count >= max_comments:
                    break
        self.test_post_comments_count = com_count
        print("Successfully pulled " + str(com_count) + " comments from post \"" + post.title + "\".")

        # reduce the test lines in the same manner as we did w/ training lines
        reduced_and_words = self.reduce_lines(comment_list)
        reduced_test_lines = reduced_and_words[0]
        # then go through the reduced lines and remove any words that aren't in training data
        testable_lines = []
        for reduced_line in reduced_test_lines:
            new_line = []
            for word in reduced_line:
                if word in self.word_bag:
                    new_line.append(word)
            # if a line doesn't include enough words used in training data, omit it
            if len(new_line) > self.min_post_line_len:
                testable_lines.append(new_line)
                # print(new_line)
        self.test_post_line_count = len(testable_lines)
        print("Out of those comments, " + str(self.test_post_line_count)
              + " were found to be classifiable, based on training data.")
        print("Building test matrix from classifiable comments...")
        # build the actual test matrix
        self.test_matrix = np.zeros((self.test_post_line_count, len(self.word_bag)))
        for i in range(0, len(testable_lines)):
            line = testable_lines[i]
            for j in range(0, len(line)):
                word = line[j]
                index = self.word_bag.index(word)
                self.test_matrix[i, index] += 1
        print("Scaling test and training matrices by TF/IDF...")
        self.scale_by_tfidf()
        print("Test and training matrices built and ready for classification.")
        # all done
        self.options[3] = "4)\tChoose a reddit post to pull comments from for testing. (" \
                          + str(self.test_post_line_count) + " classifiable comments pulled from reddit.)\n"

    def run_comparison(self):
        if self.c_zero_matrix is None or self.c_one_matrix is None or self.word_bag is None or self.test_matrix is None:
            print("Please designate classifications 0 and 1, and a reddit post for classification.")
            return
        print("Welcome to the classification part of the program.")
        k = self.initial_k
        cross_val_loops = 0
        while cross_val_loops < 1:
            print("How many times would you like to cross validate for a better value k?")
            loops_in = input("Please enter an integer value greater than zero "
                             "(keep it small or you'll be here all day).")
            try:
                cross_val_loops = int(loops_in)
            except ValueError:
                print("Invalid input, please try again.")
                cross_val_loops = 0
        print("Finding best value for k via cross validation. This will execute " + str(cross_val_loops) + " times.")
        k_results = []
        for i in range(0, cross_val_loops):
            print("Executing cross validation loop #" + str((i+1)) + " of " + str(cross_val_loops) + ".")
            print("Slicing training array into 50/50 random sampling...")
            # get random sampling from half of training data
            random_c0_picks = self.pick_random_half(self.c_zero_matrix)
            random_c1_picks = self.pick_random_half(self.c_one_matrix)
            c0_train_slice = np.zeros((len(random_c0_picks[0]), self.word_bag_len))
            c1_train_slice = np.zeros((len(random_c1_picks[0]), self.word_bag_len))
            c0_test_slice = np.zeros((len(random_c0_picks[0]), self.word_bag_len))
            c1_test_slice = np.zeros((len(random_c1_picks[0]), self.word_bag_len))

            c0_train_slice_len = len(c0_train_slice)
            c0_test_slice_len = len(c0_test_slice)
            c1_train_slice_len = len(c1_train_slice)
            c1_test_slice_len = len(c1_test_slice)
            train_grades_len = c0_train_slice_len + c1_train_slice_len
            train_grades = np.zeros(train_grades_len)
            test_grades_len = c0_test_slice_len + c1_test_slice_len
            test_grades = np.zeros(test_grades_len)
            # -1 for c0, 1 or c1
            for j in range(0, c0_train_slice_len):
                pick = random_c0_picks[0][j]
                c0_train_slice[j] = self.c_zero_matrix[pick].copy()
                train_grades[j] = -1
            for j in range(0, c0_test_slice_len):
                not_picked = random_c0_picks[1][j]
                c0_test_slice[j] = self.c_zero_matrix[not_picked].copy()
                test_grades[j] = -1
            for j in range(0, c1_train_slice_len):
                pick = random_c1_picks[0][j]
                c1_train_slice[j] = self.c_one_matrix[pick].copy()
                train_grades[j+c0_train_slice_len] = 1
            for j in range(0, c1_test_slice_len):
                not_picked = random_c0_picks[1][j]
                c0_test_slice[j] = self.c_zero_matrix[not_picked].copy()
                test_grades[j+c0_test_slice_len] = 1
            # combine training slices into single training and testing matrices
            train_slice = np.append(c0_train_slice, c1_train_slice, axis=0)
            test_slice = np.append(c0_test_slice, c1_test_slice, axis=0)
            print("Running cosine classification on designated testing half of training data using k=" + str(k) + "...")
            results = self.run_comparison_cos(k, train_slice, test_slice, train_grades)
            num_correct = 0
            for j in range(0, len(results)):
                if results[j] == test_grades[j]:
                    num_correct += 1
            accuracy = 100 * num_correct / len(test_slice)
            print("Found accuracy rate of " + str(accuracy) + "% for k=" + str(k) + ".")
            k_results.append([k, accuracy])
            k = self.next_prime(k)

        best_k = 0
        best_accuracy = 0
        for k_result in k_results:
            k = k_result[0]
            accuracy = k_result[1]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
        print("Cross validation found k=" + str(best_k) + " to be the best value for k, with an accuracy rating of "
              + str(best_accuracy) + "%.")
        print("Running classification of full test data with k=" + str(best_k) + "...")
        grades = []
        c0_len = len(self.c_zero_matrix)
        c1_len = len(self.c_one_matrix)
        for i in range(0, c0_len):
            grades.append(-1)
        for i in range(c0_len, (c0_len + c1_len)):
            grades.append(1)
        results = self.run_comparison_cos(best_k, self.train_matrix, self.test_matrix, grades)

        print("Classification complete. Generating report...")
        num_c0 = 0
        num_c1 = 0
        for i in range(0, len(results)):
            if results[i] == -1:
                num_c0 += 1
            else:
                num_c1 += 1
        similarity_c0 = num_c0 * 100 / c0_len
        similarity_c1 = 100 - similarity_c0
        print("The classifier found " + str(num_c0) + " (" + str(similarity_c0) + "%)" + " of the " + str(c0_len)
              + " classifiable comments to be similar to comments found in /r/" + self.c_zero_name + ".")
        print("The classifier found " + str(num_c1) + " (" + str(similarity_c1) + "%)" + " of the " + str(c1_len)
              + " classifiable comments to be similar to comments found in /r/" + self.c_one_name + ".")

    def next_prime(self, n):
        prime = n+1
        while not self.is_prime(prime):
            prime += 1
        return prime

    @staticmethod
    def is_prime(n):
        # only works with positive numbers
        prime = True
        if n <= 3:
            prime = True
        elif n % 2 == 0:
            prime = False
        elif n < 9:
            prime = True
        else:
            half_n = int(n/2)
            for i in range(3, half_n):
                if n % i == 0:
                    prime = False
                    break
        return prime

    @staticmethod
    def pick_random_half(matrix):
        m_len = len(matrix)
        picks = np.arange(0, m_len)
        picks = list(picks)
        not_picked = []
        while len(picks) >= (m_len/2):
            ri = rr(len(picks))
            popped = picks.pop(ri)
            not_picked.append(popped)
        return [picks, not_picked]

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
                self.get_post_comments()
            elif op_int == 5:
                self.run_comparison()
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
print("Thanks for using the reddit analysis program!\n")
quit(0)
