import praw
from praw import exceptions
import prawcore
import re
from nltk.corpus import stopwords
import nltk
import numpy as np
import math
from random import randrange as rr
import time


class Analyzer:
    def __init__(self):
        # words for sentiment analysis
        self.words_pos = None
        self.words_neg = None
        self.sentiments = None

        # this is the regex used to reduce each line, can be modified or changed
        # this iteration includes the ' symbol
        self.reduction_regex = r'[^a-zA-Z\']'
        # this iteration includes only letters
        # reduction_regex = r'[^a-zA-Z]'

        self.c_id = 'R1_WMZ0uXs7OZA'
        self.c_secret = 'ggr71VKdpYQlL7cYwyT_QAtT4VI'
        self.p_word = 'mason4lyfe'
        self.u_agent = 'Comment Pull by Reed'
        self.u_name = 'cs484_bot'
        self.running = True
        self.initial_k = 3

        self.test_url = None
        self.test_lines = None
        self.test_post_comments_count = 0
        self.test_post_line_count = 0
        self.test_matrix = None
        self.min_post_line_len = 5
        self.best_k = None

        self.train_matrix = None
        self.train_matrix_len = 0

        self.tfidf = None
        self.tfidf_normalized = None

        self.c0_lines = None
        self.c0_name = None
        self.c0_matrix = None

        self.c1_lines = None
        self.c1_name = None
        self.c1_matrix = None

        self.train_grades = None
        self.word_bag = None
        self.word_bag_len = 0
        self.test_lines = None

        self.greeting = "\nPlease choose from the following options ('q' to quit):\n"
        self.options = ["1)\tChoose subreddit representing classification 0.\n",
                        "2)\tChoose subreddit representing classification 1.\n",
                        "3)\tBuild word bag and training matrices.\n",
                        "4)\tChoose a reddit post to pull comments from for testing.\n",
                        "5)\tClassify comments from chosen reddit post based on training data.\n"]

    def read_words(self):
        self.words_pos = []
        self.words_neg = []
        filename_pos = "words_positive.txt"
        filename_neg = "words_negative.txt"
        pos_file = None
        neg_file = None
        try:
            pos_file = open(filename_pos, "r")
            self.words_pos = pos_file.read().splitlines()
            pos_file.close()
        except FileNotFoundError:
            print("Could not locate words_positive.txt")
        finally:
            if pos_file is not None:
                pos_file.close()
        try:
            neg_file = open(filename_neg, "r")
            self.words_neg = neg_file.read().splitlines()
        except FileNotFoundError:
            print("Could not locate words_negative.txt")
        finally:
            if neg_file is not None:
                neg_file.close()

    @staticmethod
    def get_tfidf(train_matrix, word_bag):
        train_matrix_len = len(train_matrix)
        word_bag_len = len(word_bag)
        print("Generating TF/IDF for matrix of size " + str(train_matrix_len) + " by " + str(word_bag_len) + ".")
        all_words_count = 0
        for i in range(0, train_matrix_len):
            all_words_count += int(np.sum(train_matrix[i]))
        tfidf = np.zeros(word_bag_len)
        total_times_used = np.sum(train_matrix, axis=0)
        for i in range(0, word_bag_len):
            tf = total_times_used[i] / all_words_count
            lines_with_word = 0
            for j in range(0, train_matrix_len):
                if train_matrix[j, i] > 0:
                    lines_with_word += 1
            idf = 0
            if lines_with_word > 0:
                idf = math.log(train_matrix_len / float(lines_with_word))
            tfidf[i] = tf * idf
        tfidf_magnitude = np.linalg.norm(tfidf)
        tfidf_normalized = tfidf / tfidf_magnitude
        return tfidf_normalized

    def scale_by_tfidf(self):
        self.train_matrix = np.append(self.c0_matrix, self.c1_matrix, axis=0)
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
                # results.append("1")
                results.append(1)
            else:
                # results.append("-1")
                results.append(-1)
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

    def reduce_lines(self, lines):
        # reddit has some stop words that are unique to it as a platform. this can probably be expanded
        stop_words = set(stopwords.words('english'))
        custom_stop_words = ["[deleted]", "[removed]", "deleted", "removed"]

        # this is arbitrary, and can probably be adjusted for better results
        min_line_len = 1

        lines_reduced = []
        for line in lines:
            line_reduced = []
            reduced = re.sub(self.reduction_regex, ' ', line)
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
        if self.c0_lines is None or self.c1_lines is None:
            print("You must choose subreddit comments for classifications 0 and 1 before building "
                  "a word bag and training matrices from them.")
            return

        if self.c0_lines is not None and self.c1_lines is not None:
            reduced_c0_lines = None
            reduced_c0_words = None
            reduced_c1_lines = None
            reduced_c1_words = None
            keep_going = True

            print("Reducing lines in classification 0...")
            reduced_c0 = self.reduce_lines(self.c0_lines)
            reduced_c0_lines = reduced_c0[0]
            reduced_c0_words = reduced_c0[1]
            print("Classification 0 reduced.")

            if len(reduced_c0_lines) is 0:
                keep_going = False
                print("No usable lines found in classification 0!")
                print("Please pull more or different comments from reddit for classification 0 and try again.")
            else:
                print("Found " + str(len(reduced_c0_lines)) + " usable lines for classification 0.")

            if len(reduced_c0_words) is 0:
                keep_going = False
                print("No usable words found in classification 0!")
                print("Please pull more or different comments from reddit for classification 0 and try again.")
            else:
                print("Found " + str(len(reduced_c0_words)) + " usable words for classification 0.")

            if keep_going:
                print("Reducing lines in classification 1...")
                reduced_c1 = self.reduce_lines(self.c1_lines)
                reduced_c1_lines = reduced_c1[0]
                reduced_c1_words = reduced_c1[1]
                print("Classification 1 reduced")

                if len(reduced_c1_lines) is 0:
                    keep_going = False
                    print("No usable lines found in classification 1!")
                    print("Please pull more or different comments from reddit for classification 1 and try again.")
                else:
                    print("Found " + str(len(reduced_c1_lines)) + " usable lines for classification 1.")

                if len(reduced_c1_words) is 0:
                    keep_going = False
                    print("No usable words found in classification 1!")
                    print("Please pull more or different comments from reddit for classification 1 and try again.")
                else:
                    print("Found " + str(len(reduced_c1_words)) + " usable words for classification 1.")

            if keep_going:
                self.word_bag = list(reduced_c1_words.union(reduced_c0_words))
                self.word_bag_len = len(self.word_bag)
                print("Both classification 1 and 2 have been reduced and a bag of words (size " + str(self.word_bag_len)
                      + ") has been defined.")

                print("Building matrices from training data...")
                self.c0_matrix = self.build_matrix(reduced_c0_lines, self.word_bag)
                self.c1_matrix = self.build_matrix(reduced_c1_lines, self.word_bag)
                matrix_len = min(len(self.c0_matrix), len(self.c1_matrix))
                print("Matrices built.")

                print("Generating grades array...")
                grades_c0 = [-1] * matrix_len
                grades_c1 = [1] * matrix_len
                grades_c0.extend(grades_c1)
                self.train_grades = grades_c0
                print("Grades array completed, length: " + str(len(self.train_grades)))

                print("Combining matrices into one, monolithic training matrix...")
                self.train_matrix = np.append(self.c0_matrix[0:matrix_len], self.c1_matrix[0:matrix_len], axis=0)
                print("Matrices combined into master training matrix with shape: (" + str(len(self.train_matrix)) + ", "
                      + str(len(self.word_bag)) + ")")

                # building TF/IDF takes a long time. Why not read/write from file?
                loaded_from_file = None
                filename = self.c0_name + "_" + self.c1_name + "_len" + str(matrix_len) + "_w"\
                           + str(self.word_bag_len) + "_tfidf.npy"
                read_file = None
                try:
                    read_file = open(filename, "rb")
                    print("Previously generated TF/IDF matching read data detected.")
                    while loaded_from_file is None:
                        text_in = input("Would you like to load from file instead of generating "
                                        "again from scratch? (y/n)")
                        if 0 < len(text_in) < 2:
                            char_in = text_in[0].lower()
                            if char_in == 'y':
                                self.tfidf = np.load(filename)
                                loaded_from_file = True
                            elif char_in == 'n':
                                loaded_from_file = False
                        if loaded_from_file is None:
                            print("Invalid input, please try again.")
                except IOError:
                    print("No file containing previously computed TF/IDF detected.")
                    loaded_from_file = False
                finally:
                    if read_file is not None:
                        read_file.close()


                if not loaded_from_file:
                    print("Generating TF/IDF from training data...")
                    self.tfidf = self.get_tfidf(self.train_matrix, self.word_bag)
                    print("Writing TF/IDF to file...")
                    write_file = open(filename, "wb")
                    try:
                        np.save(filename, self.tfidf)
                        print("TF/IDF written to file successfully.")
                    except IOError:
                        print("Error writing TF/IDF to file. You will have to build from scratch next time...")

                print("Scaling training matrix by TF/IDF...")
                self.train_matrix *= self.tfidf
                print("Training matrix built successfully.")
                self.options[2] = "3)\tBuild word bag and training matrices. (Training matrix built successfully.)\n"
        else:
            print("Please select subreddits and associated comments for classifications "
                  "0 and 1 BEFORE attempting to build bag of words.")

    def error_quit(self, error):
        print("ERROR DETECTED: " + error)
        print("Exiting program...")
        self.running = False
        pass
    
    @staticmethod
    def provide_input(word_length,is_it_int):
        user_input = ""
        if is_it_int:
            try:
                user_input = int(input())   
            except ValueError:
                print("Invalid input, setting posts to 5...")
                user_input = 5

            if user_input is not None and user_input > 0:
                return user_input 
            return 0
    
        else:
            user_input = str(input())
            if user_input is not None and 0 < len(user_input) < word_length:
                return user_input
            return None
    
         

    @staticmethod
    def select_sub():
        sub_name = None
        num_posts = 0
        num_comments = 0
        
        while sub_name is None or num_posts == 0 or num_comments == 0:
            while sub_name is None:

                print("What is the name of the subreddit you would like to pull comments from?")
                sub_name = Analyzer.provide_input(20,False)
                
                if sub_name is None:
                    print("Invalid subreddit name, please try again.") 
            
            text_one = "You have chosen /r/{subreddit} as your subreddit."

            print(text_one.format(subreddit = sub_name))

            while num_posts is 0:
               #print("How many of the top posts from /r/" + sub_name + " would you like to pull from reddit? (n>0)")
                text_two = "How many of the top posts from /r/{subreddit} would you like to pull from reddit? (n>0)"
                print(text_two.format(subreddit=sub_name))

                num_posts = Analyzer.provide_input(20,True)

                if num_posts == 0:
                    print("Please enter an integer value greater than zero.")
            #print("You have chosen to pull the top " + str(num_posts) + " posts from /r/" + sub_name + ".")

            text_three = "You have chose to pull the top {number_of_posts} posts from /r/{subreddit}."
            print(text_three.format(number_of_posts = num_posts, subreddit = sub_name))


            while num_comments is 0:
                #print("How many of the top comments from these posts in /r/" + sub_name + " would you like to pull from reddit? (n>0)")

                text_four = "How many of the top comments from these posts in /r/{subreddit} would you like to pull from reddit? (n>0)"
                print(text_four.format(subreddit = sub_name))
                num_comments = Analyzer.provide_input(20,True)
                if num_comments == 0:
                    print("Please enter an integer value greater than zero.")

            answer = None
            while answer is None:
                #print("You have chosen to pull the top " + str(num_comments) + " comments from the top "
                #      + str(num_posts) + " posts in /r/" + sub_name + ".")
                text_five = "You have chose to pull the top {number_of_comments} comments from the top {number_of_posts} posts in /r/{subreddit}."

                print(text_five.format(number_of_comments = num_comments,number_of_posts = num_posts, subreddit = sub_name))
                print("Is this correct?")
                answer = str(input()).lower()
                if answer == 'y' or answer == 'n':
                    if answer == 'n':
                        sub_name = None
                        num_posts = 0
                        num_comments = 0
                else:
                    print("Invalid input, please answer 'y' or 'n'.")
                    answer = None

        return [sub_name, num_posts, num_comments]

    def pull_comments(self, c0c1):
        selected_sub = self.select_sub()
        sub_name = selected_sub[0]
        num_posts = selected_sub[1]
        num_comments = selected_sub[2]
        pulled = None
        filename = sub_name + "_p" + str(num_posts) + "_c" + str(num_comments) + ".dat"
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
                pulled = self.get_top_comments(sub_name, num_posts, num_comments)
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
                reduced = re.sub(self.reduction_regex, ' ', comment)
                out_file.write(reduced + "\n")

        if c0c1:
            self.c0_name = sub_name
            self.c0_lines = pulled
            self.options[0] = "1)\tChoose subreddit representing classification 0. (Set to " + str(len(pulled)) \
                              + " comments from /r/" + sub_name + ".)\n"
        else:
            self.c1_name = sub_name
            self.c1_lines = pulled
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
        if self.c0_matrix is None or self.c1_matrix is None:
            print("A word bag and training matrix must be built for classifications 0 and 1 before choosing "
                  "a reddit post comments section to classify.")
            return
        # first, get comments from a given post
        reddit = self.get_reddit()
        num_comments = None
        input_url = None
        comment_list = []
        post = None
        while post is None:
            while input_url is None:
                input_url = input("What is the URL of the post you would like to pull comments from?")
            while num_comments is None:
                num_in = input("What is the max amount of comments you want to pull from this post?")
                try:
                    num_comments = int(num_in)
                except ValueError:
                    print("Please enter a valid integer value.")
            try:
                post = reddit.submission(url=input_url)
            except praw.exceptions.PRAWException as err:
                print("Reddit doesn't like the URL you gave it: " + str(err))
                print("Please try again.")
                post = None
                input_url = None
                num_comments = None
            except prawcore.PrawcoreException as err:
                print("Reddit doesn't like the URL you gave it: " + str(err))
                print("Please try again.")
                post = None
                input_url = None
                num_comments = None
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
                if com_count >= num_comments:
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
        print("Extracting sentiment from each classifiable comment...")
        self.extract_sentiments(testable_lines)
        print("Building test matrix from classifiable comments...")
        # build the actual test matrix
        self.test_matrix = np.zeros((self.test_post_line_count, len(self.word_bag)))
        for i in range(0, len(testable_lines)):
            line = testable_lines[i]
            for j in range(0, len(line)):
                word = line[j]
                index = self.word_bag.index(word)
                self.test_matrix[i, index] += 1
        print("Scaling test matrix by TF/IDF...")
        self.test_matrix *= self.tfidf
        print("Test matrix built and ready for classification.")
        # all done
        self.options[3] = "4)\tChoose a reddit post to pull comments from for testing. (" \
                          + str(self.test_post_line_count) + " classifiable comments pulled from reddit.)\n"

    def extract_sentiments(self, comments):
        sentiments = []
        pos_tags = ['JJ','JJR','JJS','RBR','RBS','RB','VB','VBD','VBG','VBN','VBZ']
        for comment in comments:
            pos_comment = nltk.pos_tag(comment)
            score = 0
            for word in pos_comment:
                if word[1] in pos_tags:
                    if word[0] in self.words_pos:
                        score += 1
                    if word[0] in self.words_neg:
                        score -= 1
            sentiments.append(score)
        self.sentiments = sentiments

    @staticmethod
    def get_random_split(n):
        picked = list(range(n))
        not_picked = []
        while len(picked) > n/2:
            rand = rr(len(picked))
            not_picked.append(picked.pop(rand))
        return [picked, not_picked]

    @staticmethod
    def get_accuracy(results, answers, total):
        if len(results) != total or len(answers) != total:
            print("You can't get accuracy from mismatched results and answers lists!")
            print(str(len(results)))
            print(str(len(answers)))
            print(total)
            # return None
        correct = 0
        incorrect = 0
        for i in range(0, total):
            result = results[i]
            answer = answers[i]
            if result == answer:
                correct += 1
            else:
                incorrect += 1
        percent_correct = 100.0 * float(correct) / float(total)
        percent_incorrect = 100.0 * float(incorrect) / float(total)
        accuracy = [correct, incorrect, percent_correct, percent_incorrect]
        return accuracy

    @staticmethod
    def log_item(string_to_write):
        with open("LOG.TXT","a+") as log_file:
            log_file.write(string_to_write)


    def cross_validate(self, n, initial_k, train_matrix, grades, word_bag):
        start_time = time.time()
        k = initial_k
        matrix_len = len(train_matrix)
        word_bag_len = len(word_bag)
        best_accuracy = 0.0
        best_k = 0
        # open a file to write the log
        log_file = open("LOG.TXT","w")
        for i in range(0, n):
            print("Running cross-validation " + str(i+1) + " of " + str(n) + ", k=" + str(k))
            first_str = "Running cross-validation " + str(i+1) + " of " + str(n) + ", k=" + str(k) + "\n" 
            Analyzer.log_item(first_str)
            random_split = self.get_random_split(matrix_len)
            train_picks = random_split[0]
            test_picks = random_split[1]
            train_slice_len = len(train_picks)
            test_slice_len = len(test_picks)
            train_slice = np.zeros((train_slice_len, word_bag_len))
            test_slice = np.zeros((test_slice_len, word_bag_len))
            train_slice_grades = []
            test_slice_grades = []
            for j in range(0, train_slice_len):
                pick = train_picks[j]
                vec = train_matrix[pick]
                grade = grades[pick]
                train_slice[j] = vec.copy()
                train_slice_grades.append(grade)
            for j in range(0, test_slice_len):
                pick = test_picks[j]
                vec = train_matrix[pick]
                grade = grades[pick]
                test_slice[j] = vec.copy()
                test_slice_grades.append(grade)
            results = self.run_comparison_cos(k, train_slice, test_slice, train_slice_grades)
            results_v_grades = self.get_accuracy(results, test_slice_grades, len(results))
            accuracy = results_v_grades[2]
            print("Found accuracy of " + str(accuracy) + "% for k=" + str(k) + ".")
            second_str = "Found accuracy of " + str(accuracy) + "% for k=" + str(k) + ".\n"
            Analyzer.log_item(second_str)

            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_k = k
            k = self.next_prime(k)
        
        print("Best value 'k' found to be " + str(best_k) + ", with an accuracy rating of " + str(best_accuracy) + "%.")
        third_str = "Best value 'k' found to be " + str(best_k) + ", with an accuracy rating of "\
                    + str(best_accuracy) + "%.\n"
        validation_time = time.time() - start_time
        print("Cross validation took " + str(validation_time) + " seconds to complete.")
        Analyzer.log_item(third_str)
        return best_k

    def classify_test_data(self):
        if self.c0_matrix is None or self.c1_matrix is None or self.word_bag is None or self.test_matrix is None:
            print("Please designate classifications 0 and 1, and a reddit post for classification.")
            return
        print("Welcome to the classification part of the program.")
        k = None
        while k is None:
            print("Would you like to provide a value for k, or cross validate for one?")
            answer = input("Please enter some integer value for k, or 0 to cross validate.")
            try:
                new_k = int(answer)
                k = new_k
            except ValueError as e:
                print("Invalid input, please try again.")
        if k > 0:
            self.best_k = k
        else:
            start_k = 0
            while start_k < 1:
                print("Cross validator will start with a given value for k, "
                      "and increment to the next prime number at every iteration.")
                start_in = input("What value for k would you like the validator to start with?")
                try:
                    start_k = int(start_in)
                    self.initial_k = start_k
                except ValueError:
                    print("Invalid input, please try again.")
            cross_validations = 0
            while cross_validations < 1:
                print("How many times would you like to cross validate for a better value k?")
                loops_in = input("Please enter an integer value greater than zero. "
                                 "You may want to keep it small, or else you'll be here all day.")
                try:
                    cross_validations = int(loops_in)
                except ValueError:
                    print("Invalid input, please try again.")
                    cross_validations = 0
            print("Finding best value for k via cross validation. Validator will begin with k=" + str(start_k) +
                  " and will execute " + str(cross_validations) + " times.")
            self.best_k = self.cross_validate(cross_validations, self.initial_k, self.train_matrix,
                                              self.train_grades, self.word_bag)
        start_time = time.time()
        print("Running classification on test data...")
        results = self.run_comparison_cos(self.best_k, self.train_matrix, self.test_matrix, self.train_grades)
        print("Classification complete. Generating report...")
        results_len = len(results)
        num_c0 = 0
        num_c1 = 0
        for i in range(0, results_len):
            r = results[i]
            if r == -1:
                num_c0 += 1
            else:
                num_c1 += 1
        similarity_c0 = float(num_c0) * 100.0 / float(results_len)
        similarity_c1 = float(num_c1) * 100.0 / float(results_len)
        sentiment_pos = 0
        sentiment_neg = 0
        for sentiment in self.sentiments:
            if sentiment > 0:
                sentiment_pos += 1
            elif sentiment < 0:
                sentiment_neg += 1
        print("The classifier found " + str(num_c0) + " of the " + str(len(results)) + " classifiable comments "
              + " (" + str(similarity_c0) + "%)" + " to be similar to comments found in /r/" + self.c0_name + ".")
        print("The classifier found " + str(num_c1) + " of the " + str(len(results)) + " classifiable comments "
              + " (" + str(similarity_c1) + "%)" + " to be similar to comments found in /r/" + self.c1_name + ".")
        print("Out of " + str(len(results)) + ", the classifier found " + str(sentiment_pos)
              + " comments from reddit post to be positive in nature, while " + str(sentiment_neg) +
              " comments from the post were found to be negative in nature.")
        run_time = time.time() - start_time
        print("Running time for classification: " + str(run_time) + "s.")

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
        str_input = str(option).lower()
        int_input = None
        try:
            int_input = int(option)
        except ValueError:
            int_input = None

        if str_input is not None and str_input == 'q':
            self.running = False
        elif int_input is not None:
            if int_input == 1:
                self.pull_comments(True)
            elif int_input == 2:
                self.pull_comments(False)
            elif int_input == 3:
                self.parse_lines()
            elif int_input == 4:
                self.get_post_comments()
            elif int_input == 5:
                self.classify_test_data()
        else:
            print("Invalid input, please try again.\n")

    def print_menu(self):
        print(self.greeting)
        for o in self.options:
            print(o)


if __name__ == '__main__':
    print("\nWelcome to the subreddit analysis program.")
    a = Analyzer()
    a.read_words()
    while a.is_running():
        a.print_menu()
        option = input()
        a.select_menu_option(option)
    print("Thanks for using the reddit analysis program!\n")
    quit(0)
