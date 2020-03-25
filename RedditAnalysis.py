import praw
from praw import exceptions
import prawcore


class Analyzer:
    def __init__(self):
        self.running = True
        self.c_zero_lines = None
        self.c_zero_name = None
        self.c_one_lines = None
        self.c_one_name = None
        self.word_bag = None
        self.test_lines = None
        self.greeting = "\nPlease choose from the following options ('q' to quit):\n"
        self.options = ["1)\tChoose subreddit representing classification 0.\n",
                        "2)\tChoose subreddit representing classification 1.\n",
                        "3)\tBuild word bag and training matrices.\n",
                        "4)\tTest a reddit post against training data.\n",
                        "5)\tTest a reddit comment against training data.\n"]

    def error_quit(self, error):
        print("ERROR DETECTED: " + error)
        print("Exiting program...")
        self.running = False
        pass

    def pull_comments(self, c0c1):
        pulled = None
        sub_name = None
        while sub_name is None:
            print("What is the name of the subreddit you would like to pull comments from?")
            name = str(input())
            if name is not None and len(name) > 0:
                print("You have chosen to pull comments from /r/" + name + ".")
                print("Would you like to proceed? (y/n)")
                answer = str(input()).lower()
                if answer == "y":
                    sub_name = name
            else:
                print("Invalid subreddit name, please try again.")

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
            max_posts = 0
            while max_posts == 0:
                print("How many of the top posts from /r/" + sub_name + " would you like to pull from reddit? (n>0)")
                m = int(input())
                if m is not None and m > 0:
                    print("You have chosen to pull " + str(m) + " comments from /r/" + sub_name + ".")
                    print("Would you like to proceed? (y/n)")
                    answer = str(input()).lower()
                    if answer == "y":
                        max_posts = m
                else:
                    print("Please enter an integer value greater than zero.")

            max_comments = 0
            while max_comments == 0:
                print("How many of the top comments from these posts in /r/"
                      + sub_name + " would you like to pull from reddit? (n>0)")
                m = int(input())
                if m is not None and m > 0:
                    print("You have chosen to pull " + str(m) + " comments from posts in /r/" + sub_name + ".")
                    print("Would you like to proceed? (y/n)")
                    answer = str(input()).lower()
                    if answer == "y":
                        max_comments = m
                else:
                    print("Please enter an integer value greater than zero.")

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

        c_name = None
        if c0c1:
            self.c_zero_name = sub_name
            self.c_zero_lines = pulled
            self.options[0] = "1)\tClassification C0 set to " + str(len(pulled)) \
                              + " comments from /r/" + sub_name + ".\n"
        else:
            self.c_one_name = sub_name
            self.c_one_lines = pulled
            self.options[0] = "2)\tClassification C1 set to " + str(len(pulled)) \
                              + " comments from /r/" + sub_name + ".\n"

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
        op_i = int(option)
        if op_str is not None and op_str == 'q':
            self.running = False
        elif op_i is not None:
            if op_i == 1:
                self.pull_comments(True)
            elif op_i == 2:
                self.pull_comments(False)
            elif op_i == 3:
                pass
            elif op_i == 4:
                pass
            elif op_i == 5:
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
print("Thanks for using the reddit analysis program!\n")
