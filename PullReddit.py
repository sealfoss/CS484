import requests
import requests.auth
import praw

c_id='R1_WMZ0uXs7OZA'
c_secret = 'ggr71VKdpYQlL7cYwyT_QAtT4VI'
p_word = 'mason4lyfe'
u_agent= 'Comment Pull by Reed'
u_name = 'cs484_bot'


sub_name = 'The_Donald'
top_sub_len = 60
top_com_len = 10

reddit = praw.Reddit(client_id=c_id,
                     client_secret=c_secret,
                     password=p_word,
                     user_agent=u_agent,
                     username=u_name)

subreddit = reddit.subreddit(sub_name)


top_submissions = []
count = 1
for submission in subreddit.top(limit=top_sub_len):
    top_submissions.append(submission)
    print("\t" + str(count) + ")  \"" + submission.title 
    + "\"\n\tScore: " + str(submission.score) + "\n")
    count += 1

f = open("donald_garbage.txt", "w+")

print("\nParsing comments...")
top_comments = []
for submission in top_submissions:
    print("Comments from submission \"" + submission.title + "\":\n")
    f.write("\nComments from submission \"" + submission.title + "\":\n")
    submission.comment_sort = 'top'
    comments = submission.comments
    com_count = 0
    for comment in comments:
        top_comments.append(comment.body)
        # print("Comment #" + str(com_count) + ":\t\"" + comment.body + "\"\n")
        f.write("Comment #" + str(com_count+1) + ":\t\"" + comment.body + "\"\n")
        com_count += 1
        if com_count >= top_com_len:
            break
    print("\n")
f.close()
"""
top_comments = []
submission = submissions[0]
comments = submission.comment_sort = 'top'

count = 1
for comment in submission.comments:
    print("\nComment #" + str(count) + ": \"" + comment.body + "\"")
    count += 1
"""


