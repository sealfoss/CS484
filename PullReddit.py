import praw
import time

c_id='R1_WMZ0uXs7OZA'
c_secret = 'ggr71VKdpYQlL7cYwyT_QAtT4VI'
p_word = 'mason4lyfe'
u_agent= 'Comment Pull by Reed'
u_name = 'cs484_bot'


# sub_name = 'The_Donald'
# sub_name = "JoeBiden"
sub_name = "ElizabethWarren"
max_top_posts = 200
max_top_comm = 100
out_filename = sub_name + "-posts" + str(max_top_posts) + "-comments" + str(max_top_comm) + ".dat"

print("Reading the top " + str(max_top_comm) + " comments from the top "
      + str(max_top_posts) + " posts from /r/" + sub_name)
reddit = praw.Reddit(client_id=c_id,
                     client_secret=c_secret,
                     password=p_word,
                     user_agent=u_agent,
                     username=u_name)

subreddit = reddit.subreddit(sub_name)


f = open(out_filename, "w+")
lines = []

top_submissions = []
count = 1
for submission in subreddit.top(limit=max_top_posts):

    top_submissions.append(submission)
    print("\t" + str(count) + ")  \"" + submission.title + "\"\n\tScore: " + str(submission.score) + "\n")
    count += 1

# top_comments = []
print("\nParsing comments...")
for submission in top_submissions:
    print("\nComments from submission \"" + submission.title + "\"")
    lines.append(submission.title)
    submission.comments.replace_more(limit=0)
    submission.comment_sort = 'top'
    comments = submission.comments.list()

    com_count = 0
    for comment in comments:
        # top_comments.append(comment.body)
        # print("Comment #" + str(com_count) + ":\t\"" + comment.body + "\"\n")
        # f.write("Comment #" + str(com_count+1) + ":\t\"" + comment.body + "\"\n")
        body = str(comment.body)
        if body is not None:
            lines.append(comment.body)
            com_count += 1
            if com_count >= max_top_comm:
                break
    print("\n")

print("Writing " + str(len(lines)) + " lines to file \"" + out_filename)
for line in lines:
    formatted = line.replace("\n", " ")
    formatted = formatted.replace("\t", " ")
    f.write(formatted + "\n")
f.close()
print("ALL DONE!")

