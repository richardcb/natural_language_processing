from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

# Twitter Dev consumer key, consumer secret, access token, access secret.
ckey = "MwKOyCPQCQbEzpZ0e3zIbcEng"
csecret = "EP4P4sMHBpYEBQKNzDfYX2lWw5AE6WWSXWWt1EwqcQ9Ef0ROvw"
atoken = "864730751631097856-dUstQsyjvGjmc1FJqkMQg1hnWDvhqN1"
asecret = "ADWW7n89J92VVqOfEcTbk4TaQkk93USLCsAOXmftYJa3e"


class Listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)
        if confidence*100 >= 80:
            output = open("twitteroutput.txt", "a")
            output.write(sentiment_value)
            output.write("\n")
            output.close()

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, Listener())
twitterStream.filter(track=["fun"])
