from gensim.models import Word2Vec
import pandas as pd

'''
Here we train our word2vec model on our corpus of tweets and save the model to |'../datasets/w2v_tweets.bin'|,
which can be accessed via Word2Vec.load('../datasets/w2v_tweets.bin'). This code was inspired from
https://machinelearningmastery.com/develop-word-embeddings-python-gensim/.
'''
tweet_dataset_filename = '../datasets/tesla-tweets-18-1-1to18-11-27.csv'
print 'Loading Tesla tweets...'
tweet_dataset = pd.read_csv(tweet_dataset_filename)

tweets = [tweet.split() for tweet in tweet_dataset['content']]

print 'Training word2vec model...'

w2v = Word2Vec(tweets)
w2v.train(tweets, total_examples=len(tweets), epochs=100)
w2v.save('../datasets/w2v_tweets.bin')