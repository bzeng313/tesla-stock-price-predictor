from collections import defaultdict

class linearRegression(object):
	def __init__(self):
		self.weights = defaultdict(float)

	def prediction(self, phi):
		return sum(self.weights[key]*phi[key] for key in phi)

	def computeGradient(self, phi, y):
		gradient = defaultdict(float)
		increment(gradient, 2*(dotProd(self.weights, phi) - y), phi)
		return gradient

	def gradientDescent(self, data, eta, iters):
		for _ in range(iters):
			gradient = defaultdict(float)
			for phi, y in data:
				increment(gradient, 1, self.computeGradient(phi, y))
			increment(self.weights, -eta, gradient)


def dotProd(v1, v2):
	return sum(v1[key]*v2[key] for key in v2)
def tweetFeatureExtractor(tweet):
	tweetPhi = defaultdict(int)
	tweetPhi['BIASUNIT'] = 1
	for word in tweet.split():
		tweetPhi[word] += 1
	return tweetPhi

def increment(v1, scale, v2):
	for key in v2:
		v1[key] += scale * v2[key]
		if v1[key] == 0:
			del v1[key]

def extractDatesAndTweets(tweetFile):
	tweets = open(tweetFile)
	dateTweetDict = defaultdict(str)
	while True:
		tweetDate = tweets.readline().strip()
		tweet = tweets.readline().strip()
		if tweetDate == '':
			break
		dateTweetDict[tweetDate] += ' ' + tweet
	return dateTweetDict

def extractDatesAndPrices(stockFile):
	stockprices = open(stockFile)
	datePriceDict = defaultdict(float)
	while True:
		priceDate = stockprices.readline().strip()
		price = stockprices.readline().strip()
		if priceDate == '':
			break
		datePriceDict[priceDate] = float(price)
	return datePriceDict


dateTweetDict = extractDatesAndTweets('tweets.txt')
datePriceDict = extractDatesAndPrices('stockprices.txt')
data = []
for date in datePriceDict:
	data.append((tweetFeatureExtractor(dateTweetDict[date]), datePriceDict[date]))

lr = linearRegression()
lr.gradientDescent(data, 0.007, 10000)
for word in lr.weights:
	print word, lr.weights[word]
print lr.prediction(tweetFeatureExtractor('Send me ur dankest memes!!'))
