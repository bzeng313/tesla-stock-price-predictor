import csv
from collections import defaultdict

def parseStockPrices(stockFile):
	dateToClosingPrice = []
	monthToDay = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	with open(stockFile) as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    line = 0
	    expectedMonth = 1
	    expectedDay = 2
	    for row in csv_reader:
	    	if line == 0:
	    	    print 'Column names are {}'.format(', '.join(row))
	    	else:
	    	    year, month, day = row[0].split('-')
	    	    year = year[2:]
	    	    month = str(int(month))
	    	    day = str(int(day))
			
		    #we are missing some days
		    if int(month) != expectedMonth or int(day) != expectedDay:
		        
			fillMissingDays(dateToClosingPrice, expectedMonth, expectedDay, days row[4], year, monthToDay)
		    
	    	    date = month + '/' + day + '/' + year
	    	    dateToClosingPrice.append((date, row[4]))
		
		    if expectedDay + 1 > monthToDay[expectedMonth - 1]:
		        expectedMonth += 1
		        expectedDay = 1
		    else:
		        expectedDay += 1
	    	line += 1
	
	def fillMissingDays(dateToClosingPrice, expectedMonth, expectedDay, daysMissed, newPrice, year, monthToDay):
	    previousPrice = dateToClosingPrice[len(dateToClosingPrice) - 1][1]
	    for i in range(daysMissed):
		date = str(expectedMonth) + '/' + str(expectedDay) + '/' + year
		missingPrice = float(previousPrice + newPrice) / 2
		dateToClosingPrice.append((date, missingPrice))
		previousPrice = missingPrice
		
		#update day
		#if we are going to a new month
		if expectedDay + 1 > monthToDay[expectedMonth - 1]:
		    expectedMonth += 1
		    expectedDay = 1
		else:
		    expectedDay += 1
				
	return dateToClosingPrice

def parseTweets(tweetFile):
	dateToTweet = defaultdict(str)
	with open(tweetFile) as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    line = 0
	    prevDate = 1
	    for row in csv_reader:
	    	if line == 0:
	    		print 'Column names are {}'.format(', '.join(row))
	    	else:
	    		dateToTweet[row[1].split()[0]] = row[2]
	    	line += 1
	return dateToTweet

def getPercentageChanges(dateToClosingPrice):
	dateRangeToPercentChange = []
	for i in range(len(dateToClosingPrice) - 1):
		date1 = dateToClosingPrice[i][0]
		date2 = dateToClosingPrice[i + 1][0]
		dateRange = date1 + ' ' + date2
		percentChange = (float(dateToClosingPrice[i+1][1]) - float(dateToClosingPrice[i][1]))/float(dateToClosingPrice[i][1])
		dateRangeToPercentChange.append((dateRange, percentChange))
	return dateRangeToPercentChange

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




data = []
dateRangeToPercentChange = getPercentageChanges(parseStockPrices('TSLA.csv'))
dateToTweet = parseTweets('elonmusk_tweets.csv')

for entry in dateRangeToPercentChange:
	data.append((tweetFeatureExtractor(dateToTweet[entry[0].split()[1]]), entry[1]))

lr = linearRegression()
lr.gradientDescent(data, 0.003, 1000)
for word in lr.weights:
	print word, lr.weights[word]
print lr.prediction(tweetFeatureExtractor('Tesla is going bankrupt. company closing'))

