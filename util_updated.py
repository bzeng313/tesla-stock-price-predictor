from collections import defaultdict
from scipy.stats import norm, lognorm
from scipy.integrate import nquad
from copy import deepcopy
from datetime import date, timedelta
import csv
class BayesianNetwork(object):
	def __init__(self, nodes=[], factors=defaultdict(list)):
		'''
		Constructor method.
		'''
		self.nodes = nodes
		self.factors = factors


	def add_factor(self, related_nodes, new_factor):
		'''
		Takes a set of all nodes (|related_nodes|) that the new factor (|new_factor|)
		depends on. |new_factor| is either a probability density function (PDF) or
		probability mass function (PDF). |new_factor| takes in a defaultdict of
		nodes to values. 
		'''
		self.factors[related_nodes].append(new_factor)

	def add_node(self, new_node):
		'''
		Adds |new_node| |to self.nodes|.
		'''
		if new_node in self.nodes:
			return
		self.nodes.append(new_node)



###############################################################################
'''
Here we will create our stock price and twitter sentiment Bayesian network. For a graphical
depiction, refer to our progress report. 
'''
class StockAndTweetBayesianNetwork(BayesianNetwork):

	#Best values for hyperparameters that are found experimentally.
	N = 3 #Number of previous stock price nodes that S_{i} depends on.
	W = 0.2 #How important the tweet sentiment is. Is always a positive number, small and near 0.
	def __init__(self, width, nodes=[], factors=defaultdict(list)):
		'''
		Constructor method. |width| is the number of stock nodes, which is the same as the number of 
		tweet nodes.
		'''
		BayesianNetwork.__init__(self, nodes, factors)
		self.width = width

		#Initialize our network.
		for i in range(self.width):
			#Add stock node.
			self.nodes.append('S_%d'%i)
			#Add tweet sentiment node.
			self.nodes.append('T_%d'%i)

			#Get set of related tweet sentiment nodes
			related_nodes = set([])
			related_nodes.add('T_%d'%i)

			#Add tweet factor. 
			def tweet_sentiment_factor(node_to_val_dict):
				'''
				DETERMINE WHAT TO DO FROM HERE. REFER TO BOTTOM.
				'''
			self.add_factor(related_nodes, tweet_sentiment_factor)

			#Get set of related tweet sentiment nodes (already done above) and related stock nodes
			num_dependent = N if i >= N else i #Number of stock nodes S_{i} depends on
			for j in range(i - num_dependent, i + 1):
				related_nodes.add('S_%d'%j)
			#Add stock and tweet factor
			def stock_given_prev_stock_prices_and_tweet_sentiment(node_to_val_dict):
				'''
				DETERMINE WHAT TO DO FROM HERE. REFER TO BOTTOM.
				'''
	def infer(query, evidence):
		'''
		Does Bayesian inference to determine the probability of |query| given |evidence|. 
		|query| and |evidence| are defaultdicts of nodes to values. 
		Right now need to find a way to eliminate non-query, non-evidence nodes... Also
		need to find a way to normalize. 
		'''
		factors = deepcopy(self.factors)
		node_to_val_dict = dict(query, **evidence)

		#STEP1 CONDITION

		#STEP2 ELIMINATE

		#STEP3 RETURN

'''
Common factors that may be used and their format.

###############################################################################
Returns the probability that a tweet sentiment is one of neutral, good, or bad.
P(T_{i} = 'neutral') = p_n, P(T_{i} = 'good') = p_g, P(T_{i} = 'bad') = p_b.
p_n, p_g, and p_b are found experimentally from all tweets in day i. 

def tweet_sentiment_factor(node_to_val_dict):
	tweet_sentiment_value = node_to_val_dict['T_{i}']
	if tweet_sentiment_value == 'neutral':
		return p_n
	elif tweet_sentiment_value == 'good':
		return p_g
	elif tweet_sentiment_value == 'bad':
		return p_b
	raise Exception('T_{i} cannot take on a value of {}'.format(tweet_sentiment_value))

###############################################################################
Alternate version of above. Returns PDF of a normal distribution evaluated at T_{i}'s 
value. |mu| and |std| are found experimentally from all tweets in day i. 

def tweet_sentiment_factor(node_to_val_dict):
	tweet_sentiment_value = node_to_val_dict['T_{i}']
	return norm.pdf(tweet_sentiment_value, loc=mu, scale=std)

###############################################################################
Returns the PDF of a log-normal distribution of S_{i} given S_{i-1}, S_{i-2},
..., S_{i-N} and T_{i}. |mu_prime| is given by (1 - |W|(0.5 - T_{i}))*|mu|. 
|std_prime| is found by (1 - |W|(0.5 - T_{i}))^2*|std|. The best |W| and |N| is 
found experimentally. |mu| and |std| are found experimentally from S_{i-1}, 
S_{i-2}, ..., and S_{i-N}. Given a T_{i} is discrete and NOT normally distributed,
we let neutral = 0.5, good = 1, bad = 0.

FIGURE OUT WHAT TO DO WHEN S_{i} depends on no nodes or only a single node...
b/c std would be undefined.... Perhaps just make probability of first 2 nodes
1???????

def stock_given_prev_stock_prices_and_tweet_sentiment(node_to_val_dict):
	tweet_sentiment_value = node_to_val_dict['T_{i}']
	potential_stock_price = ndoe_to_val_dict['S_{i}']
	mu = 0.0

	num_dependent = N if i > N else i - 1
	for j in range(i - num_dependent, i):
		mu += node_to_val_dict['S_{j}']
	mu /= n

	std = 0.0
	for j in ranage(i - n, i):
		std += (node_to_val_dict['S_{j}'] - mu)**2
	std /= n
	std **= 0.5

	mu_prime =  (1 - W*(0.5 - tweet_sentiment_value))*mu
	std_prime = (1 - W*(0.5 - tweet_sentiment_value))**2*std

	return lognormal.pdf(potential_stock_price, std_prime, loc=0, scale=mu_prime)
	
'''


###############################################################################
'''
Here are some useful functions. 
'''

def increment(v1, scale, v2):
	'''
	Increments a defualtdict(float) |v1| by another defualtdict(float) |v2|, whose 
	values have been scaled by a float |scale|. (i.e. |v1| = |scale|*|v2|)
	'''
	for key in v2:
		v1[key] += scale*v2[key]

def scale(v1, scale):
	'''
	Scales a defaultdict(float) |v1| by a float |scale|. (i.e. |v1| = |scale|*|v1|)
	'''
	for key in v1:
		v1[key] *= scale

def dot_product(v1, v2):
	'''
	Returns the dot product between a defaultdict(float) |v1| and another defualtdict(float) |v2|. 
	(i.e. |v1|*|v2|)
	'''

###############################################################################
'''
Here we will implement our linear regression model.
'''

class LinearRegression(object):
	def __init__(self, weights=defaultdict(float)):
		'''
		Constructor method. Initializes |self.weights| to the defaultdict(float) |weights|.
		'''
		self.weights = weights

	def prediction(self, phi):
		'''
		Outputs the prediction given a defaultdict(float) feature vector |phi|.
		'''
		return sum(self.weights[key]*phi[key] for key in phi)

	# def computeGradient(self, phi, y):
	# 	'''
	# 	Computes the gradient of 
	# 	'''
	# 	gradient = defaultdict(float)
	# 	increment(gradient, 2*(dotProd(self.weights, phi) - y), phi)
	# 	return gradient

	def train(self, data, eta, iters):
		'''
		Trains the linear regression model using batch gradient descent given a list of |data|, a learning 
		rate |eta| and number of iterations |iter|. |data| is a list with elements of the form (|phi|, |y|), 
		where |phi| is a feature vector and |y| is the real value associated with this feature vector. 
		'''
		def compute_gradient_for_single_point(phi, y):
			'''
			Computes the gradient for a single data point (|phi|, |y|). We use a squared loss function
			(|phi|*|self.weights| - |y|)^2, so the gradient is 2*(|phi|*|self.weights| - |y|)*|self.weights|.
			Returns this single gradient as a defaultdict(float)
			'''
			single_gradient = defaultdict(float)
			increment(single_gradient, 2.0*(dot_product(self.weights, phi) - y), phi)
			return single_gradient

		for _ in range(iters): 
			gradient = defaultdict(float)
			for phi, y in data:
				increment(gradient, 1.0, compute_gradient_for_single_point(phi, y))
			increment(self.weights, -eta, gradient)


	def gradientDescent(self, data, eta, iters):
		for _ in range(iters):
			gradient = defaultdict(float)
			for phi, y in data:
				increment(gradient, 1, self.computeGradient(phi, y))
			increment(self.weights, -eta, gradient)

	def get_error(self, data, eta, iters):
		pass

###############################################################################
'''
Here we will preprocess the data for our project and output a list of (|phi|, |y|)'s
'''
def parse_stocks(stock_file):
	'''
	Returns a defaultdict(float) of date objects to the closing stock price of that date.
	Takes in a csv file name |stock_file| formatted as Date | Open | High | Low | Close | Adj Close | Volume.
	The first row of the csv file will be the category of each column seperated by a ','. All other rows will
	have the actual values of the entries seperated by a ','. A Date entry is of the form YYYY-MM-DD.
	'''
	date_to_closing_price = defaultdict(float)
	one_day = timedelta(days=1)
	prev_date = None
	with open(stock_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		row_count = 0
		for row in csv_reader:
			if row_count == 0:
				print 'Reading each line as {}'.format(', '.join(row))
			else:
				year, month, day = row[0].split('-')
				year = int(year)
				month = int(month)
				day = int(day)

				this_date = date(year=year, month=month, day=day)
				date_to_closing_price[this_date] = float(row[4])

				#Fills in missing days.
				if prev_date != None and this_date - prev_date > one_day:
					earliest_closing_price = date_to_closing_price[prev_date]
					latest_closing_price = date_to_closing_price[this_date]
					for i in range((this_date - prev_date).days - 1):
						date_to_closing_price[prev_date + timedelta(days=i + 1)] = (earliest_closing_price + latest_closing_price)/2.0
						earliest_closing_price = date_to_closing_price[prev_date + timedelta(days=i + 1)]
				prev_date = this_date

			row_count += 1

	return date_to_closing_price

def parse_tweet_sentiments(tweet_sentiments_file):
	'''
	Returns a defaultdict(list) of date objects to the list of tweet sentiment values.
	Takes in a csv file name |tweet_sentiment_file| formatted as 
	Date | Very Negative | Negative | Neutral | Positive | Very Positive | Sentiment Classification.
	The first row of the csv file will be the category of each column seperated by a ','. All other rows will
	have the actual values of the entries seperated by a ','. A Date entry is of the form YYYY-MM-DD.
	'''
	date_to_tweet_sentiments = defaultdict(list)
	with open(tweet_sentiments_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		line = 0
		row_count = 0
		for row in csv_reader:
			if row_count == 0:
				print 'Reading each line as {}'.format(', '.join(row))
			else:
				month, day, year = row[0].split('/')
				year = int("20" + year)
				if int(month) < 10:
					month = "0" + month
				if int(day) < 10:
					day = "0" + day
				month = int(month)
				day = int(day)

				this_date = date(year=year, month=month, day=day)
				date_to_tweet_sentiments[this_date] = [float(val) for val in row[1:]]
			row_count += 1
	return date_to_tweet_sentiments

def average_sentiment(date_to_tweet_sentiments):
	for date in date_to_tweet_sentiments:
		numElem = len(date_to_tweet_sentiments[date])
		aggregatedSum = [0, 0, 0, 0, 0]
		for entry in date_to_tweet_sentiments[date]:
			for i in range(5):
				aggregatedSum[i] += entry[i]
		for i in range(5):
			aggregatedSum[i] /= numElem
		date_to_tweet_sentiments[date] = aggregatedSum

data = []
date_to_closing_price = parse_stocks('/Users/brianzeng/Downloads/cs221-project-master/TSLA.csv')
date_to_tweet_sentiments = parse_tweet_sentiments('/Users/brianzeng/Downloads/cs221-project-master/tweet_sentiments_file.csv')
average_sentiment(date_to_tweet_sentiments)

for date in date_to_tweet_sentiments:
	print date, date_to_tweet_sentiments[date], '\n'



# for date in date_to_closing_price:
# 	#list of features we want to have

# 	data.append(())
####







