import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import KFold
import math
from datetime import date, timedelta

dataset_file_name1 = '../datasets/tesla-tweets-18-1-1to18-11-27.csv'
dataset_file_name2 = '../datasets/tesla-stocks-18-1-1to18-11-27.csv'
dataset_file_name3 = '../datasets/tesla-average-tweets-sentiment-18-1-1to18-11-27.csv'

print 'Loading Tesla tweets and sentiments dataset and Tesla stocks dataset...'
tweet_dataset = pd.read_csv(dataset_file_name1)
stock_price_dataset = pd.read_csv(dataset_file_name2)
tweet_sentiment_dataset = pd.read_csv(dataset_file_name3)

#################################################################################################################
'''
In this section, we will process |tweet_dataset| to obtain a list of lists of tuples, |tweet_data|. Each list within |tweet_data|
corresponds to a particular date, with the first element of |tweet_data| corresponding to 1/1/18 and the last element
corresponding to the 11/27/18. Each tuple within the list contains a tweet and that tweet's number of replies, 
number of retweets, and number of favorites, i.e. (tweet, #replies, #retweets, #favorites).
'''

print 'Processing tweets dataset...'
def convert_to_int(input):
    '''
    A custom method to convert #replies, #retweets, #favorites to floats from |tweet_dataset|, since they come in the form
    float('nan'), #.#K, or #... for some odd reason.
    '''
    if not isinstance(input, str) and math.isnan(input):
        return 0
    if 'K' in input:
        return float(1000*float(input[:len(input) - 1]))
    return float(input)

tweet_data = [[]]
dates = tweet_dataset['published_date']
tweets = tweet_dataset['content']
replies = tweet_dataset['replies']
retweets = tweet_dataset['retweets']
favorites = tweet_dataset['favorites']

sentiment = tweet_sentiment_dataset['Score']

prev_date = dates[0].split()[0]
prev_i = 0

for i in range(len(dates)):
    curr_date = dates[i].split()[0]

    if curr_date != prev_date:
        tweet_data[len(tweet_data) - 1].append(sentiment[prev_i])
        prev_i += 1
        tweet_data.append([])
        prev_date = curr_date

    tweet = tweets[i]
    num_replies = convert_to_int(replies[i])
    num_retweets = convert_to_int(retweets[i])
    num_favorites = convert_to_int(favorites[i])
    tweet_data[len(tweet_data) - 1].append((tweet, num_replies, num_retweets, num_favorites))

print '{} days worth of tweet data read...'.format(len(tweet_data))
#################################################################################################################
'''
In this section we will process |stock_price_dataset| to obtain a list of tuples, |stock_price_data|. Each tuple contains a closing stock 
price and volume for a particular date. Each tuple within |stock_price_data| also corresponds to a particular date, with the 
first element of |stock_price_data| corresponding to 1/1/18 and the last element corresponding to the 11/27/18. Note that the 
stock market is closed on holidays and weekends, therefore we will estimate missing stock prices and volumes
via (x + y)/2, where x is the last known stock price, and y is the next available stock price. 
Form: (stock_price, stock_volume)
'''

print 'Processing stocks dataset...'
stock_price_data = []

dates = stock_price_dataset['Date']
closing_stock_prices = stock_price_dataset['Close']
volumes = stock_price_dataset['Volume']

one_day = timedelta(days=1)
year, month, day = dates[0].split('-')
year = int(year)
month = int(month)
day = int(day)
prev_date = date(year=year, month=month, day=day)
prev_stock_price = float(closing_stock_prices[0])
prev_volume = float(volumes[0])

for i in range(1, len(dates)):

    year, month, day = dates[i].split('-')
    year = int(year)
    month = int(month)
    day = int(day)

    curr_date = date(year=year, month=month, day=day)
    curr_stock_price = float(closing_stock_prices[i])
    curr_volume = float(volumes[i])

    if curr_date - prev_date > one_day:
        prev_date += one_day
        while prev_date != curr_date:
            prev_stock_price = (prev_stock_price + curr_stock_price)/2
            prev_volume = (prev_volume + curr_volume)/2
            stock_price_data.append((prev_stock_price, prev_volume))
            prev_date += one_day

    stock_price_data.append((curr_stock_price, curr_volume))
    prev_date = curr_date

#removes the entries for  12/30/17 and 12/31/17
stock_price_data.pop(0)
stock_price_data.pop(0)

print "{} days worth of stock data read...".format(len(stock_price_data))
#################################################################################################################
'''
Here we will construct a consolidated dataset, |future_X_data| and |future_Y_data|. |future_X_data| will be a list of tuples. Each tuple 
will be of the form tuple(|tweet_data[i:i+M]| + |stock_price_data[i:i+M]|). |future_Y_data| will be a list of our stock prices, starting 
from |stock_price_data[i+M][0]|. i.e., we are developing the infrastructure for an auto-regressive model of memory |M|.
'''

M = 5

future_X_data = []
future_Y_data = []

for i in range(len(tweet_data) - M):
    future_X_data.append(tuple(tweet_data[i:i+M] + stock_price_data[i:i+M]))
    future_Y_data.append(stock_price_data[i+M][0])


#####################################################################################################################################################################
'''
Here we will declare methods to extract features from each tuple in |future_X_data|. Note these features 
are written under the assumption that each tuple is in the form (tweet_info_0, ..., tweet_info_M, (close_price_0, volume_0), 
..., (close_price_M, volume_M)). Each tweet_info_i is a list of tuples, where each tuple is of the form 
(tweet, #replies, #retweets, #favorites). 

|features| is a list of functions. Each function takes an element of |future_X_data| in the format described above and outputs
feature value. 
'''

features = []

def compute_phi(x):
    '''
    Computes the feature vector |phi| from the datapoint |x| and returns it. 
    '''
    phi = []
    for phi_i in features:
        phi.append(phi_i(x))
    return phi

#####################################################
'''
Create features here and describe them. Append them to |features| in the order they were created. 
'''

#Add a bias feature
def bias(x):
    return 1
features.append(bias)

def sentiment(x):
    return x[M - 1][len(x[M-1]) - 1]
features.append(sentiment)

#Add values of the stocks from the previous |M| days as features.
for i in range(M, len(future_X_data[0])):
    def stock_val(x):
        return x[i][0]
    features.append(stock_val)

#Add volumes of the stocks from the previous |M| days as features.
for i in range(M, len(future_X_data[0])):
    def volume(x):
        return x[i][1]
    features.append(volume)

#Proposed features to add for each day and reason
    #Proposed features for each tweet and reason
        #Each feature will be multiplied by (#replies + #retweets + #favorites) to scale for relevance
            #maybe quantify the tweet by repeating it (#replies + #retweets + #favorites)... might not be 
            #the greatest idea if we have to repeat the tweet 1000 times
        #(word count)... more words means usually means a big announcement
        #contains word increase
        #contains word rise
        #contains word fraud
        #contains word bankrupt
        #contains word fail
        #weighted n-gram of all tweets
        #look up word2vec
            #for above features 'contains word', just compute the similarity of each word
            #in a tweet to them
        #some multiplication of these features?

for i in range(M, len(future_X_data[0])):
    pass

############################################################################################################################
'''
Using these feature extractors, we will construct feature vectors, |phi|, where |phi| is a list of feature values and 
then construct a new dataset |dataPHI|. |dataPHI| will be directly fed into our proposed learning models. 

We also normalize each feature vector |phi| via standardization because the scales for each feature may be very different.
'''

print 'Computing dataset feature vectors...'
dataPHI = []
for x in future_X_data:
    phi = compute_phi(x)
    dataPHI.append(phi)

dataPHI = preprocessing.scale(dataPHI)

#Hack: The bias feature becomes 0 after scaling, so we will re-add it here.
for phi in dataPHI:
    phi[0] = 1.0


############################################################################################################################
'''
Next, we will want to split our data into a training set, development set, and test set in order to cross-validate and tune
hyperparameters. From |dataPHI| and |dataY|, we will allocate the latter |TEST_SIZE|*100% of it to be the test set. Then,
we will allocate the former |TRAIN_SIZE|*100% to be the train set. From the train set, we will do k-fold cross-validation
to determine the generality of our models later on.
'''

TRAIN_SIZE = 0.9
TEST_SIZE = 1 - TRAIN_SIZE


print ('Splitting data into train ({}%) and test ({}%) sets...'.format(TRAIN_SIZE*100, TEST_SIZE*100))
dataPHI_train, dataPHI_test, dataY_train, dataY_test = train_test_split(dataPHI, future_Y_data, test_size=TEST_SIZE, random_state=0, shuffle=False)

############################################################################################################################
'''
We will now implement all our proposed models and do k-fold cross-validation. The first will be our baseline
linear regression model.
'''

print('Starting k-fold cross-validation for linear regression model')
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, random_state=1, shuffle=True)

average_kfold_train_error = 0
average_kfold_train_score = 0
average_kfold_dev_error = 0
average_kfold_dev_score = 0
for train_i, dev_i in kf.split(dataPHI_train):
    dataPHI_kfold_train = [dataPHI_train[i] for i in train_i]
    dataY_kfold_train = [dataY_train[i] for i in train_i]
    dataPHI_kfold_dev = [dataPHI_train[i] for i in dev_i]
    dataY_kfold_dev = [dataY_train[i] for i in dev_i]
    lr = LinearRegression().fit(dataPHI_kfold_train, dataY_kfold_train)
    predY_kfold_train = lr.predict(dataPHI_kfold_train)
    predY_kfold_dev = lr.predict(dataPHI_kfold_dev)
    average_kfold_train_error += mean_squared_error(predY_kfold_train, dataY_kfold_train)/N_SPLITS
    average_kfold_train_score += lr.score(dataPHI_kfold_train, dataY_kfold_train)/N_SPLITS
    average_kfold_dev_error += mean_squared_error(predY_kfold_dev, dataY_kfold_dev)/N_SPLITS
    average_kfold_dev_score += lr.score(dataPHI_kfold_dev, dataY_kfold_dev)/N_SPLITS
print('k-fold evaluations for linear regression model:')
print('k-fold train mean squared error: {}'.format(average_kfold_train_error))
print('k-fold train score: {}'.format(average_kfold_train_score))
print('k-fold dev mean squared error: {}'.format(average_kfold_dev_error))
print('k-fold dev score: {}'.format(average_kfold_dev_score))

print ('Training linear regression model on entire train set...\n')
lr = LinearRegression().fit(dataPHI_train, dataY_train)
predY_train = lr.predict(dataPHI_train)
predY_test = lr.predict(dataPHI_test)
print ('Train mean squared error: {}'.format(mean_squared_error(predY_train, dataY_train)))
print ('Train score: {}'.format(lr.score(dataPHI_train, dataY_train)))
print ('Test mean squared error: {}'.format(mean_squared_error(predY_test, dataY_test)))
print ('Test score: {}'.format(lr.score(dataPHI_test, dataY_test)))

############################################################################################################################
'''
Here we will implement our neural network model.
# '''

print('Starting k-fold cross-validation for neural network model')
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, random_state=1, shuffle=True)
average_kfold_train_error = 0
average_kfold_train_score = 0
average_kfold_dev_error = 0
average_kfold_dev_score = 0
for train_i, dev_i in kf.split(dataPHI_train):
    dataPHI_kfold_train = [dataPHI_train[i] for i in train_i]
    dataY_kfold_train = [dataY_train[i] for i in train_i]
    dataPHI_kfold_dev = [dataPHI_train[i] for i in dev_i]
    dataY_kfold_dev = [dataY_train[i] for i in dev_i]
    nn = MLPRegressor(hidden_layer_sizes=(10), activation='logistic', max_iter = 10000, learning_rate_init=0.03).fit(dataPHI_kfold_train, dataY_kfold_train)
    predY_kfold_train = nn.predict(dataPHI_kfold_train)
    predY_kfold_dev = nn.predict(dataPHI_kfold_dev)
    average_kfold_train_error += mean_squared_error(predY_kfold_train, dataY_kfold_train)/N_SPLITS
    average_kfold_train_score += nn.score(dataPHI_kfold_train, dataY_kfold_train)/N_SPLITS
    average_kfold_dev_error += mean_squared_error(predY_kfold_dev, dataY_kfold_dev)/N_SPLITS
    average_kfold_dev_score += nn.score(dataPHI_kfold_dev, dataY_kfold_dev)/N_SPLITS
print('k-fold evaluations for neural network model:')
print('k-fold train mean squared error: {}'.format(average_kfold_train_error))
print('k-fold train score: {}'.format(average_kfold_train_score))
print('k-fold dev mean squared error: {}'.format(average_kfold_dev_error))
print('k-fold dev score: {}'.format(average_kfold_dev_score))

print ('Training neural network model on entire train set...\n')
nn = MLPRegressor(hidden_layer_sizes=(10), activation='logistic', max_iter = 10000, learning_rate_init=0.03).fit(dataPHI_train, dataY_train)
predY_train = nn.predict(dataPHI_train)
predY_test = nn.predict(dataPHI_test)
print ('Train mean squared error: {}'.format(mean_squared_error(predY_train, dataY_train)))
print ('Train score: {}'.format(lr.score(dataPHI_train, dataY_train)))
print ('Test mean squared error: {}'.format(mean_squared_error(predY_test, dataY_test)))
print ('Test score: {}'.format(lr.score(dataPHI_test, dataY_test)))


def percent_correct(predictions, actuals):
    count = 0.0
    for i in range(len(stock_price_data)):
        if stock_price_data[i][0] in actuals:
            previous = stock_price_data[i-1][0]
            index = actuals.index(stock_price_data[i][0])
            if (actuals[index] >= previous and predictions[index] >= previous) or (actuals[index] <= previous and predictions[index] <= previous):
                count += 1
    return count / len(predictions)
print 'Train percent correct', percent_correct(predY_train, dataY_train)
print 'Test percent correct', percent_correct(predY_test, dataY_test)

############################################################################################################################
'''
Here we will plot important results
'''









