import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import math
from datetime import date, timedelta

dataset_file_name1 = '../datasets/tesla-tweets-18-1-1to18-11-27.csv'
dataset_file_name2 = '../datasets/tesla-stocks-18-1-1to18-11-27.csv'

print 'Loading Tesla tweets dataset and Tesla stocks dataset...'
dataset1 = pd.read_csv(dataset_file_name1)
dataset2 = pd.read_csv(dataset_file_name2)

#################################################################################################################
'''
In this section, we will process |dataset1| to obtain a list of lists of tuples, |data1|. Each list within |data1|
corresponds to a particular date, with the first element of |data1| corresponding to 1/1/18 and the last element
corresponding to the 11/27/18. Each tuple within the list contains a tweet and that tweet's number of replies, 
number of retweets, and number of favorites, i.e. (tweet, #replies, #retweets, #favorites).
'''

print 'Processing tweets dataset...'
def convert_to_int(input):
    '''
    A custom method to convert #replies, #retweets, #favorites to floats from |dataset1|, since they come in the form
    float('nan'), #.#K, or #... for some odd reason.
    '''
    if not isinstance(input, str) and math.isnan(input):
        return 0
    if 'K' in input:
        return float(1000*float(input[:len(input) - 1]))
    return float(input)

data1 = [[]]
dates = dataset1['published_date']
tweets = dataset1['content']
replies = dataset1['replies']
retweets = dataset1['retweets']
favorites = dataset1['favorites']

prev_date = dates[0].split()[0]

for i in range(len(dates)):
    curr_date = dates[i].split()[0]

    if curr_date != prev_date:
        data1.append([])
        prev_date = curr_date

    tweet = tweets[i]
    num_replies = convert_to_int(replies[i])
    num_retweets = convert_to_int(retweets[i])
    num_favorites = convert_to_int(favorites[i])
    data1[len(data1) - 1].append((tweet, num_replies, num_retweets, num_favorites))

print '{} days worth of tweet data read...'.format(len(data1))
#################################################################################################################
'''
In this section we will process |dataset2| to obtain a list of tuples, |data2|. Each tuple contains a closing stock 
price and volume for a particular date. Each tuple within |data2| also corresponds to a particular date, with the 
first element of |data1| corresponding to 1/1/18 and the last element corresponding to the 11/27/18. Note that the 
stock market is closed on holidays and weekends, therefore we will estimate missing stock prices and volumes
via (x + y)/2, where x is the last known stock price, and y is the next available stock price. 
'''

print 'Processing stocks dataset...'
data2 = []

dates = dataset2['Date']
closing_stock_prices = dataset2['Close']
volumes = dataset2['Volume']

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
            data2.append((prev_stock_price, prev_volume))
            prev_date += one_day

    data2.append((curr_stock_price, curr_volume))
    prev_date = curr_date

#removes the entries for  12/30/17 and 12/31/17
data2.pop(0)
data2.pop(0)

print "{} days worth of stock data read...".format(len(data2))
#################################################################################################################
'''
Here we will construct a consolidated dataset, |dataX| and |dataY|. |dataX| will be a list of tuples. Each tuple 
will be of the form tuple(|data1[i:i+M]| + |data2[i:i+M]|). |dayaY| will be a list of our stock prices, starting 
from |data2[i+M][0]|. i.e., we are developing the infrastructure for an auto-regressive model of memory |M|.
'''

M = 5

dataX = []
dataY = []

for i in range(len(data1) - M):
    dataX.append(tuple(data1[i:i+M] + data2[i:i+M]))
    dataY.append(data2[i+M][0])


#####################################################################################################################################################################
'''
Here we will declare methods to extract features from each tuple in |dataX|. Note these features 
are written under the assumption that each tuple is in the form (tweet_info_0, ..., tweet_info_M, (close_price_0, volume_0), 
..., (close_price_M, volume_M)). Each tweet_info_i is a list of tuples, where each tuple is of the form 
(tweet, #replies, #retweets, #favorites). 

|features| is a list of functions. Each function takes an element of |dataX| in the format described above and outputs
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

#Add values of the stocks from the previous |M| days as features.
for i in range(M, len(dataX[0])):
    def stock_val(x):
        return x[i][0]
    features.append(stock_val)

#Add volumes of the stocks from the previous |M| days as features.
for i in range(M, len(dataX[0])):
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

############################################################################################################################
'''
Using these feature extractors, we will construct feature vectors, |phi|, where |phi| is a list of feature values and 
then construct a new dataset |dataPHI|. |dataPHI| will be directly fed into our proposed learning models. 

We also normalize each feature vector |phi| via standardization because the scales for each feature may be very different.
'''

print 'Computing dataset feature vectors...'
dataPHI = []
for x in dataX:
    phi = [phi_i(x) for phi_i in features]
    dataPHI.append(phi)

dataPHI = preprocessing.scale(dataPHI)

#Hack: The bias feature becomes 0 after scaling, so we will re-add it here.
for phi in dataPHI:
    phi[0] = 1.0


############################################################################################################################
'''
Next, we will want to split our data into a training set, development set, and test set in order to cross-validate and tune
hyperparameters. From |dataPHI| and |dataY|, we will allocate the latter |TEST_SIZE|*100% of it to be the test set. Then, 
we will allocate the former |TRAIN_SIZE|*100% to be the train set, and the latter (1 - |TRAIN_SIZE|)*100% to be the dev set.
i.e. |ORIGINALDATA| = | TRAIN_SIZE | DEV_SIZE | TEST_SIZE|
'''

TEST_SIZE = 0.25
TRAIN_SIZE = 0.75

print 'Splitting data into train ({}%), dev ({}%), and test ({}%) sets...'.format((1 - TEST_SIZE)*TRAIN_SIZE*100, (1 - TEST_SIZE)*(1 - TRAIN_SIZE)*100, TEST_SIZE*100)
dataPHI_former, dataPHI_test, dataY_former, dataY_test = train_test_split(dataPHI, dataY, test_size=TEST_SIZE, random_state=0, shuffle=False)
dataPHI_train, dataPHI_dev, dataY_train, dataY_dev = train_test_split(dataPHI_former, dataY_former, test_size=TEST_SIZE, random_state=0, shuffle=False)

############################################################################################################################
'''
For our final step, we will implement all our proposed models. The first will be our baseline linear regression model.
'''
print 'Training linear regression model...\n'
lr = LinearRegression().fit(dataPHI_train, dataY_train)
predY_train = lr.predict(dataPHI_train)
predY_dev = lr.predict(dataPHI_dev)
predY_test = lr.predict(dataPHI_test)
print 'Train mean squared error: ', mean_squared_error(predY_train, dataY_train)
print 'Train score: ', lr.score(dataPHI_train, dataY_train)
print 'Dev mean squared error: ', mean_squared_error(predY_dev, dataY_dev)
print 'Dev score: ', lr.score(dataPHI_dev, dataY_dev)
print 'Test mean squared error: ', mean_squared_error(predY_test, dataY_test)
print 'Test score: ', lr.score(dataPHI_test, dataY_test)

############################################################################################################################
'''
Here we will implement our neural network model.
'''
print '\nTraining neural network model...\n'
nn = MLPRegressor(hidden_layer_sizes=(150), activation='logistic', max_iter = 10000, learning_rate_init=0.03).fit(dataPHI, dataY)
predY_train = nn.predict(dataPHI_train)
predY_dev = nn.predict(dataPHI_dev)
predY_test = nn.predict(dataPHI_test)
print 'Train mean squared error: ', mean_squared_error(predY_train, dataY_train)
print 'Train score: ', nn.score(dataPHI_train, dataY_train)
print 'Dev mean squared error: ', mean_squared_error(predY_dev, dataY_dev)
print 'Dev score: ', nn.score(dataPHI_dev, dataY_dev)
print 'Test mean squared error: ', mean_squared_error(predY_test, dataY_test)
print 'Test score: ', nn.score(dataPHI_test, dataY_test)

#fit(dataPHI, dataY)









