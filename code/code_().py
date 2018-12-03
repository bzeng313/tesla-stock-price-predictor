import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import date, timedelta
from gensim.models import Word2Vec

#########################################################################################################################################
'''
Here we will use the pandas module to upload our relevant datasets.
'''
tweets_dataset_filename = '../datasets/tesla-tweets-18-1-1to18-11-27.csv'
stock_prices_dataset_filename = '../datasets/tesla-stocks-18-1-1to18-11-27.csv'

print('Loading Tesla tweets dataset and Tesla stocks dataset...')
tweets_dataset = pd.read_csv(tweets_dataset_filename)
stock_prices_dataset = pd.read_csv(stock_prices_dataset_filename)

#########################################################################################################################################
'''
In this section, we will process |tweets_dataset| to obtain a list of lists of tuples, |tweets_data|. Each list within |tweets_data|
corresponds to a particular date, with the first element of |tweets_data| corresponding to 1/1/18 and the last element
corresponding to the 11/27/18. Each tuple within the list contains a tweet and that tweet's number of replies, 
number of retweets, and number of favorites, i.e. (tweet, #replies, #retweets, #favorites).
'''

print('Processing tweets dataset...')
def convert_to_float(input):
    '''
    A custom method to convert #replies, #retweets, #favorites to floats from |tweets_dataset|, since they come in the form
    float('nan'), #.#K, or #... for some odd reason.
    '''
    if not isinstance(input, str) and math.isnan(input):
        return 0.0
    if 'K' in input:
        return float(1000*float(input[:len(input) - 1]))
    return float(input)

tweets_data = [[]]
dates = tweets_dataset['published_date']
tweets = tweets_dataset['content']
replies = tweets_dataset['replies']
retweets = tweets_dataset['retweets']
favorites = tweets_dataset['favorites']

prev_date = dates[0].split()[0]

for i in range(len(dates)):
    curr_date = dates[i].split()[0]

    if curr_date != prev_date:
        tweets_data.append([])
        prev_date = curr_date

    tweet = tweets[i]
    num_replies = convert_to_float(replies[i])
    num_retweets = convert_to_float(retweets[i])
    num_favorites = convert_to_float(favorites[i])
    tweets_data[len(tweets_data) - 1].append((tweet, num_replies, num_retweets, num_favorites))

print('{} days worth of tweet data read...'.format(len(tweets_data)))

avg = 0
for day in tweets_data:
    avg += 1.0*len(day)/len(tweets_data)
print avg
#########################################################################################################################################
'''
In this section we will process |stock_prices_dataset| to obtain a list of tuples, |stock_prices_data|. Each tuple contains a closing stock 
price and volume for a particular date. Each tuple within |stock_prices_data| also corresponds to a particular date, with the 
first element of |stock_prices_data| corresponding to 1/1/18 and the last element corresponding to the 11/27/18. Note that the 
stock market is closed on holidays and weekends, therefore we will estimate missing stock prices and volumes via (x + y)/2, where x is 
the last known stock price, and y is the next available stock price, i.e. (stock_price, stock_volume).
'''

print('Processing stocks dataset...')
stock_prices_data = []

dates = stock_prices_dataset['Date']
closing_stock_prices = stock_prices_dataset['Close']
volumes = stock_prices_dataset['Volume']

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
            stock_prices_data.append((prev_stock_price, prev_volume))
            prev_date += one_day

    stock_prices_data.append((curr_stock_price, curr_volume))
    prev_date = curr_date

#removes the entries for  12/30/17 and 12/31/17
stock_prices_data.pop(0)
stock_prices_data.pop(0)

print('{} days worth of stock data read...'.format(len(stock_prices_data)))

#########################################################################################################################################
'''
Here we will construct a consolidated dataset, |pastM_dataX| and |next_price_dataY|. |pastM_dataX| will be a list of tuples. Each tuple 
will be of the form tuple(|tweets_data[i:i+M]| + |stock_prices_data[i:i+M]|). |next_price_dataY| will be a list of our stock prices, starting 
from |stock_prices_data[i+M][0]|. i.e., we are developing the infrastructure for an auto-regressive model of memory |M|.
'''

M = 10

pastM_dataX = []
next_price_dataY = []

for i in range(len(tweets_data) - M):
    pastM_dataX.append(tuple(tweets_data[i:i+M] + stock_prices_data[i:i+M]))
    next_price_dataY.append(stock_prices_data[i+M][0])

##########################################################################################################################################
'''
Here we will declare methods to extract features from each tuple in |pastM_dataX|. Note these features are written under the assumption 
that each tuple |x| is in the form (tweet_info_0, ..., tweet_info_M, (close_price_0, volume_0), ..., (close_price_M, volume_M)). Each 
tweet_info_i is a list of tuples, where each tuple is of the form (tweet, #replies, #retweets, #favorites). 

|features| is a list of functions. Each function takes an element of |pastM_dataX| in the format described above and outputs
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

##########################################################################################################################################
'''
Create features here and describe them. Append them to |features| in the order they were created. NOTE: when adding features via a for loop,
and each feature function depends on the current 'i' value of that for loop, declare a second argument, 'i=i'. This way, the 'i' value within
the function will not be changed with future changes to 'i' outside the function. See the |stock_value| features for a concrete example.
'''

#Add a bias feature
def bias(x):
    return 1.0
features.append(bias)

#Add values of the stocks from the previous |M| days as features.
for i in range(M, len(pastM_dataX[0])):
    def stock_val(x, i=i):
        return x[i][0]
    features.append(stock_val)

#Add volumes of the stocks from the previous |M| days as features.
for i in range(M, len(pastM_dataX[0])):
    def volume(x, i=i):
        return x[i][1]
    features.append(volume)

#Add indicator variables for each of the |M| days in |x| that indicates whether or not that stock price was above or below the mean of the stock
#prices for the |M| days.
# for i in range(M, len(pastM_dataX[0])):
#     def mean_indication(x):
#         mean = 0
#         for j in range(M, len(x)):
#             mean += x[j][0]/M
#         return 1.0 if x[i][0] >= mean else -1.0
#     features.append(mean_indication)

#Add feature that captures the trend of the market. The slope of the market is weighted towards the latest stock price.
def weighted_trend(x):
    weighted_slope = 0

    weights = [math.exp(i) for i in range(0, M)]
    weights_total = sum(weights)
    for i in range(M, len(x) - 1):
        weighted_slope += weights[i - M]*(x[i + 1][0] - x[i][0])/weights_total
    return weighted_slope
features.append(weighted_trend)

#Standard deviation of the past |M| days of stock prices. 
# def std(x):
#     mean = 0
#     for j in range(M, len(x)):
#         mean += x[j][0]/M
#     variance = 0
#     for j in range(M, len(x)):
#         variance += (x[j][0] - mean)**2/M
#     return variance**0.5
# features.append(std)

# #Here we load our word2vec model, which was trained on all our tweet data. From this model, we will extract features from 
# #tweets to use in our stock prediciton model. 

# w2v = Word2Vec.load('../datasets/w2v_tweets.bin')

# #Add a feature that outputs the 
# word_list = [
#                 ['increase'] + [word for word, score in w2v.most_similar('increase', topn=3)],
#                 ['decrease'] + [word for word, score in w2v.most_similar('decrease', topn=3)],
#                 ['bankrupt'] + [word for word, score in w2v.most_similar('bankrupt', topn=3)],
#                 ['fraud']    + [word for word, score in w2v.most_similar('fraud', topn=3)],
#                 ['love']     + [word for word, score in w2v.most_similar('love', topn=3)],
#                 ['short']     + [word for word, score in w2v.most_similar('short', topn=3)]
#             ]


# for words in word_list:
#     for i in range(M):
#         def words_presence(x, i=i, words=words):
#             total_replies_retweets_and_num_favorites = 0
#             weighted_avg = 0
#             for tweet, num_replies, num_retweets, num_favorites in x[i]:
#                 weighted_avg += sum([(1.0 if word in words else 0.0) * (num_replies + num_retweets + num_favorites) for word in tweet.split()])
#                 total_replies_retweets_and_num_favorites += num_replies + num_retweets + num_favorites
#             weighted_avg /= total_replies_retweets_and_num_favorites
#             return float(weighted_avg)
#         features.append(words_presence)

# #The above word_presence feature multiplied by the std feature.
# for words in word_list:
#     for i in range(M):
#         def word_presence_and_std(x, i=i, words=words):
#             return std(x) * words_presence(x, i=i, words=words)
#         features.append(word_presence_and_std)


# for i in range(len(word_list)):
#     similars = model.wv.most_similar(word_list[i], topn=5)
#     similars.append((word_list[i], 1.0))
#     word_and_similars_list.append(similars)

# for i in range(len(word_and_similars_list)):
#     similars = word_and_similars_list[i]
#     def totalValue(x):
#         value = 0.0
#         for word_value in similars:
#             for j in range(M):
#                 for tweet in x[j]:
#                     value += tweet[0].count(word_value[0]) * sum(tweet[1:]) * word_value[1]
#         return value
#     features.append(totalValue)

##########################################################################################################################################
'''
Using these feature extractors, we will construct feature vectors, |phi|, where |phi| is a list of feature values and then construct a new 
dataset |dataPHI|. |dataPHI| will be directly fed into our proposed learning models. 

We also normalize each feature vector |phi| via standardization because the scales for each feature may be very different.
'''

print('Computing dataset feature vectors...')
dataPHI = []
for x in pastM_dataX:
    phi = compute_phi(x)
    dataPHI.append(phi)

# for i in range(len(dataPHI)):
#     for j in range(len(dataPHI[0])):
#         if not isinstance(dataPHI[i][j], float) and not isinstance(dataPHI[i][j], int):
#             print i, j
#             print dataPHI[i][j]
#             print type(dataPHI[i][j])
#             raise Exception()
dataPHI = preprocessing.scale(dataPHI)

#Hack: The bias feature becomes 0 after scaling, so we will re-add it here.
for phi in dataPHI:
    phi[0] = 1.0


##########################################################################################################################################
'''
Next, we will want to split our data into a training set, development set, and test set in order to cross-validate and tune hyperparameters. 
From |dataPHI| and |dataY|, we will allocate the latter |TEST_SIZE|*100% of it to be the test set. Then, we will allocate the former 
|TRAIN_SIZE|*100% to be the train set. From the train set, we will do k-fold cross-validation to tune hyperparameters such that the dev
error is small.
'''

TRAIN_SIZE = 0.85
TEST_SIZE = 1 - TRAIN_SIZE

print('Splitting data into train ({}%) and test ({}%) sets...\n'.format(TRAIN_SIZE*100, TEST_SIZE*100))
dataPHI_train, dataPHI_test, dataY_train, dataY_test = train_test_split(dataPHI, next_price_dataY, test_size=TEST_SIZE, random_state=0, shuffle=False)

##########################################################################################################################################
'''
Here we declare a KFold object, |kf|, which allows us to split our train set into a sub-train set and a dev set. 4/|N_SPLITS| of the 
train set will be allocated towards the sub-train set and 1/|N_SPLITS| will be allocated towards the dev set. 
'''

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, random_state=1, shuffle=True)

##########################################################################################################################################
'''
Now we conduct k-fold cross-validation on our linear regression model. This process will split the train set into a sub-train set and a dev
set 5 different times, each with a different 1/|N_SPLITS| of the train set being allocated ot the dev set. We would like to then find the 
average sub-train mean squared error (MSE), average sub-train score, average dev MSE, and average dev score from the |N_SPLITS| 
iterations of the k-fold cross-validation. After observing these metrics, we tune hyperparameters such that the dev MSE
is small and similar to the test mean squared error.
'''

print('Starting k-fold cross-validation for linear regression model with fold = {}'.format(N_SPLITS))

average_kfold_subtrain_MSE = 0
average_kfold_subtrain_score = 0
average_kfold_dev_error = 0
average_kfold_dev_score = 0

for subtrain_i, dev_i in kf.split(dataPHI_train):
    dataPHI_kfold_subtrain = [dataPHI_train[i] for i in subtrain_i]
    dataY_kfold_subtrain = [dataY_train[i] for i in subtrain_i]
    dataPHI_kfold_dev = [dataPHI_train[i] for i in dev_i]
    dataY_kfold_dev = [dataY_train[i] for i in dev_i]

    lr = LinearRegression().fit(dataPHI_kfold_subtrain, dataY_kfold_subtrain)
    predY_kfold_subtrain = lr.predict(dataPHI_kfold_subtrain)
    predY_kfold_dev = lr.predict(dataPHI_kfold_dev)

    average_kfold_subtrain_MSE += mean_squared_error(predY_kfold_subtrain, dataY_kfold_subtrain)/N_SPLITS
    average_kfold_subtrain_score += lr.score(dataPHI_kfold_subtrain, dataY_kfold_subtrain)/N_SPLITS
    average_kfold_dev_error += mean_squared_error(predY_kfold_dev, dataY_kfold_dev)/N_SPLITS
    average_kfold_dev_score += lr.score(dataPHI_kfold_dev, dataY_kfold_dev)/N_SPLITS

print('k-fold evaluations for linear regression model:')
print('k-fold train mean squared error: {}'.format(average_kfold_subtrain_MSE))
print('k-fold train score: {}'.format(average_kfold_subtrain_score))
print('k-fold dev mean squared error: {}'.format(average_kfold_dev_error))
print('k-fold dev score: {}'.format(average_kfold_dev_score))

##########################################################################################################################################
'''
Upon observing that our dev MSE is small, we will now train our linear regression model on the entire train set, and compare the MSE of the
train set with the MSE of the test set. If they are similar, we have a generalized model. If they are both small, we have a good model.
'''
print('\nTraining linear regression model on entire train set...\n')
lr = LinearRegression().fit(dataPHI_train, dataY_train)
predY_train = lr.predict(dataPHI_train)
predY_test = lr.predict(dataPHI_test)
print('Train mean squared error: {}'.format(mean_squared_error(predY_train, dataY_train)))
print('Train score: {}'.format(lr.score(dataPHI_train, dataY_train)))
print('Test mean squared error: {}'.format(mean_squared_error(predY_test, dataY_test)))
print('Test score: {}\n'.format(lr.score(dataPHI_test, dataY_test)))

##########################################################################################################################################
'''
Now we will use the same methods for k-fold cross-validation, but applied to our neural network model and tune hyperparameters in the same
manner. 
'''
HIDDEN_LAYER_SIZES = ((len(features) + 1)/2)
ACTIVATION = 'tanh'
MAX_ITER = 10000
LEARNING_RATE_INIT = 0.03

print('Starting k-fold cross-validation for neural network model with fold = {}'.format(N_SPLITS))

average_kfold_subtrain_MSE = 0
average_kfold_subtrain_score = 0
average_kfold_dev_error = 0
average_kfold_dev_score = 0

for subtrain_i, dev_i in kf.split(dataPHI_train):
    dataPHI_kfold_subtrain = [dataPHI_train[i] for i in subtrain_i]
    dataY_kfold_subtrain = [dataY_train[i] for i in subtrain_i]
    dataPHI_kfold_dev = [dataPHI_train[i] for i in dev_i]
    dataY_kfold_dev = [dataY_train[i] for i in dev_i]

    nn = MLPRegressor(hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation=ACTIVATION, max_iter=MAX_ITER, learning_rate_init=LEARNING_RATE_INIT).fit(dataPHI_kfold_subtrain, dataY_kfold_subtrain)
    predY_kfold_subtrain = nn.predict(dataPHI_kfold_subtrain)
    predY_kfold_dev = nn.predict(dataPHI_kfold_dev)

    average_kfold_subtrain_MSE += mean_squared_error(predY_kfold_subtrain, dataY_kfold_subtrain)/N_SPLITS
    average_kfold_subtrain_score += nn.score(dataPHI_kfold_subtrain, dataY_kfold_subtrain)/N_SPLITS
    average_kfold_dev_error += mean_squared_error(predY_kfold_dev, dataY_kfold_dev)/N_SPLITS
    average_kfold_dev_score += nn.score(dataPHI_kfold_dev, dataY_kfold_dev)/N_SPLITS

print('k-fold evaluations for neural network model:')
print('k-fold train mean squared error: {}'.format(average_kfold_subtrain_MSE))
print('k-fold train score: {}'.format(average_kfold_subtrain_score))
print('k-fold dev mean squared error: {}'.format(average_kfold_dev_error))
print('k-fold dev score: {}'.format(average_kfold_dev_score))

##########################################################################################################################################
'''
Upon observing that our dev MSE is small, we will now train our neural network model on the entire train set, and compare the MSE of the
train set with the MSE of the test set. If they are similar, we have a generalized model. If they are both small, we have a good model.
'''
print('\nTraining neural network model on entire train set...\n')
nn = MLPRegressor(hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation=ACTIVATION, max_iter=MAX_ITER, learning_rate_init=LEARNING_RATE_INIT).fit(dataPHI_train, dataY_train)
predY_train = nn.predict(dataPHI_train)
predY_test = nn.predict(dataPHI_test)
print('Train mean squared error: {}'.format(mean_squared_error(predY_train, dataY_train)))
print('Train score: {}'.format(nn.score(dataPHI_train, dataY_train)))
print('Test mean squared error: {}'.format(mean_squared_error(predY_test, dataY_test)))
print('Test score: {}'.format(nn.score(dataPHI_test, dataY_test)))


def percent_correct(predictions, actuals):
    count = 0.0
    for i in range(len(stock_prices_data)):
        if stock_prices_data[i][0] in actuals:
            previous = stock_prices_data[i-1][0]
            index = actuals.index(stock_prices_data[i][0])
            if (actuals[index] >= previous and predictions[index] >= previous) or (actuals[index] <= previous and predictions[index] <= previous):
                count += 1
    return count / len(predictions)
print 'Train percent correct', percent_correct(predY_train, dataY_train)
print 'Test percent correct', percent_correct(predY_test, dataY_test)

############################################################################################################################
'''
Here we will plot important results.
'''

pca = PCA(n_components=2)
pca_dataPHI = pca.fit_transform(dataPHI)
print('The explained variance for each principal component is {} (should be >= 85%).'.format(pca.explained_variance_ratio_)) 

pc = [pca_dataPHI[i][0] for i in range(len(pca_dataPHI))]












