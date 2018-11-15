import csv
from collections import defaultdict
def findDaysMissed(expectedMonth, expectedDay, month, day, monthToDay):
    daysMissed = 0
    while expectedMonth != month or expectedDay != day:
        if expectedDay + 1 > monthToDay[expectedMonth - 1]:
            expectedMonth += 1
            expectedDay = 1
        else:
            expectedDay += 1
            daysMissed += 1
    return daysMissed

def fillMissingDays(dateToClosingPrice, expectedMonth, expectedDay, daysMissed, newPrice, year, monthToDay):
    previousPrice = dateToClosingPrice[len(dateToClosingPrice) - 1][1]
    for i in range(daysMissed):
        date = str(expectedMonth) + '/' + str(expectedDay) + '/' + year
        missingPrice = (previousPrice + newPrice) / 2
        dateToClosingPrice.append((date, missingPrice))
        previousPrice = missingPrice
        #update day
        #if we are going to a new month
        if expectedDay + 1 > monthToDay[expectedMonth - 1]:
            expectedMonth += 1
            expectedDay = 1
        else:
            expectedDay += 1
    return (expectedMonth, expectedDay)
				
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
                daysMissed = findDaysMissed(expectedMonth, expectedDay, int(month), int(day), monthToDay)
                expectedMonth, expectedDay = fillMissingDays(dateToClosingPrice, expectedMonth, expectedDay, daysMissed, float(row[4]), year, monthToDay)
                
                date = month + '/' + day + '/' + year
                dateToClosingPrice.append((date, float(row[4])))
                
            if expectedDay + 1 > monthToDay[expectedMonth - 1]:
                expectedMonth += 1
                expectedDay = 1
            else:
                expectedDay += 1
            line += 1
            
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

