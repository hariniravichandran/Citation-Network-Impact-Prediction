import re
import numpy
import pandas

from keras.models import Sequential
from keras.layers.core import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#knn
import numpy as np
from sklearn.metrics import *
from sklearn.neighbors import *

#svm
from sklearn import svm

#krr
from sklearn.kernel_ridge import KernelRidge

#LR
from sklearn import linear_model

def init():
    global pubDict, mappings, authorsDict, venuesDict, featureVector, citationVector, scaledFeatureVector, years
    mappings = {'#*': 'title', '#c': 'venue', '#!': 'abstract', '#@': 'authors', '#%': 'references', '#index': 'pubId', '#t': 'year'}
    pubDict = {}
    authorsDict = {}
    venuesDict = {}
    featureVector = []
    scaledFeatureVector = []
    citationVector = []
    years = []


def getLines(path):
    with open(path, 'r') as f:
        for line in f:
            yield line

def initAuthorDict(author):
    global authorsDict    
    authorsDict[author] = {}
    authorsDict[author]['pubList'] = []
    authorsDict[author]['pubCount'] = 0
    authorsDict[author]['citationCount'] = 0
    authorsDict[author]['citationCountByYear'] = {}
    authorsDict[author]['pubYears'] = {}
    # authorsDict[author]['coAuthors'] = {}
    # authorsDict[author]['coAuthorCounts'] = {}    
    # authorsDict[author]['selfCitationCount'] = 0
    # authorsDict[author]['coAuthorCitationCount'] = 0 

def initVenueDict(venue):
    global venuesDict
    venuesDict[venue] = {}
    venuesDict[venue]['pubList'] = []
    venuesDict[venue]['pubCount'] = 0
    venuesDict[venue]['citationCount'] = 0
    venuesDict[venue]['citationCountByYear'] = {}
    venuesDict[venue]['pubYears'] = {}

def getCitations():
    global pubDict, authorsDict, venuesDict
    print "getting citations"

    # Citations breakdown by publication (total and yearwise)
    print "Publication Level Citations"
    for pubId in pubDict:
        references = pubDict[pubId]['references']
        for reference in references:
            if reference in pubDict:
                pubDict[reference]['citedBy'].append(pubId)
                pubDict[reference]['citationCount'] += 1
                if pubDict[pubId]['year'] not in pubDict[reference]['citationCountByYear']:
                    pubDict[reference]['citationCountByYear'][pubDict[pubId]['year']] = 0
                pubDict[reference]['citationCountByYear'][pubDict[pubId]['year']] += 1 
    
    # Citations breakdown by author (total and yearwise)
    print "Author Level Citations"
    for author in authorsDict:
        pubList = authorsDict[author]['pubList']
        for pubId in pubList:
            authorsDict[author]['citationCount'] += pubDict[pubId]['citationCount']
            if pubDict[pubId]['year'] not in authorsDict[author]['citationCountByYear']:
                authorsDict[author]['citationCountByYear'][pubDict[pubId]['year']] = 0
            authorsDict[author]['citationCountByYear'][pubDict[pubId]['year']] += 1 

        # coAuthors = authorsDict[author]['coAuthors']
        # for coAuthor in coAuthors:
        #     pubList = coAuthors[coAuthor]['pubList']
        #     for pubId in pubList:
        #         coAuthors[coAuthor]['citeCount'] += pubDict[pubId]['citationCount']
    
    # Citations breakdown by venue (total and yearwise)
    print "Venue Level Citations"
    for venue in venuesDict:
        pubList = venuesDict[venue]['pubList']
        for pubId in pubList:
            venuesDict[venue]['citationCount'] += pubDict[pubId]['citationCount']
            if pubDict[pubId]['year'] not in venuesDict[venue]['citationCountByYear']:
                venuesDict[venue]['citationCountByYear'][pubDict[pubId]['year']] = 0
            venuesDict[venue]['citationCountByYear'][pubDict[pubId]['year']] += 1 

def populateAuthorDict(pubData):
    global authorsDict
    for author in pubData['authors']:
        if author not in authorsDict:
            initAuthorDict(author)
        authorsDict[author]['pubList'].append(pubData['pubId'])
        if pubData['year'] not in authorsDict[author]['pubYears']:
            authorsDict[author]['pubYears'][pubData['year']] = 0
        authorsDict[author]['pubYears'][pubData['year']] += 1
        # coAuthors = [x for x in authors if x != author]
        # for coAuthor in coAuthors:
        #     if coAuthor not in authorsDict[author]['coAuthors']:
        #         authorsDict[author]['coAuthors'][coAuthor] = {}
        #         authorsDict[author]['coAuthors'][coAuthor]['pubList'] = []
        #         authorsDict[author]['coAuthors'][coAuthor]['pubCount'] = 0
        #         authorsDict[author]['coAuthors'][coAuthor]['citeCount'] = 0
        #     authorsDict[author]['coAuthors'][coAuthor]['pubList'].append(pubData['pubId'])
        #     authorsDict[author]['coAuthors'][coAuthor]['pubCount'] += 1
        # coAuthorCount = len(coAuthors)
        # if coAuthorCount not in authorsDict[author]['coAuthorCounts']:
        #     authorsDict[author]['coAuthorCounts'][coAuthorCount] = 0
        # authorsDict[author]['coAuthorCounts'][coAuthorCount] += 1
        authorsDict[author]['pubCount'] += 1

def populateVenueDict(pubData):
    global venuesDict
    if pubData['venue'] not in venuesDict:
        initVenueDict(pubData['venue'])
    venuesDict[pubData['venue']]['pubList'].append(pubData['pubId'])
    if pubData['year'] not in venuesDict[pubData['venue']]['pubYears']:
        venuesDict[pubData['venue']]['pubYears'][pubData['year']] = 0
    venuesDict[pubData['venue']]['pubYears'][pubData['year']] += 1
    venuesDict[pubData['venue']]['pubCount'] += 1

def getAuthorFeatureCountUntilYear(author, feature, publishedYear):
    totalCount = 0
    for year, count in authorsDict[author][feature].items():
        if year <= publishedYear:
            totalCount += count
    return totalCount

def populateAuthorSecondaryFeatures():
    global authorsDict, pubDict
    print "Getting secondary features for author" 
    # for author, data in authorsDict.iteritems():  
    for author in authorsDict:
        citationCountForPublication = []                     
        for publication in authorsDict[author]['pubList']:                       
            # publicationData = pubDict[publication]                
            citationCountForPublication.append(pubDict[publication]['citationCount'])
            # selfCitation = [i for i, j in zip(data['pubList'], publicationData['references']) if i == j]
            # data['selfCitationCount'] += len(selfCitation)
            # for coAuthor in data['coAuthors']:
            #     coAuthorCitation = [i for i, j in zip(authorsDict[coAuthor]['pubList'], publicationData['references']) if i == j and i not in data['pubList']]                
            #     if len(coAuthorCitation) > 0:
            #         data['coAuthorCitationCount'] += len(coAuthorCitation)
        citationCountForPublication.sort(reverse=True)
        hIndex = 0
        i = 1
        while(i<=len(citationCountForPublication)):
            if i <= citationCountForPublication[i-1]:
                hIndex = i
            i += 1
        authorsDict[author]['hIndex'] = hIndex

def filterPapers(pubId):
    global pubDict
    if not (pubDict[pubId]['authors'] and \
        pubDict[pubId]['venue'] and \
        pubDict[pubId]['year'] and \
        (int(pubDict[pubId]['year']) >= 1930 and \
        int(pubDict[pubId]['year']) <= 2015)):

        del pubDict[pubId]
        return True
    return False

def populateDicts(path, trainingYear = 3, predictingYear = 10):
    global pubDict, mappings, authorsDict
    i = 0
    tempDict = {'title':None, 'venue':None, 'abstract':None, 'authors':[], 'references':[], 'pubId':None, 'year':None}
    for line in getLines(path):
        i += 1
        if i % 500000 == 0:
            print i
        if line.startswith('#'):
            if line[:2] in mappings or line[:6] in mappings:
                currLineKey = line[:6] if line[:6] in mappings else line[:2]
                if currLineKey == '#%':
                    if not tempDict[mappings[currLineKey]]:
                        tempDict[mappings[currLineKey]] = []
                    reference = line[2:].strip().replace('\r','').replace('\n','') if line[2:].strip() else None
                    if reference:
                        tempDict[mappings[currLineKey]].append(reference)
                else:
                    if mappings[currLineKey] in tempDict and not tempDict[mappings[currLineKey]]:
                        tempDict[mappings[currLineKey]] = line[6:].replace('\r','').replace('\n','') if line[:6] in mappings else line[2:].replace('\r','').replace('\n','')
        else:
            pubId = tempDict['pubId']
            # if tempDict['year'] == '-1':
            #     continue
            if pubId not in pubDict:
                pubDict[pubId] = {}
                for k in tempDict:
                    pubDict[pubId][k] = tempDict[k]
                filtered = filterPapers(pubId)
                if filtered:
                    continue
                pubDict[pubId]['year'] = int(pubDict[pubId]['year'])
                pubDict[pubId]['authors'] = [author.strip() for author in pubDict[pubId]['authors'].strip().split(',')]
                authors = pubDict[pubId]['authors']
                pubDict[pubId]['authorCount'] = len(authors) if authors else 0
                pubDict[pubId]['venue'] = pubDict[pubId]['venue'].strip()
                references = pubDict[pubId]['references']
                pubDict[pubId]['referenceCount'] = len(references) if references else 0
                pubDict[pubId]['citationCount'] = 0
                pubDict[pubId]['citedBy'] = []
                pubDict[pubId]['citationCountByYear'] = {}
                if tempDict['year'] not in years:
                    years.append(tempDict['year'])
            tempDict = {key:None for key in tempDict}
            tempDict['references'] = []
            tempDict['authors'] = []
            populateAuthorDict(pubDict[pubId])
            populateVenueDict(pubDict[pubId])
    getCitations()
    for pubId, pubData in pubDict.items():
        publishedYear = pubData['year']        
        pubData['trainYearsCitationCount'] = 0        
        pubData['testYearsCitationCount'] = 0 
        pubData['trainYearsPublication'] = []        
        pubData['testYearsPublication'] = []
        for citingPaperId in pubData['citedBy']:            
            if pubDict[citingPaperId]['year'] <= (publishedYear + trainingYear):
                pubData['trainYearsCitationCount'] += 1
                pubData['trainYearsPublication'].append(citingPaperId)
            else:
                pubData['testYearsCitationCount'] += 1
                pubData['testYearsPublication'].append(citingPaperId)

    populateAuthorSecondaryFeatures()

def getHindexForYear(author, givenYear):
    global authorsDict, pubDict
    authorPublicationList =[]
    for pubId in authorsDict[author]['pubList']:
        publication = pubDict[pubId]
        if int(publication['year']) > givenYear:
            continue
        else:
            citationCountUntilYear = sum([citation for year, citation in publication['citationCountByYear'].items() if int(year) <= givenYear])
            print pubId, " ", citationCountUntilYear
            print publication['citationCountByYear']
            authorPublicationList.append(citationCountUntilYear)
    authorPublicationList.sort(reverse=True)
    hIndex = 0
    i = 1
    while(i<=len(authorPublicationList)):
        if i <= authorPublicationList[i-1]:
            hIndex = i
        i += 1
    return hIndex

def buildFeatureVector(trainingYear = 3, predictingYear = 10, timeFrameEnd = 2013):
    global pubDict, authorsDict, venuesDict, featureVector, years, citationVector
    for pubId in pubDict:
        pubFeatures = []
        authorFeatures = []
        venueFeatures = []
        allFeatures = []
        publishedYear = pubDict[pubId]['year']   
        if publishedYear > timeFrameEnd - predictingYear:               
            continue
        # Venue Level Features
        venue = pubDict[pubId]['venue']
        venuePubCounts = [0] * (trainingYear+1)
        venueCitationCounts = [0] * (trainingYear+1)
        for iterYear in range(trainingYear + 1):            
            venuePubCounts[iterYear] = venuesDict[venue]['pubYears'].get(publishedYear+iterYear, 0)
            venueCitationCounts[iterYear] = venuesDict[venue]['citationCountByYear'].get(publishedYear+iterYear, 0)            
        venueFeatures.append(sum(venuePubCounts))
        # venueFeatures.extend(venuePubCounts)
        venueFeatures.append(sum(venueCitationCounts))
        # venueFeatures.extend(venueCitationCounts)

        # Publication Level Features
        pubCiteCntByYear = [0] * (trainingYear + 1)
        for iterYear in range(trainingYear+1):
            pubCiteCntByYear[iterYear] = pubDict[pubId]['citationCountByYear'].get(publishedYear+iterYear, 0)           
        
        pubFeatures.extend(pubCiteCntByYear)
        pubFeatures.append(len(pubDict[pubId]['authors']))

        # Author Level Features
        # authors = pubDict[pubId]['authors']
        authorCitationCounts = [[getAuthorFeatureCountUntilYear(author, 'citationCountByYear', publishedYear), author] for author in pubDict[pubId]['authors']]
        authorCitationCounts.sort(reverse=True)
        for citationCount, author in authorCitationCounts[:2]:            
            authorPubCount = getAuthorFeatureCountUntilYear(author, 'pubYears', publishedYear)
            authorPubYears = []
            for year in authorsDict[author]['pubYears']:  
                if year <= publishedYear:              
                    authorPubYears.append(year)

            authorPubYears.sort()
            # authorPubYearsDiff = [j-i for i, j in zip(authorPubYears[:-1], authorPubYears[1:])]    
            # minGap = min(authorPubYearsDiff) if authorPubYearsDiff else 0
            # maxGap = max(authorPubYearsDiff) if authorPubYearsDiff else 0

            # authorFeatures.append(authorsDict[author]['hIndex'])
            authorFeatures.append(getHindexForYear(author, publishedYear))
            authorFeatures.append(authorPubCount)
            authorFeatures.append(citationCount)

            # authorFeatures.append(minGap)
            # authorFeatures.append(maxGap)
            # authorFeatures.append(min(authorsDict[author]['pubYears'].values()))
            # authorFeatures.append(max(authorsDict[author]['pubYears'].values()))
        if len(authorCitationCounts) < 2:
            authorFeatures = authorFeatures + [0] * (2-len(authorFeatures))

        allFeatures = pubFeatures + authorFeatures + venueFeatures
        featureVector.append(allFeatures)

        # Citation counts (Y value)
        citationOverTestPeriod = 0
        for year, count in pubDict[pubId]['citationCountByYear'].items():            
            if year <= publishedYear + predictingYear:                 
                citationOverTestPeriod += count
        citationVector.append(citationOverTestPeriod)

def buildNN(features):
    featuresLength = len(features[0])
    model = Sequential()
    model.add(Dense(featuresLength, input_dim=featuresLength, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(featuresLength/2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def runNN(features, predictions, epochs=5):
    xTrain, xTest, yTrain, yTest = train_test_split(features, predictions, test_size=0.25)
    model = buildNN(features)
    #estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
    model.fit(xTrain, yTrain, epochs=epochs, batch_size=32)
    predictedY = model.predict(xTest, batch_size=128)
    predictedY = [round(y) for y in predictedY]
    mse = mean_squared_error(yTest, predictedY)
    correct = sum([1 for i in xrange(len(xTest)) if yTest[i] == predictedY[i]])
    print "NN accuracy = ", correct/float(len(yTest))
    print "NN MSE = ", mse
    return [yTest, predictedY, correct, mse]

def runKNN(features, predictions):
    accuracies = []
    xTrain, xTest, yTrain, yTest = train_test_split(features, predictions, test_size=0.25)
    print "KNN Model"
    for n in range(1, 6):
        print "Training with ", n, " neighbours, accuracy, mse = ",
        model = KNeighborsRegressor(n_neighbors=n, algorithm='auto')
        model.fit(np.array(xTrain), np.array(yTrain))
        predictedY = model.predict(xTest)
        predictedY = [round(y) for y in predictedY]
        accuracy = accuracy_score(yTest, predictedY)
        mse = mean_squared_error(yTest, predictedY)
        print accuracy, mse
        accuracies.append([n, accuracy, mse, yTest, predictedY])
    return accuracies

def runSVR(features, predictions):
    accuracies = []
    xTrain, xTest, yTrain, yTest = train_test_split(features, predictions, test_size=0.25)
    print "Training SVM"
    model = svm.SVR(kernel='rbf', cache_size=500, C = 1)
    model.fit(np.array(xTrain), np.array(yTrain))
    print "fitted model"
    predictedY = model.predict(xTest)
    predictedY = [round(y) for y in predictedY]
    print "Predictions completed"
    accuracy = accuracy_score(yTest, predictedY)
    mse = mean_squared_error(yTest, predictedY)
    print "accuracy = ", accuracy
    print "mse = ", mse
    accuracies.append([n, accuracy, mse, yTest, predictedY])
    return accuracies
    #return [yTest, predictedY, accuracy]

def runLR(features, predictions):
    accuracies = []
    xTrain, xTest, yTrain, yTest = train_test_split(features, predictions, test_size=0.25)
    regr = linear_model.LinearRegression()
    regr.fit(np.array(xTrain), np.array(yTrain))
    predictedY = regr.predict(xTest)
    predictedY = [round(y) for y in predictedY]
    mse = np.mean((np.array(predictedY) - np.array(yTest)) ** 2)
    print("Mean squared error: %.2f"% mse)
    var = regr.score(xTest, yTest)
    print('Variance score: %.2f' % var)
    accuracies.append([mse, yTest, predictedY])
    return accuracies

def runRidge(features, predictions):
    accuracies = []
    xTrain, xTest, yTrain, yTest = train_test_split(features, predictions, test_size=0.25)
    regr = linear_model.Ridge(alpha=0.1)
    regr.fit(np.array(xTrain), np.array(yTrain))
    predictedY = regr.predict(xTest)
    predictedY = [round(y) for y in predictedY]
    mse = np.mean((np.array(predictedY) - np.array(yTest)) ** 2)
    print("Mean squared error: %.2f"% mse)
    var = regr.score(xTest, yTest)
    print('Variance score: %.2f' % var)
    accuracies.append([mse, yTest, predictedY])
    return accuracies

def scaleFeatures():
    global featureVector, scaledFeatureVector
    tempFeature = zip(*featureVector)
    minMax = [[min(x), max(x)] for x in tempFeature]
    for fv in featureVector:
        scaledFeatureVector.append([])
        for pos in range(len(fv)):
            if minMax[pos][1] == minMax[pos][0]:
                scaledFeatureVector[-1].append(0)
            else:
                scaledFeatureVector[-1].append((fv[pos]-minMax[pos][0])/float(minMax[pos][1]-minMax[pos][0]))

def checkCitationDistribution():
    global pubDict
    res = []
    hist = {}
    for pubId in pubDict:
        pubYear = pubDict[pubId]['year']
        pubYears = pubDict[pubId]['citationCountByYear'].keys()
        pubYearsCount = [pubDict[pubId]['citationCountByYear'][x] for x in pubYears if int(x) > (int(pubYear)+10)]
        if pubYearsCount:
            res.append([pubId, sum(pubYearsCount)])
        for year in pubYears:
            bucket = int(year) - int(pubYear)
            if bucket not in hist:
                hist[bucket] = 0
            hist[bucket] += 1
    return hist

init()
#populateDicts('./testData.txt')
populateDicts('/Users/hariniravichandran/Documents/SML/data/citation-acm-v8.txt')
buildFeatureVector()
scaleFeatures()
NNResults = runNN(featureVector, citationVector, 10)
KNNResults = runKNN(featureVector, citationVector)
SVRResults = runSVR(featureVector, citationVector)
KRRResults = runKRR(featureVector, citationVector)
LRResults = runLR(featureVector, citationVector)
RidgeResults = runRidge(featureVector, citationVector)

print "scaled version"
NNResultsScaled = runNN(scaledFeatureVector, citationVector, 10)
KNNResultsScaled = runKNN(scaledFeatureVector, citationVector)
SVRResultsScaled = runSVR(scaledFeatureVector, citationVector)

# res = checkCitationDistribution()
