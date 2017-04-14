import re
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
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

def init():
    global pubDict, mappings, authorsDict, venuesDict, featureVector, citationVector
    mappings = {'#*': 'title', '#c': 'venue', '#!': 'abstract', '#@': 'authors', '#%': 'references', '#index': 'pubId', '#t': 'year'}
    pubDict = {}
    authorsDict = {}
    venuesDict = {}
    featureVector = []
    citationVector = []

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
    authorsDict[author]['coAuthors'] = {}
    authorsDict[author]['coAuthorCounts'] = {}    
    authorsDict[author]['selfCitationCount'] = 0
    authorsDict[author]['coAuthorCitationCount'] = 0 

def initVenueDict(venue):
    global venuesDict
    venuesDict[venue] = {}
    venuesDict[venue]['pubList'] = []
    venuesDict[venue]['pubCount'] = 0
    venuesDict[venue]['citationCount'] = 0
    venuesDict[venue]['citationCountByYear'] = {}
    venuesDict[venue]['pubYears'] = {}


def populateAuthorDict(pubData):
    global authorsDict
    authors = pubData['authors']
    for author in authors:
        if author not in authorsDict:
            initAuthorDict(author)
        authorsDict[author]['pubList'].append(pubData['pubId'])
        if pubData['year'] not in authorsDict[author]['pubYears']:
            authorsDict[author]['pubYears'][pubData['year']] = 0
        authorsDict[author]['pubYears'][pubData['year']] += 1
        coAuthors = [x for x in authors if x != author]
        for coAuthor in coAuthors:
            if coAuthor not in authorsDict[author]['coAuthors']:
                authorsDict[author]['coAuthors'][coAuthor] = {}
                authorsDict[author]['coAuthors'][coAuthor]['pubList'] = []
                authorsDict[author]['coAuthors'][coAuthor]['pubCount'] = 0
                authorsDict[author]['coAuthors'][coAuthor]['citeCount'] = 0
            authorsDict[author]['coAuthors'][coAuthor]['pubList'].append(pubData['pubId'])
            authorsDict[author]['coAuthors'][coAuthor]['pubCount'] += 1
        coAuthorCount = len(coAuthors)
        if coAuthorCount not in authorsDict[author]['coAuthorCounts']:
            authorsDict[author]['coAuthorCounts'][coAuthorCount] = 0
        authorsDict[author]['coAuthorCounts'][coAuthorCount] += 1
        authorsDict[author]['pubCount'] += 1

def populateVenueDict(pubData):
    global venuesDict
    venue = pubData['venue']
    if venue not in venuesDict:
        initVenueDict(venue)
    venuesDict[venue]['pubList'].append(pubData['pubId'])
    if pubData['year'] not in venuesDict[venue]['pubYears']:
        venuesDict[venue]['pubYears'][pubData['year']] = 0
    venuesDict[venue]['pubYears'][pubData['year']] += 1
    venuesDict[venue]['pubCount'] += 1


def populateDicts(path):
    global pubDict, mappings, authorsDict
    tempDict = {'title':None, 'venue':None, 'abstract':None, 'authors':None, 'references':None, 'pubId':None, 'year':None}
    for line in getLines(path):
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
            if pubId not in pubDict:
                pubDict[pubId] = {}
                for k in tempDict:
                    pubDict[pubId][k] = tempDict[k]
                pubDict[pubId]['authors'] = [author.strip() for author in pubDict[pubId]['authors'].strip().split(',')]
                authors = pubDict[pubId]['authors']
                pubDict[pubId]['authorCount'] = len(authors) if authors else 0
                pubDict[pubId]['venue'] = pubDict[pubId]['venue'].strip()
                references = pubDict[pubId]['references']
                pubDict[pubId]['referenceCount'] = len(pubDict[pubId]['references'])
                pubDict[pubId]['citationCount'] = 0
                pubDict[pubId]['citedBy'] = []
                pubDict[pubId]['citationCountByYear'] = {}
            tempDict = {key:None for key in tempDict}
            populateAuthorDict(pubDict[pubId])
            populateVenueDict(pubDict[pubId])
    getCitations()
    populateAuthorSecondaryFeatures()


def getCitations():
    global pubDict, authorsDict, venuesDict
    print "getting citations"

    # Citations breakdown by publication (total and yearwise)
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
    for author in authorsDict:
        pubList = authorsDict[author]['pubList']
        for pubId in pubList:
            authorsDict[author]['citationCount'] += pubDict[pubId]['citationCount']
            if pubDict[pubId]['year'] not in authorsDict[author]['citationCountByYear']:
                authorsDict[author]['citationCountByYear'][pubDict[pubId]['year']] = 0
            authorsDict[author]['citationCountByYear'][pubDict[pubId]['year']] += 1 

        coAuthors = authorsDict[author]['coAuthors']
        for coAuthor in coAuthors:
            pubList = coAuthors[coAuthor]['pubList']
            for pubId in pubList:
                coAuthors[coAuthor]['citeCount'] += pubDict[pubId]['citationCount']
    
    # Citations breakdown by venue (total and yearwise)
    for venue in venuesDict:
        pubList = venuesDict[venue]['pubList']
        for pubId in pubList:
            venuesDict[venue]['citationCount'] += pubDict[pubId]['citationCount']
            if pubDict[pubId]['year'] not in venuesDict[venue]['citationCountByYear']:
                venuesDict[venue]['citationCountByYear'][pubDict[pubId]['year']] = 0
            venuesDict[venue]['citationCountByYear'][pubDict[pubId]['year']] += 1 


def buildFeatureVector():
    global pubDict, authorsDict, venuesDict, featureVector, citationVector
    for pubId in pubDict:
        pubFeatures = []
        authorFeatures = []
        venueFeatures = []
        allFeatures = []

        # Venue Level Features
        venue = pubDict[pubId]['venue']
        venuePubCounts = [0 for i in xrange(1968, 2014)]
        for year in venuesDict[venue]['pubYears']:
            venuePubCounts[int(year) - 1968] = venuesDict[venue]['pubYears'][year]

        venueFeatures.append(venuesDict[venue]['pubCount'])
        venueFeatures.extend(venuePubCounts)
        venueFeatures.append(venuesDict[venue]['citationCount'])

        # Publication Level Features
        pubCiteCntByYear = [0 for i in xrange(1968, 2014)]
        for year in pubDict[pubId]['citationCountByYear']:
            pubCiteCntByYear[int(year) - 1968] = pubDict[pubId]['citationCountByYear'][year]

        pubFeatures.append(len(pubDict[pubId]['authors']))
        pubFeatures.extend(pubCiteCntByYear)

        # Author Level Features
        authors = pubDict[pubId]['authors']
        authorCitationCounts = [[authorsDict[author]['citationCount'], author] for author in authors]
        authorCitationCounts.sort()
        for citationCount, author in authorCitationCounts[:2]:
            authorPubCounts = [0 for i in xrange(1968, 2014)]
            authorPubYears = []
            for year in authorsDict[author]['pubYears']:
                authorPubCounts[int(year) - 1968] = authorsDict[author]['pubYears'][year]
                authorPubYears.append(int(year))

            authorPubYears.sort()
            authorPubYearsDiff = [j-i for i, j in zip(authorPubYears[:-1], authorPubYears[1:])]    
            minGap = min(authorPubYearsDiff) if authorPubYearsDiff else 0
            maxGap = max(authorPubYearsDiff) if authorPubYearsDiff else 0

            authorFeatures.append(authorsDict[author]['hIndex'])
            authorFeatures.append(authorsDict[author]['pubCount'])
            authorFeatures.append(authorsDict[author]['citationCount'])
            authorFeatures.extend(authorPubCounts)

            authorFeatures.append(minGap)
            authorFeatures.append(maxGap)
            authorFeatures.append(min(authorsDict[author]['pubYears'].values()))
            authorFeatures.append(max(authorsDict[author]['pubYears'].values()))
        if len(authorCitationCounts) < 2:
            authorFeatures = authorFeatures + [0] * len(authorFeatures)

        allFeatures = pubFeatures + authorFeatures + venueFeatures
        featureVector.append(allFeatures)

        # Citation counts (Y value)
        citationVector.append(pubDict[pubId]['citationCount'])

def populateAuthorSecondaryFeatures():
    global authorsDict, pubDict
    print "Getting secondary features for author" 
    for author, data in authorsDict.iteritems():  
        citationCountForPublication = []                     
        for publication in data['pubList']:                       
            publicationData = pubDict[publication]                
            citationCountForPublication.append(publicationData['citationCount'])
            selfCitation = [i for i, j in zip(data['pubList'], publicationData['references']) if i == j]
            data['selfCitationCount'] += len(selfCitation)
            for coAuthor in data['coAuthors']:
                coAuthorCitation = [i for i, j in zip(authorsDict[coAuthor]['pubList'], publicationData['references']) if i == j and i not in data['pubList']]                
                if len(coAuthorCitation) > 0:
                    data['coAuthorCitationCount'] += len(coAuthorCitation)
        citationCountForPublication.sort(reverse=True)
        hIndex = 0
        i = 1
        while(i<=len(citationCountForPublication)):
            if i <= citationCountForPublication[i-1]:
                hIndex = i
            i += 1
        data['hIndex'] = hIndex

def buildLR(features):
    featuresLength = len(features[0])
    model = Sequential()
    model.add(Dense(featuresLength, input_dim=featuresLength, kernel_initializer='normal', activation='relu'))
    model.add(Dense(featuresLength/2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def runLR(features, predictions):
    xTrain, xTest, yTrain, yTest = train_test_split(features, predictions, test_size=0.6)
    #print 'xtr', xTrain, 'xte', xTest, 'ytr', yTrain, 'yte', yTest
    model = buildLR(features)
    #estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
    model.fit(xTrain, yTrain, epochs=5, batch_size=32)
    predictedY = model.predict(xTest, batch_size=128)
    correct = sum([1 for i in xrange(len(xTest)) if yTest[i] == predictedY[i]])
    #print 'ytest', yTest, 'predictedY', predictedY
    return [yTest, predictedY, correct]

def runKNN(features, predictions):
    accuracies = []
    xTrain, xTest, yTrain, yTest = train_test_split(features, predictions, test_size=0.6)
    for n in range(1, 26):
        print "Training with ", n, " neighbours, accuracy = ",
        model = KNeighborsClassifier(n_neighbors=n, algorithm='auto')
        model.fit(np.array(xTrain), np.array(yTrain))
        predictedY = model.predict(xTest)
        accuracy = accuracy_score(yTest, predictedY)
        print accuracy
        accuracies.append([n, accuracy, yTest, predictedY])
    return accuracies
    #return [yTest, predictedY, accuracy]

def runSVM(features, predictions):
    accuracies = []
    xTrain, xTest, yTrain, yTest = train_test_split(features, predictions, test_size=0.6)
    for n in range(1, 2):
        print "Training SVM"
        model = svm.SVC(kernel='linear', C = n)
        model.fit(np.array(xTrain), np.array(yTrain))
        print "fitted model"
        predictedY = model.predict(xTest)
        print "Predictions completed"
        accuracy = accuracy_score(yTest, predictedY)
        print "accuracy = ", accuracy
        accuracies.append([n, accuracy, yTest, predictedY])
    return accuracies
    #return [yTest, predictedY, accuracy]


init()
#populateDicts('./testData.txt')
#populateDicts('/Users/hariniravichandran/Documents/SML/data/DBLP_Citation_2014_May/domains/Artificial intelligence.txt')
populateDicts('/Users/agalya/Documents/sml/project/datasets/DBLP_Citation_2014_May/domains/Artificial intelligence.txt')
buildFeatureVector()
#features = [[1,2,3,4],[12,3,18,10],[3,2,1,5],[24,21,16,43],[1,1,1,1],[3,4,10,4]]
#predictions = [0,1,0,1,0,1]
#LRResults = runLR(featureVector, citationVector)
#KNNResults = runKNN(featureVector, citationVector)
#SVMResults = runSVM(featureVector, citationVector)
