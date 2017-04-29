import re
import numpy
import pandas
import csv

# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split

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

#RF
from sklearn.ensemble import RandomForestRegressor

#GBR
from sklearn.ensemble import GradientBoostingRegressor

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
            for relevantYear, pubCiteCount in pubDict[pubId]['citationCountByYear'].items():
                if relevantYear not in authorsDict[author]['citationCountByYear']:
                    authorsDict[author]['citationCountByYear'][relevantYear] = 0
                authorsDict[author]['citationCountByYear'][relevantYear] += pubCiteCount



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
            for relevantYear, pubCiteCount in pubDict[pubId]['citationCountByYear'].items():
                if relevantYear not in venuesDict[venue]['citationCountByYear']:
                    venuesDict[venue]['citationCountByYear'][relevantYear] = 0
                venuesDict[venue]['citationCountByYear'][relevantYear] += pubCiteCount 

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
            # print pubId, " ", citationCountUntilYear
            # print publication['citationCountByYear']
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
            # authorFeatures.append(getHindexForYear(author, publishedYear))
            authorFeatures.append(authorsDict[author]['hIndex'])
            authorFeatures.append(authorPubCount)
            authorFeatures.append(citationCount)

            # authorFeatures.append(minGap)
            # authorFeatures.append(maxGap)
            # authorFeatures.append(min(authorsDict[author]['pubYears'].values()))
            # authorFeatures.append(max(authorsDict[author]['pubYears'].values()))
        if len(authorCitationCounts) < 2:
            authorFeatures = authorFeatures + [0] * ((2-len(authorCitationCounts)) * 3)

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

def runNN(xTrain, xTest, yTrain, yTest, features, epochs=5):
    model = buildNN(features)
    #estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
    model.fit(xTrain, yTrain, epochs=epochs, batch_size=32)
    predictedY = model.predict(xTest, batch_size=128)
    predictedY = [round(y) for y in predictedY]
    return predictedY

def runKNN(xTrain, xTest, yTrain, yTest):
    yValues = []
    print "KNN Model"
    for n in range(1, 21):
        print "Training with ", n, " neighbours",
        model = KNeighborsRegressor(n_neighbors=n, algorithm='auto')
        model.fit(np.array(xTrain), np.array(yTrain))
        predictedY = model.predict(xTest)
        predictedY = [round(y) for y in predictedY]
        yValues.append(predictedY)
    return yValues

def runSVR(xTrain, xTest, yTrain, yTest, c=1, kernel='rbf', test_size=0.25):
    model = svm.SVR(kernel=kernel, cache_size=500, C = c)
    model.fit(np.array(xTrain), np.array(yTrain))
    predictedY = model.predict(xTest)
    predictedY = [round(y) for y in predictedY]
    return predictedY

def runSVRCV(xTrain, xTest, yTrain, yTest, c=1, kfold=5):
    model = svm.SVR(kernel='rbf', cache_size=500, C = c)
    scores = cross_val_score(model, features, predictions, cv=kfold, scoring='mean_squared_error')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores

def runLR(xTrain, xTest, yTrain, yTest):
    accuracies = []
    regr = linear_model.LinearRegression()
    regr.fit(np.array(xTrain), np.array(yTrain))
    predictedY = regr.predict(xTest)
    predictedY = [round(y) for y in predictedY]
    return predictedY

def runRidge(xTrain, xTest, yTrain, yTest):
    regr = linear_model.Ridge(alpha=0.1)
    regr.fit(np.array(xTrain), np.array(yTrain))
    predictedY = regr.predict(xTest)
    predictedY = [round(y) for y in predictedY]
    return predictedY

def predict0(xTrain, xTest, yTrain, yTest):
    print "Predict 0"
    predictedY = [0] * len(yTest)
    return predictedY

def predictSumOfThree(xTrain, xTest, yTrain, yTest):
    predictedY = [sum(record[0:4]) for record in xTest]
    return predictedY

def runRF(xTrain, xTest, yTrain, yTest, max_depth=None):
    model = RandomForestRegressor(max_depth=max_depth, random_state=2)
    model.fit(xTrain, yTrain)
    predictedY = model.predict(xTest)
    return predictedY

def runGBR(xTrain, xTest, yTrain, yTest, max_depth=None):
    model = GradientBoostingRegressor(max_depth=max_depth, random_state=2)
    model.fit(xTrain, yTrain)
    predictedY = model.predict(xTest)
    return predictedY

def getMSE(yTrue, yPred):
    mse = np.mean((np.array(yPred) - np.array(yTrue)) ** 2)
    return mse

def getR2(yTrue, yPred):
    r2 = r2_score(yTrue, yPred, multioutput='variance_weighted')
    return r2

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

def getCitationVsPubDistribution():
    global pubDict
    hist = {}
    for pubId in pubDict:
        if pubDict[pubId]['citationCount'] not in hist:
            hist[pubDict[pubId]['citationCount']] = 0
        hist[pubDict[pubId]['citationCount']] += 1
    return hist

def checkCitationDistributionbyYear():
    global pubDict
    hist = {}
    for pubId in pubDict:
        pubYears = pubDict[pubId]['citationCountByYear'].keys()
        for each in pubYears:
            if each not in hist:
                hist[each] = pubDict[pubId]['citationCountByYear'][each]
            else:
                hist[each] += pubDict[pubId]['citationCountByYear'][each]
    return hist


def writeFeatureVector():
    global featureVector, citationVector
    with open('./FeatureVectorPublications_ai.csv', "wb") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(featureVector)

    with open('./CitationVectorPublications_ai.csv', "wb") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(citationVector)    

# To import features from CSV
# featureVector1 = pandas.read_csv('./FeatureVectorPublications.csv', sep='\t', header=None)
# featureVector = featureVector1.values.tolist()
# featureVector1 = pandas.read_csv('./ScaledFeatureVectorPublications.csv', sep='\t', header=None)
# scaledFeatureVector = featureVector1.values.tolist()
# featureVector1 = pandas.read_csv('./CitationVectorPublications.csv', sep='\t', header=None)
# citationVector = featureVector1.values.tolist()
# citationVector = citationVector[0]

init()
populateDicts('../data/domains/Artificial intelligence.txt')
buildFeatureVector()
writeFeatureVector()
xTrain, xTest, yTrain, yTest = train_test_split(featureVector, citationVector, test_size=0.25)
NNPrediction = runNN(xTrain, xTest, yTrain, yTest, featureVector, 10)
KNNPrediction = runKNN(xTrain, xTest, yTrain, yTest)
SVRPrediction = runSVR(xTrain, xTest, yTrain, yTest, 0.5)
LRPrediction = runLR(xTrain, xTest, yTrain, yTest)
RidgePrediction = runRidge(xTrain, xTest, yTrain, yTest)
ZeroPrediction = predict0(xTrain, xTest, yTrain, yTest)
SumOfThree = predictSumOfThree(xTrain, xTest, yTrain, yTest)
SVRCVPrediction = runSVRCV(xTrain, xTest, yTrain, yTest, 6.5, 1)
RFPrediction = runRF(xTrain, xTest, yTrain, yTest)
GBRPrediction = runGBR(xTrain, xTest, yTrain, yTest)

mse = getMSE(yTest, NNPrediction)
mse = getMSE(yTest, SVRPrediction)
mse = getMSE(yTest, LRPrediction)
mse = getMSE(yTest, RidgePrediction)
mse = getMSE(yTest, ZeroPrediction)
mse = getMSE(yTest, SumOfThree)
mse = getMSE(yTest, SVRCVPrediction)
mse = getMSE(yTest, RFPrediction)
mse = getMSE(yTest, GBRPrediction)
for yPred in KNNPrediction:
    print getMSE(yTest, yPred)

R2 = getR2(yTest, NNPrediction)
R2 = getR2(yTest, SVRPrediction)
R2 = getR2(yTest, LRPrediction)
R2 = getR2(yTest, RidgePrediction)
R2 = getR2(yTest, ZeroPrediction)
R2 = getR2(yTest, SumOfThree)
R2 = getR2(yTest, SVRCVPrediction)
R2 = getR2(yTest, RFPrediction)
R2 = getR2(yTest, GBRPrediction)
for yPred in KNNPrediction:
    print getR2(yTest, yPred)

# SVR For Difference C values
c = 0
inc = 0.5
cMap = []
while c < 31:
    c += inc
    print "c = ", c
    cMap.append(runSVR(xTrain, xTest, yTrain, yTest, c))

for yPred in cMap:
    print getMSE(yTest, yPred)
print "----"
for yPred in cMap:
    print getR2(yTest, yPred)


res = checkCitationDistributionbyYear()
