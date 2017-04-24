import re

def init():
    global pubDict, mappings, authorsDict, venuesDict, featureVector, years
    mappings = {'#*': 'title', '#c': 'venue', '#!': 'abstract', '#@': 'authors', '#%': 'references', '#index': 'pubId', '#t': 'year'}
    pubDict = {}
    authorsDict = {}
    venuesDict = {}
    featureVector = []
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

def filterPapers(pubId):
    global pubDict
    if not (pubDict[pubId]['authors'] and pubDict[pubId]['venue'] and pubDict[pubId]['year']):
        del pubDict[pubId]
        return True
    return False

def populateDicts(path):
    global pubDict, mappings, authorsDict
    i = 0
    tempDict = {'title':None, 'venue':None, 'abstract':None, 'authors':[], 'references':[], 'pubId':None, 'year':None}
    for line in getLines(path):
        i += 1
        if i % 100000 == 0:
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
            if tempDict['year'] == '-1':
                continue
            if pubId not in pubDict:
                pubDict[pubId] = {}
                for k in tempDict:
                    pubDict[pubId][k] = tempDict[k]
                filtered = filterPapers(pubId)
                if filtered:
                    continue
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
    global pubDict, authorsDict, venuesDict, featureVector, years
    for pubId in pubDict:
        pubFeatures = []
        authorFeatures = []
        venueFeatures = []
        allFeatures = []

        # Venue Level Features
        venue = pubDict[pubId]['venue']
        years.sort()
        venuePubCounts = [0 for i in xrange(int(years[0]), int(years[-1])+1)]
        for year in venuesDict[venue]['pubYears']:
            venuePubCounts[int(year) - int(years[0])] = venuesDict[venue]['pubYears'][year]

        venueFeatures.append(venuesDict[venue]['pubCount'])
        venueFeatures.extend(venuePubCounts)
        venueFeatures.append(venuesDict[venue]['citationCount'])

        # Publication Level Features
        pubCiteCntByYear = [0 for i in xrange(int(years[0]), int(years[-1])+1)]
        for year in pubDict[pubId]['citationCountByYear']:
            pubCiteCntByYear[int(year) - int(years[0])] = pubDict[pubId]['citationCountByYear'][year]

        pubFeatures.append(len(pubDict[pubId]['authors']))
        pubFeatures.extend(pubCiteCntByYear)

        # Author Level Features
        authors = pubDict[pubId]['authors']
        authorCitationCounts = [[authorsDict[author]['citationCount'], author] for author in authors]
        authorCitationCounts.sort(reverse=True)
        for citationCount, author in authorCitationCounts[:2]:
            authorPubCounts = [0 for i in xrange(int(years[0]), int(years[-1])+1)]
            authorPubYears = []
            for year in authorsDict[author]['pubYears']:
                authorPubCounts[int(year) - int(years[0])] = authorsDict[author]['pubYears'][year]
                authorPubYears.append(int(year))

            authorPubYears.sort()
            authorPubYearsDiff = [j-i for i, j in zip(authorPubYears[:-1], authorPubYears[1:])]    
            minGap = min(authorPubYearsDiff) if authorPubYearsDiff else 0
            maxGap = max(authorPubYearsDiff) if authorPubYearsDiff else 0

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
# buildFeatureVector()
res = checkCitationDistribution()
