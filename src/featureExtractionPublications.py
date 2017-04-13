import re

def init():
    global pubDict, mappings, authorsDict, venuesDict
    mappings = {'#*': 'title', '#c': 'venue', '#!': 'abstract', '#@': 'authors', '#%': 'references', '#index': 'pubId', '#t': 'year'}
    pubDict = {}
    authorsDict = {}
    venuesDict = {}

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
            tempDict = {key:None for key in tempDict}
            populateAuthorDict(pubDict[pubId])
            populateVenueDict(pubDict[pubId])
    getCitations()
    populateAuthorSecondaryFeatures()
    #print pubDict


def getCitations():
    global pubDict, authorsDict, venuesDict
    print "getting citations"
    for pubId in pubDict:
        references = pubDict[pubId]['references']
        for reference in references:
            if reference in pubDict:
                pubDict[reference]['citedBy'].append(pubId)
                pubDict[reference]['citationCount'] += 1
    for author in authorsDict:
        pubList = authorsDict[author]['pubList']
        for pubId in pubList:
            authorsDict[author]['citationCount'] += pubDict[pubId]['citationCount']
        coAuthors = authorsDict[author]['coAuthors']
        for coAuthor in coAuthors:
            pubList = coAuthors[coAuthor]['pubList']
            for pubId in pubList:
                coAuthors[coAuthor]['citeCount'] += pubDict[pubId]['citationCount']
    for venue in venuesDict:
        pubList = venuesDict[venue]['pubList']
        for pubId in pubList:
            venuesDict[venue]['citationCount'] += pubDict[pubId]['citationCount']


def populateAuthorSecondaryFeatures():
    print "getting secondary features for author" 
    for author, data in authorsDict.iteritems():                       
        for publication in data['pubList']:            
            publicationData = pubDict[publication]                
            selfCitation = [i for i, j in zip(data['pubList'], publicationData['references']) if i == j]
            data['selfCitationCount'] += len(selfCitation)
            for coAuthor in data['coAuthors']:
                coAuthorCitation = [i for i, j in zip(authorsDict[coAuthor]['pubList'], publicationData['references']) if i == j and i not in data['pubList']]                
                if len(coAuthorCitation) > 0:
                    print author, coAuthor, coAuthorCitation
                    data['coAuthorCitationCount'] += len(coAuthorCitation)                           


def buildInputFeatures():
    global pubDict
    inputFeatuers = []

init()

populateDicts('./testData.txt')
# populateDicts('/Users/hariniravichandran/Documents/SML/data/DBLP_Citation_2014_May/domains/Artificial intelligence.txt')
# populateDicts('/Users/agalya/Documents/sml/project/datasets/DBLP_Citation_2014_May/domains/Artificial intelligence.txt')
#populateDicts('DBLP_Citation_2014_May/domains/Artificial intelligence.txt')
