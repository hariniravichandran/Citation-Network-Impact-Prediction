import re

def init():
    global pubs, mappings
    mappings = {'#*': 'title', '#c': 'venue', '#!': 'abstract', '#@': 'authors', '#%': 'references', '#index': 'pubId', '#t': 'year'}
    pubs = {}

def getLines(path):
    with open(path, 'r') as f:
        for line in f:
            yield line

def populatePublicationDict(path):
    global pubs, mappings
    tempDict = {'title':None, 'venue':None, 'abstract':None, 'authors':None, 'references':None, 'pubId':None, 'year':None}
    for line in getLines(path):
        if line.startswith('#'):
            if line[:2] in mappings or line[:6] in mappings:
                currLineKey = line[:6] if line[:6] in mappings else line[:2]
                if mappings[currLineKey] in tempDict and not tempDict[mappings[currLineKey]]:
                    tempDict[mappings[currLineKey]] = line[6:].replace('\r','').replace('\n','') if line[:6] in mappings else line[2:].replace('\r','').replace('\n','')
        else:
            pubId = tempDict['pubId']
            if pubId not in pubs:
                pubs[pubId] = {}
                for k in tempDict:
                    pubs[pubId][k] = tempDict[k]
            tempDict = {key:None for key in tempDict}
    print pubs

init()
populatePublicationDict('/Users/agalya/Documents/sml/project/code/Citation-Network-Impact-Prediction/testData.txt')
