import os
import sys
import codecs

try:
    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
except:
    pass


negativeTrainSetPath = os.getcwd() + "\IMDB_Data\\train\\neg"
vocabWeightFile = "unalteredWordWeightMap.txt"


vocabWeightDict = dict()
vocabWeightFile = open(vocabWeightFile, "r", encoding= "utf-8")
for lines in vocabWeightFile:
    word, spacer, weight = lines.split()
    vocabWeightDict[word] = weight


for files in os.listdir(negativeTrainSetPath):
    with open(os.path.join(negativeTrainSetPath, files), "r") as reviewFile:
        fileContents = reviewFile.read()
        print(fileContents)
        break
