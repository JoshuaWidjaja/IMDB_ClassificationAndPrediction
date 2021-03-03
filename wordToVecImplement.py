import os
import sys
import codecs
from gensim.utils import simple_preprocess

positiveTrainSetPath = os.getcwd() + "/IMDB_Data/train/pos"
negativeTrainSetPath = os.getcwd() + "/IMDB_Data/train/neg"
try:
    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
except:
    pass


wordMap = dict()
for filename in os.listdir(positiveTrainSetPath):
    with open(os.path.join(positiveTrainSetPath, filename), 'r', encoding="utf-8") as file:
        contents = file.read()
        preprocessedList = simple_preprocess(contents, True)
        filenameSplit = filename.split('_')
        wordMap[int(filenameSplit[0])] = preprocessedList
        
        file.close()

for filename in os.listdir(negativeTrainSetPath):
    with open(os.path.join(negativeTrainSetPath, filename), 'r', encoding="utf-8") as file:
        contents = file.read()
        preprocessedList = simple_preprocess(contents, True)
        filenameSplit = filename.split('_')
        wordMap[int(filenameSplit[0]) + 12500] = preprocessedList
        
        file.close()

count = 0
outputFile = open("reviewNumberWordMap.txt", "a", encoding="utf-8")
for key, value in sorted(wordMap.items()):
    outputFile.write(str(key) + ":" + str(value) + "\n")


outputFile.close()

