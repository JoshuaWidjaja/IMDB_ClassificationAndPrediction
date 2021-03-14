#In this block, we do our primary preprocessing and organization into the data we need to train our models. 

#Imports needed for proper preprocessing.
import nltk
import os
import sys
import codecs
from nltk.corpus import stopwords
from collections import defaultdict
nltk.download('stopwords')

#Needed to adjust encoding to avoid errors when using vocabWeightDict. 
try:
    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
except:
    pass

#Paths relative to local file setup
trainDataLabeledBOWFile = os.getcwd() + "/IMDB_Data/train/labeledBow.feat"
trainDataUnlabeledBOWFile = os.getcwd() + "/IMDB_Data/train/unsupBow.feat"
testDataLabeledBOWFile = os.getcwd() + "/IMDB_Data/test/labeledBow.feat"
vocabWeightFile = os.getcwd() + "/unalteredWordWeightMap.txt"
positiveTrainSetPath = os.getcwd() + "/IMDB_Data/train/pos"
negativeTrainSetPath = os.getcwd() + "/IMDB_Data/train/neg"


#Open and creates a dictionary from existing vocabWeightFile created earlier. We also create a list of all the words in order. This is used to create and organize data regarding the reviews.
#Output is the following: vocabWeightDict is a Dictionary with structure of WORD: WEIGHT
#vocabList is a List that contains a different word at each index, [WORD1, WORD2, ... ]
vocabWeightDict = dict()
vocabWeightFile = open(vocabWeightFile, "r", encoding= "utf-8")
for lines in vocabWeightFile:
    word, spacer, weight = lines.split()
    vocabWeightDict[word] = weight

vocabList = list(vocabWeightDict)
vocabWeightFile.close()

'''
Function for creating an InfoList
Below syntax may be confusing. I will explain it here.
vocabWeightDict is the dictionary that maps WORD : WEIGHT.
vocabList is the List that contains a list of all words that occur in the reviews, in the same order as given in the files. For examples vocabList[0] is "the" 
BOWList is the mapping from the BOW that associated Word Index : Word Occurrence. For example 0 : 9 means the word at index 0 of the vocabList (in this case "the") occurs 9 times in that review.

Therefore, when we do vocabWeightDict[vocabList[int(BOWList[0])]] we are doing the following:
Assuming BOWList is [0, 9]. Then vocabList[int(BOWList[0])] is simply vocabList[0] which is the word "the".
Then vocabWeightDict[vocabList[int(BOWList)]] is just doing vocabWeightDict["the"] to get the corresponding weight. We then multiply this by BOWList[1] which is the occurence count.
'''
def createInfoList(infoFile: str) -> list:
  #Here, we use NLTK's stopword set.
  stopWords = set(stopwords.words("english"))
  fileNum = 0
  documentInfoList = []
  BOWContents = open(infoFile, "r", encoding="UTF-8")
  for lines in BOWContents:
    totalWeight = 0
    excludeStopWordsWeight = 0
    fileLength = 0
    uniqueWords = set()
    splitText = lines.split()
    for i in range(len(splitText)):
      if i == 0:
        pass
      else:
        BOWList = splitText[i].split(":")
        totalWeight += float(vocabWeightDict[vocabList[int(BOWList[0])]]) * float(BOWList[1])
        fileLength += int(BOWList[1])
        uniqueWords.add(vocabList[int(BOWList[0])])
        if vocabList[int(BOWList[0])] not in stopWords:
          excludeStopWordsWeight += float(vocabWeightDict[vocabList[int(BOWList[0])]]) * float(BOWList[1])
    documentInfoList.append( (fileNum, eval(splitText[0]), totalWeight, excludeStopWordsWeight, fileLength, len(uniqueWords)))
    fileNum +=1
  BOWContents.close()
  return documentInfoList

#documentInfoList contains the following information in the following order: 
#(Number of File being used, Actual Rating of the File, Weight when counting all words, Weight when excluding stop words.)

'''
Function removes entries in the documentInfoList that surpass the bounds given within lowerBound and upperBound.
We utilized this to see how removing outliers would effect the accuracy of our models.
We do this so that we can remove any significant outliers from the data, and then train using this dataset.
'''
def optimizeInfoList(infoList: list, lowerBound: float, upperBound: float) -> list:
  optimizedInfoList = []
  for i in range(len(infoList)):
    if infoList[i][2] > lowerBound and infoList[i][2] < upperBound:
      optimizedInfoList.append(infoList[i])
  return optimizedInfoList

#Creating the three infoLists, one for each type of data.
trainingDocumentInfoList = createInfoList(trainDataLabeledBOWFile)
testDocumentInfoList = createInfoList(testDataLabeledBOWFile)
unlabeledReviewDocumentInfoList = createInfoList(trainDataUnlabeledBOWFile)

#Creating the optimized infoLists, only for labeled training and testing data.
optimizedTrainingList = optimizeInfoList(trainingDocumentInfoList, -30.0, 30.0)
optimizedTestingList = optimizeInfoList(testDocumentInfoList, -30.0, 30.0)

#Uncomment to view what the document info list looks like.
#print("Training InfoList: " + str(trainingDocumentInfoList))
#print("Testing InfoList: " + str(testDocumentInfoList))

#Print lengths of the infoLists to confirm completion of the script.
print(len(trainingDocumentInfoList), len(testDocumentInfoList))
print(len(optimizedTrainingList), len(optimizedTestingList))


####### Information and Statistic gathering on the data we are using is located below #######

#Helper Function - Not used directly
#Opens files from specified file path (positive or negative) training data. Specify the directory in filePath, and the name of the file in fileName. Example function call below.
#checkReviewFile(positiveTrainSetPath, "0_9.txt") --- Will travel to the directory set in variable positiveTrainSetPath, and open file named 0_9.txt
def checkReviewFile(filePath: str, fileName: str) -> None:
    with open(os.path.join(filePath, fileName), "r") as reviewFile:
        fileContents = reviewFile.read()

#Everything Below this line is used to check specifics of data. We will be using this for the report or for specific checkups.

#Variables to store results
weightGreaterWithStopWords = 0
weightLesserWithStopWords = 0
RatingWeightDictWithStopWords = defaultdict(list)


#Can change the infoList being iterated over to any of the three. Primarily used on (trainingDocumentInfoList) or (testDocumentInfoList)
#Uncomment the lines below if you wish to assign values to the variables above.

# for entry in trainingDocumentInfoList:
#  RatingWeightDictWithStopWords[entry[1]].append(entry[2])
#  if entry[2] > entry[3]:
#   weightGreaterWithStopWords += 1
#  else:
#   weightLesserWithStopWords += 1

#This prints the number of reviews that end up with a greater weight when stopwords are included, and the number that end up with a lesser weight when stopwords are included.
#print(weightGreaterWithStopWords, weightLesserWithStopWords)

#Prints the number of reviews with a specific rating and the average weight of the reviews with that rating
#for keys in RatingWeightDictWithStopWords.keys():
# print("NUMBER OF REVIEWS WITH RATING " + str(keys) + ": " + str(len(RatingWeightDictWithStopWords[keys])))
# print("AVERAGE WEIGHT OF REVIEWS WITH RATING " + str(keys) +": " + str(sum(RatingWeightDictWithStopWords[keys])/len(RatingWeightDictWithStopWords[keys])))




