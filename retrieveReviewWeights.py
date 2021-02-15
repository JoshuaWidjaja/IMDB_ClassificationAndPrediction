import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import defaultdict
import os
import sys
import codecs

#Needed to adjust encoding to avoid errors when using vocabWeightDict. 
try:
    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
except:
    pass

#Paths relative to Google Colab setup
trainDataLabeledBOWFile = os.getcwd() + "/IMDB_Data/train/labeledBow.feat"
trainDataUnlabeledBOWFILE = os.getcwd() + "/IMDB_Data/train/unsupBow.feat"
testDataLabeledBOWFile = os.getcwd() + "/IMDB_Data/test/labeledBow.feat"
vocabWeightFile = os.getcwd() + "/unalteredWordWeightMap.txt"
positiveTrainSetPath = os.getcwd() + "/IMDB_Data/train/pos"
negativeTrainSetPath = os.getcwd() + "/IMDB_Data/train/neg"


#Open and creates a dict from existing vocabWeightFile. This is used to create and organize data regarding the reviews. 
vocabWeightDict = dict()
vocabWeightFile = open(vocabWeightFile, "r", encoding= "utf-8")
for lines in vocabWeightFile:
    word, spacer, weight = lines.split()
    vocabWeightDict[word] = weight
vocabList = list(vocabWeightDict)
vocabWeightFile.close()

#### Function for creating an InfoList
def createInfoList(infoFile):
  fileNum = 0
  documentInfoList = []
  stopWords = set(stopwords.words("english"))
  BOWContents = open(infoFile, "r", encoding="UTF-8")

  #Below syntax may be confusing. I will explain it here.
  # vocabList is the List that contains a list of all words that occur in the reivews, in the same order as given in the files. For examples vocabList[0] is "the" 
  # tempList is the mapping from the BOW that associated Word Index : Word Occurrence. For example 0 : 9 means the word at index 0 of vocabList occurs 9 times in that review.
  # Therefore, when we do vocabWeightDict[vocabList[int(tempList[0])]] we are doing the following:
    # Assuming tempList is [0, 9]. Then vocabList[int(tempList[0])] is simply vocabList[0] which is the word "the".
    # Then vocabWeightDict[vocabList[int(tempList)]] is just doing vocabWeightDict["the"] to get the corresponding weight. We then multiply this by tempList[1] which is the occurence count.
  for lines in BOWContents:
    totalWeight = 0
    excludeStopWordsWeight = 0
    splitText = lines.split()
    for i in range(len(splitText)):
      if i == 0:
        pass
      else:
        tempList = splitText[i].split(":")
        totalWeight += float(vocabWeightDict[vocabList[int(tempList[0])]]) * float(tempList[1])
        if vocabList[int(tempList[0])] not in stopWords:
          excludeStopWordsWeight += float(vocabWeightDict[vocabList[int(tempList[0])]]) * float(tempList[1])
    documentInfoList.append( (fileNum, eval(splitText[0]), totalWeight, excludeStopWordsWeight))
    fileNum +=1
    #Increase to 25000 to see both positive and negative files.
    if fileNum == 25000:
      break
  BOWContents.close()
  ##documentInfoList contains the following information in the following order: 
  #(Number of File being used, Actual Rating of the File, Weight when counting all words, Weight when excluding stop words.)
  return documentInfoList

#Creating the documentInfoList
trainingDocumentInfoList = createInfoList(trainDataLabeledBOWFile)
testDocumentInfoList = createInfoList(testDataLabeledBOWFile)

#Uncomment to view what the document info list looks like.
print(len(trainingDocumentInfoList))
print(len(testDocumentInfoList))

####### Information and Statistic gathering on the data we are using is located below #######

def checkReviewFile(filePath: str, fileName: str) -> None:
    with open(os.path.join(filePath, fileName), "r") as reviewFile:
        fileContents = reviewFile.read()

#Variables to store results
weightGreaterWithStopWords = 0
weightLesserWithStopWords = 0
RatingWeightDictWithStopWords = defaultdict(list)

#for entry in documentInfoList:
  #Uncomment below to get the rating:weight dictionary with stop words included
  #RatingWeightDictWithStopWords[entry[1]].append(entry[2])
  #Uncomment the below code to get the number of reviews that end up with a greater/lesser weight when stopwords are included
  #if entry[2] > entry[3]:
  #  weightGreaterWithStopWords += 1
  #else:
  #  weightLesserWithStopWords += 1

#This prints the number of reviews that end up with a greater weight when stopwords are included, and the number that end up with a lesser weight when stopwords are included.
# print(weightGreaterWithStopWords, weightLesserWithStopWords)

#Prints the number of reviews with a specific rating and the average weight of the reviews with that rating
# for keys in RatingWeightDictWithStopWords.keys():
#   print("NUMBER OF REVIEWS WITH RATING " + keys + ": " + str(len(RatingWeightDictWithStopWords[keys])))
#   print("AVERAGE WEIGHT OF REVIEWS WITH RATING " + keys +": " + str(sum(RatingWeightDictWithStopWords[keys])/len(RatingWeightDictWithStopWords[keys])))


# documentInfoList index 0 is (0, 9, 15.92304840444889, 16.36440210426261): (file number, rating, total weight with stop words, weight without stop words) 

## Helper Function - Not used directly
#Opens files from specified file path (positive or negative) training data. Specify the directory in filePath, and the name of the file in fileName. Example function call below.
#checkReviewFile(positiveTrainSetPath, "0_9.txt") --- Will travel to the directory set in variable positiveTrainSetPath, and open file named 0_9.txt

