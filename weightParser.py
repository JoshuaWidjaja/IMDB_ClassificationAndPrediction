import os

basePath = os.getcwd() + "\IMDB_Data"
vocabFile = "\imdb.vocab"
weightFile = "\imdbEr.txt"

vocabList = []
counter = 0

vocabFile = open( basePath+ vocabFile, "r", encoding= "utf-8")
try:
    for lines in vocabFile:
        if counter % 1000 == 0:
            print("ON WORD COUNT: " + str(counter))
        vocabList.append(lines.strip("\n"))
        counter += 1
except UnicodeDecodeError:
    print("ERROR AT: " + str(counter) + str(lines))

vocabFile.close()

#Below code retrieves the weight of each word from the weightFile.
weightFile = open(basePath + weightFile, "r", encoding = "utf-8")
unalteredWordWeightMap = open("unalteredWordWeightMap.txt", "w", encoding="utf-8")
weightCounter = 0

#This dictionary maps each word with its perceived weight value
vocabWeight = dict()
testCounter = 0
for weight in weightFile:
  vocabWeight[vocabList[weightCounter]] = (float(weight.strip("\n")))
  unalteredWordWeightMap.write(vocabList[weightCounter] + " : " + str(vocabWeight[vocabList[weightCounter]]) + "\n")
  weightCounter += 1


weightFile.close()
unalteredWordWeightMap.close()

#Two dictionaries that have the vocabWeight sorted from greatest to lowest, and lowest to greatest respectively.
greatestWeightDict = sorted(vocabWeight.items(), key = lambda x: x[1], reverse=True)
lowestWeightDict = sorted(vocabWeight.items(), key = lambda x: x[1])

greatestWordWeightMap = open("greatestWordWeightMap.txt", "w", encoding= "utf-8")

for word in greatestWeightDict:
    greatestWordWeightMap.write(word[0] + " : " + str(word[1]) + "\n")
greatestWordWeightMap.close()

lowestWordWeightMap = open("lowestWordWeightMap.txt", "w", encoding="utf-8")
for word in lowestWeightDict:
    lowestWordWeightMap.write(word[0] + " : " + str(word[1]) + "\n")
lowestWordWeightMap.close()


#print(vocabList)
#print(greatestWeightDict)
#print(lowestWeightDict)
