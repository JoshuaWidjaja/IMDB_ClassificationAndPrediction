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

print(vocabList[-1])
#print(vocabList)
#print("LENGTH OF LIST IS: " + str(len(vocabList)))