import os
import torch.nn as nn
import torch as torch
from retrieveReviewWeights import createInfoList
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
#Using this resource as a very rough template: https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948

#Creating Tensors of size 25000 using information from documentInfoList
# both with and without stop words
trainDataBOWFile = os.getcwd() + "/IMDB_Data/train/labeledBow.feat"
documentInfoList = createInfoList(trainDataBOWFile)
print(len(documentInfoList))

tensor_weights_documentInfoList_both = torch.FloatTensor(documentInfoList) 

#Separates the "big" tensor three smaller ones: Ratings, weightWithStopWords, weightWithoutStopWords. 
#Cast into Numpy Array at first for what I did below, may need to change depending on what you need to do.
ratingFeature = np.asarray([tensor_weights_documentInfoList_both[i][1] for i in range(len(tensor_weights_documentInfoList_both))])
withStopWordsFeature = np.asarray([tensor_weights_documentInfoList_both[i][2] for i in range(len(tensor_weights_documentInfoList_both))])
removeStopWordsFeature =np.asarray([tensor_weights_documentInfoList_both[i][3] for i in range(len(tensor_weights_documentInfoList_both))])

#Right now , using 80% of training data as train and the other 20% to validate.
dataSplitFrac = 0.8
#batchSize affects how DataLoaders organize results. Example: Batchsize of 50 means will report results 50 at a time. Keep to 1 as default here.
#dataSize is just the size of the numpy array.
batchSize = 1
dataSize = len(withStopWordsFeature)

#Split into training and validation data for both WeightWithStopWords and Ratings.
trainX = withStopWordsFeature[0:int(dataSize * dataSplitFrac)]
trainY = ratingFeature[0:int(dataSize*dataSplitFrac)]

validX = withStopWordsFeature[int(dataSize*dataSplitFrac):]
validY = ratingFeature[int(dataSize * dataSplitFrac):]
# "Bundles" both WeightWithStopWords and Ratings into tensors 
trainData = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
validData = TensorDataset(torch.from_numpy(validX), torch.from_numpy(validY))

# Uses the trainData, which are now tensors, and shuffles them. Stores them in the DataLoader class.
trainLoader = DataLoader(trainData, shuffle=True, batch_size=batchSize)
validLoader = DataLoader(validData, shuffle=True, batch_size=batchSize)

#Results dictionary
trainLoaderResults = dict()

#Since batch size is 1, trainLoader will report one pair at a time.
#The first occurrence of a review with a certain rating has its weight multipled by .01. All further occurrences 
# of that rating has their weight multipled by .0001 and is added to the dictionary. These multiplication values are arbitrary.
for pair in trainLoader:
  if int(pair[1]) not in trainLoaderResults:
    trainLoaderResults[int(pair[1])] = pair[0] * .01
  else:
    trainLoaderResults[int(pair[1])] += (pair[0] * .0001)

#Uncomment to view what the result dictoinary looks like.
#print(sorted(trainLoaderResults.items()))

#Initializing an array of 5000 zeroes. 
correctPrediction = np.zeros(len(validLoader))

counter = 0
#For reviews in the validation dataset. If the rating of a review is > 5 and the weight is greater than the associated weight in trainLoaderResults, for this experiment
#we say the "prediction" is correct, and update the correctPrediction array by setting the index to 1.
# If the rating of a review is <5 and the weight is less than the associated weight, we also say we are correct and set the index to 1.
# Else, we set the index to 0.
for pair in validLoader:
  if int(pair[1]) > 5 and pair[0] > trainLoaderResults[int(pair[1])]:
    correctPrediction[counter] = 1
    counter += 1
  elif int(pair[1]) < 5 and pair[0] < trainLoaderResults[int(pair[1])]:
    correctPrediction[counter] = 1
    counter += 1
  else:
    #print(pair)
    correctPrediction[counter] = 0
    counter += 1

#Print our results of attemping to "predict" the validation set.
print("Percentage of Predictions Correct: " + str(sum(correctPrediction)/len(correctPrediction)* 100))
print("Amount of reviews predicted correct: " + str(sum(correctPrediction)))
print("Amount of reviews predicted incorrect: " + str(len(correctPrediction) - sum(correctPrediction)))
