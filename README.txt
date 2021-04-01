### INTRO ###

The focus of this project is two things:
1. To classify sentiment of movie reviews based on the contents of the review. Classification is categorized
as either Positive, Negative, or Neutral.

2. To train a model to predict the score of a review given the contents, and compare that accuracy to the 
actual review score.

To do this, we are using the IMDB Large Movie Review Dataset obtainable at https://ai.stanford.edu/~amaas/data/sentiment/
Proper Citation as requested:
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}

This project was created for the purpose of the CS 175: Project in AI class for UCI. It was created by the 
students Siddharth Bhansali, John Nguyen, and Joshua Widjaja under the supervision of Professor Padhraic Smyth.

### CONTENTS ###

Colab Files Folder - Contains both the demonstration and actual ipynb files as well as an HTML version of both.

Demonstration Files Folder - Folder contains pretrained models and data. These are used in a shorter simplified
ipynb file that is meant to be run as a demonstration/example within Google Colab. The contents are described in further detail below

	#------------Joblib Files------------#
	classification_KNN.joblib - Exported model of KNN classification only classifying positive and negative.

	classification_KNN_Neutral.joblib - Exported model of KNN classifcation classifying positive, negative, and neutral

	classification_LogisticClassifier.joblib - Exported model of Logistic classification classifying positive and negative

	classification_wordToVecLogistic.joblib - Exported model of the the WordToVec classification classifying only positive and negative

	#------------NPZ Files------------#
	demonstrationData.npz - Exported data used for non Word2Vec models.
	
	wordToVecDemonstration.npz - Exported data used for Word2Vec models

	#------------PT (Pytorch) Files------------#
	score_prediction.pt - Exported model used for score prediction


IMDB_Data Folder - Folder contains the IMDB dataset and all of its contents. This data is preprocessed and used 
in the primary ipynb file.

Other Files - Other files currently not located in any folder are gone into detail below.

#------------Python Files------------#

retrieveReviewWeights.py - This file uses the created text files (listed here) to create the documentInfoList
that we primarily use for training our models and also contains some functions that helped us gather information
regarding the data. The proper code is included in the Google Colab file.

simpleLearningModel.py - We used this to get some understanding of SkLearn and Pytorch for creating models.
This code is not included in the Google Colab and was only used for learning purposes, not for actual training/testing.

weightParser.py - This script uses our given data from our dataset, combines it, and outputs it to a file in a format very easy to read
and utilize for preprocessing. The form is of WORD : WEIGHT, and this code is included in the Google Colab file.

wordToVecImplement.py - This script parses through every labeled training review, and outputs contents to a file with the format of
REVIEWNUM : [WORDS] where the words are in order of occurrence. We use this to preprocess and train our Word2Vec Model imported
from Gensim. This code is included in the Google Colab file.

#------------Text Files------------#
greatestWordWeightMap.txt - Output of weightParser.py, orders words based on weight in descending order.

lowestWordWeightMap.txt - Output of weightParser.py, orders words based on weight in ascending order.

reviewNumberWordMap.txt - Output of wordToVecImplement.py, each line is one review where the REVIEWNUM : [CONTAINED WORDS]

unalteredWordWeightMap.txt - Output of weightParser.py, has the default ordering of the words based on weight.

