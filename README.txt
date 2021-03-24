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
ipynb file that is meant to be run as a demonstration/example within Google Colab.

IMDB_Data Folder - Folder contains the IMDB dataset and all of its contents. This data is preprocessed and used 
in the primary ipynb file.

