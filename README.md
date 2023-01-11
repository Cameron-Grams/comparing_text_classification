# Comparison of Nine Classification Models 

This project is a comparison of 9 different techniques for text classification. The goal was to answer the question:

_Does this text require simplification in order to make the text more easily understood?_

The techniques involved both supervised and unsupervised models for classification.  The training and test texts are included in the `data` folder, or can be referenced from the Kaggle web page, [UMich SIADS 695 Fall21: Predicting text difficulty](https://www.kaggle.com/competitions/umich-siads-695-fall21-predicting-text-difficulty/data). 

The models were examined in four stages: 
1. **Stage One** used features derived from the texts, such as level of vocabulary or Flesch-Kincaid Readability rated difficulty.
2. **Stage Two** used the distribution of the vocabulary in the two classes. 
3. **Stage Three** explored supervised and unsupervised techniques in a reduced feature space derived from the features in Stage One.
4. **Stage Four** explored a vector space model using cosine similarity with ideal class vectors representing each class. 

A detailed discussion is available in the report. 
