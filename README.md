Project Overview

This repository contains a MATLAB implementation of a Naive Bayes classifier built from scratch to perform sentiment analysis on movie reviews. The model classifies text-based movie reviews as either "positive" or "negative".

This project showcases a fundamental NLP and machine learning pipeline:

Data Preprocessing: Cleaning and tokenizing the raw text of movie reviews.

Model Training: Building a probabilistic model by calculating the likelihood of each word appearing in positive versus negative reviews. This involves creating word frequency maps and calculating prior probabilities for each class.

Inference: Using the trained model and Bayes' theorem to predict the sentiment of new, unseen reviews.

Evaluation: Calculating the accuracy of the classifier on a test set to measure its performance.

The script will load and process the dataset, train the Naive Bayes model from scratch, and print the final classification accuracy on the test set to the Command Window.
