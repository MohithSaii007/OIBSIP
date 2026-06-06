 Email Spam Detection using Machine Learning
Project Overview

This project is part of the Oasis Infobyte Internship.

It builds a Machine Learning model that can classify messages as:

 Ham (Not Spam)
 Spam (Junk / Phishing / Promotional)

The model learns from text data and predicts whether a given message is spam or not.

 Dataset Information

The dataset contains SMS/email messages with labels:

Column	Description
v1	Label (ham or spam)
v2	Message text

After preprocessing:

ham → 0
spam → 1

 Objective:-
To build a spam detection system that:
Understands text messages
Learns patterns of spam messages
Predicts whether a new message is spam or not
Technologies Used :-

Python 
Pandas
NumPy
Scikit-learn
TfidfVectorizer
Naive Bayes Classifier
 Machine Learning Model :-
 
Algorithm: Multinomial Naive Bayes
Feature Extraction: TF-IDF Vectorization
Problem Type: Text Classification
 Workflow :- 
 
Load dataset
Clean data
Convert labels (spam/ham → 1/0)
Convert text into numerical features using TF-IDF
Train-test split
Train Naive Bayes model
Evaluate accuracy
Predict new messages

Results :-
High accuracy (around 95% - 99%)
Efficient spam detection
Works well for real-world SMS/email filtering
