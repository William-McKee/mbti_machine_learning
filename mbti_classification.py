# -*- coding: utf-8 -*-
"""
Meyers Briggs Machine Learning Algorithm
Classification Functions
William McKee
October 2017
"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def get_best_parameters(classifier, param_dict, data_train, labels_train):
    '''
    Determine the best scoring parameters for classifier
    classifier = supervised learning classifier to be scored
    param_dict = dictionary containing keyword arguments and list of values to try
    data_train = training data
    labels_train = correct answers for the trained data points
       
    Returns the best estimator dictionary
    '''
    #data_train_scaled = preprocessing.scale(data_train)
    grid = GridSearchCV(estimator=classifier, param_grid=param_dict)
    grid.fit(data_train, labels_train)
    return (grid.best_estimator_)

def score_classifier(classifier, data_train, labels_train, data_test, labels_test):
    '''
    Train and test a classifier.  Show performance to the user.
    classifier = supervised learning classifier to be scored
    data_train = training data
    labels_train = correct answers for the trained data points
    data_test = testing data
    labels_test = correct answer for the tested data points
       
    No return type
    '''
    # Fit classifier
    classifier_fit = classifier.fit(data_train, labels_train)
    accuracy = classifier_fit.score(data_test, labels_test)
    print("Accuracy: ", accuracy)
    print()
    
    # Make predictions
    predictions = classifier_fit.predict(data_test)
    print("Classification Report:")
    print(classification_report(labels_test, predictions))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(labels_test, predictions))
    print()