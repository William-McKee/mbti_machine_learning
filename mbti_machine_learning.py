# -*- coding: utf-8 -*-
"""
Meyers Briggs Machine Learning Algorithm
Main Program
William McKee
October 2017
"""

# Initial library declarations
import pandas as pd
from nltk.corpus import stopwords
from mbti_cleaning import clean_up_post

# General population percentages for MBTI types
# http://www.myersbriggs.org/my-mbti-personality-type/my-mbti-results/how-frequent-is-my-type.htm?bhcp=1
mbti_types = { 'ENFJ':  [2.5,'NF'],
               'ENFP':  [8.1,'NF'],
               'ENTJ':  [1.8,'NT'],
               'ENTP':  [3.2,'NT'],
               'ESFJ': [12.3,'SJ'],
               'ESFP':  [8.5,'SP'],
               'ESTJ':  [8.7,'SJ'],
               'ESTP':  [4.3,'SP'],
               'INFJ':  [1.5,'NF'],
               'INFP':  [4.4,'NF'],
               'INTJ':  [2.1,'NT'],
               'INTP':  [3.3,'NT'],
               'ISFJ': [13.8,'SJ'],
               'ISFP':  [8.8,'SP'],
               'ISTJ': [11.6,'SJ'],
               'ISTP':  [5.4,'SP'] }

# Read the MBTI file
mbti_data = pd.read_csv('mbti_1.csv')

# Data set basic properties
print("Data dimensions: ")
print(mbti_data.shape)
print()

print("Column values:")
print(mbti_data.columns.values)
print()

print("Data description:")
print(mbti_data.describe)
print()

# Assure every row has a proper MBTI type
incorrect_types = []
for index, row in mbti_data.iterrows():    
    if (row['type'] not in mbti_types.keys()):
        incorrect_types.append(row['type'])
        
if (len(incorrect_types) > 0):
    print("Incorrect types: ")
    print(incorrect_types)
else:
    print("There are no incorrect Meyers-Briggs Types in any rows")
print()

TOTAL = len(mbti_data)
def get_sample_percentage(item):
    '''return the percentage of this item among all elements in MBTI data set'''
    return round((item / TOTAL) * 100.0, 1)

def get_population_percentage(item):
    '''return the percentage of this item among the population'''
    return mbti_types[item][0]

def get_temperament(item):
    '''return the temperament of this item'''
    return mbti_types[item][1]

# Group by type and list percentages
print("MBTI TYPES TABLE\n")
mbti_type_counts = mbti_data.groupby('type').count()
mbti_type_counts['percent_sample'] = mbti_type_counts.apply(get_sample_percentage)
mbti_type_counts['percent_population'] = mbti_type_counts.index.map(get_population_percentage)
mbti_type_counts['temperament'] = mbti_type_counts.index.map(get_temperament)
print(mbti_type_counts)
print()

# Group by temperament and list percentages
print("MBTI TEMPERAMENTS TABLE\n")
mbti_temperament_counts = mbti_type_counts.groupby('temperament').sum()
print(mbti_temperament_counts)
print()
   
# Store posts by MBTI temperment
mbti_types_list = []
mbti_posts_list = []
stops_set = set(stopwords.words("english"))
for index,row in mbti_data.iterrows():
    mbti_types_list.append(mbti_types[row['type']][1])
    mbti_posts_list.append(clean_up_post(row['posts'], stops_set, mbti_types))
    
# ===== NATURAL LANGUAGE PROCESSING =====
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Split data into training and testing sets
RANDOM_STATE = 32
TEST_SIZE = 0.2
data_train, data_test, labels_train, labels_test = \
    train_test_split(mbti_posts_list, mbti_types_list, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("Training / Testing / Total Data lengths: ")
print(len(data_train), len(data_test), len(data_train) + len(data_test))
print()

# Perform vectorization based on word frequency
vectorizer = TfidfVectorizer(analyzer='word', min_df=0.01, stop_words = 'english')
X_train_tfidf = vectorizer.fit_transform(data_train)
X_test_tfidf = vectorizer.transform(data_test)

print("Training Data Matrix Shape:")
print(X_train_tfidf.shape)
print()

print("Testing Data Matrix Shape:")
print(X_test_tfidf.shape)
print()

# ===== MACHINE LEARNING EXPERIMENTATION =====
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from mbti_classification import get_best_parameters, score_classifier
    
# Decision Tree
print("DECISION TREE CLASSIFIER:")
tree_parameters = {'min_samples_split': [2,5,10], 
                   'max_leaf_nodes': [10,25,50]}
tree_classifier = tree.DecisionTreeClassifier()
best_estimator = get_best_parameters(tree_classifier, tree_parameters, X_train_tfidf, labels_train)

tree_classifier = tree.DecisionTreeClassifier(min_samples_split=best_estimator.min_samples_split, 
                                              max_leaf_nodes=best_estimator.max_leaf_nodes, 
                                              random_state=RANDOM_STATE)
score_classifier(tree_classifier, X_train_tfidf, labels_train, X_test_tfidf, labels_test)

# Random Forest
print("RANDOM FOREST CLASSIFIER:")
forest_parameters = {'n_estimators': [10,20,30],
                     'min_samples_split': [2,5,10], 
                     'max_leaf_nodes': [25,50,75]}
forest_classifier = RandomForestClassifier()
best_estimator = get_best_parameters(forest_classifier, forest_parameters, X_train_tfidf, labels_train)

forest_classifier = RandomForestClassifier(n_estimators=best_estimator.n_estimators, 
                                           min_samples_split=best_estimator.min_samples_split, 
                                           max_leaf_nodes=best_estimator.max_leaf_nodes, 
                                           random_state=RANDOM_STATE)
score_classifier(forest_classifier, X_train_tfidf, labels_train, X_test_tfidf, labels_test)

# Naive Bayes
print("NAIVE BAYES CLASSIFIER:")
nb_parameters = {'alpha': [0.001, 0.01, 0.1]}
nb_classifier = MultinomialNB()
best_estimator = get_best_parameters(nb_classifier, nb_parameters, X_train_tfidf, labels_train)

nb_classifier = MultinomialNB(alpha=best_estimator.alpha, fit_prior=False)
score_classifier(nb_classifier, X_train_tfidf, labels_train, X_test_tfidf, labels_test)

# Stochastic Gradient Descent 
print("STOCHASTIC GRADIENT DESCENT CLASSIFIER:")
sgd_parameters = {'max_iter': [2,5,10]}
sgd_classifier = SGDClassifier()
best_estimator = get_best_parameters(sgd_classifier, sgd_parameters, X_train_tfidf, labels_train)

sgd_classifier = SGDClassifier(max_iter=best_estimator.max_iter, random_state=RANDOM_STATE)
score_classifier(sgd_classifier, X_train_tfidf, labels_train, X_test_tfidf, labels_test)

# Select K Best
print("SELECT K BEST CLASSIFIER:")
kbest_filter = SelectKBest(f_classif, k=2500)
kbest_classifier = SGDClassifier(max_iter=best_estimator.max_iter, random_state=RANDOM_STATE)
kbest_pipeline = make_pipeline(kbest_filter, kbest_classifier)
score_classifier(kbest_pipeline, X_train_tfidf, labels_train, X_test_tfidf, labels_test)

# Support Vector Machine
print("SUPPORT VECTOR MACHINE:")
svm_parameters = {'C': [0.1, 1.0, 10.0],
                  'loss': ['hinge', 'squared_hinge']}
svm_classifier = LinearSVC()
best_estimator = get_best_parameters(svm_classifier, svm_parameters, X_train_tfidf, labels_train)

svm_classifier = LinearSVC(C=best_estimator.C, 
                           loss=best_estimator.loss, 
                           random_state=RANDOM_STATE)
score_classifier(svm_classifier, X_train_tfidf, labels_train, X_test_tfidf, labels_test)