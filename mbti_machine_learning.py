# -*- coding: utf-8 -*-
"""
Meyers Briggs Machine Learning Algorithm
William McKee
October 2017
"""

# Initial library declarations
import re
import pandas as pd
from nltk.corpus import stopwords

# General population percentages for MBTI types
# http://www.myersbriggs.org/my-mbti-personality-type/my-mbti-results/how-frequent-is-my-type.htm?bhcp=1
mbti_types = { 'ENFJ':  2.5,
               'ENFP':  8.1,
               'ENTJ':  1.8,
               'ENTP':  3.2,
               'ESFJ': 12.3,
               'ESFP':  8.5,
               'ESTJ':  8.7,
               'ESTP':  4.3,
               'INFJ':  1.5,
               'INFP':  4.4,
               'INTJ':  2.1,
               'INTP':  3.3,
               'ISFJ': 13.8,
               'ISFP':  8.8,
               'ISTJ': 11.6,
               'ISTP':  5.4 }

mbti_temperments = { 'ENFJ': 'NF',
                     'ENFP': 'NF',
                     'ENTJ': 'NT',
                     'ENTP': 'NT',
                     'ESFJ': 'SJ',
                     'ESFP': 'SP',
                     'ESTJ': 'SJ',
                     'ESTP': 'SP',
                     'INFJ': 'NF',
                     'INFP': 'NF',
                     'INTJ': 'NT',
                     'INTP': 'NT',
                     'ISFJ': 'SJ',
                     'ISFP': 'SP',
                     'ISTJ': 'SJ',
                     'ISTP': 'SP' }

# Read the MBTI file
mbti_data = pd.read_csv('mbti_1.csv')

# Data set basic properties
print("Data dimensions: ")
print(mbti_data.shape)
print("\n")

print("Column values:")
print(mbti_data.columns.values)
print("\n")

print("Data description:")
print(mbti_data.describe)
print("\n")

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
print("\n")

TOTAL = len(mbti_data)
def get_sample_percentage(item):
    '''return the percentage of this item among all elements in MBTI data set'''
    return round((item / TOTAL) * 100.0, 1)

def get_population_percentage(item):
    '''return the percentage of this item among the population'''
    return mbti_types[item]

# Group by type and list percentages
mbti_type_counts = mbti_data.groupby('type').count()
mbti_type_counts['percent_sample'] = mbti_type_counts.apply(get_sample_percentage)
mbti_type_counts['percent_population'] = mbti_type_counts.index.map(get_population_percentage)
print(mbti_type_counts)
print("\n")

def clean_up_post(post, stops_set):
    '''Change post contents so that they contain only significant words'''
    # Cleanup post
    post_words = re.sub('[^a-zA-Z]', '', post)
    
    # Remove splitter since posts are for same MBTI type
    post_words = re.sub('|||', '', post)
        
    # Remove links        
    post_words = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', post_words, flags=re.MULTILINE) 
        
    # Convert to lower case, split into individual words
    post_words = post_words.lower().split()
        
    # Remove stop words
    meaningful_words = [w for w in post_words if not w in stops_set]
        
    # Final post for storage
    final_post = ''.join(meaningful_words)
    
    return (final_post)
    
# Store posts by MBTI temperment
mbti_types_list = []
mbti_posts_list = []
stops_set = set(stopwords.words("english"))
for index,row in mbti_data.iterrows():
    mbti_types_list.append(mbti_temperments[row['type']])
    mbti_posts_list.append(clean_up_post(row['posts'], stops_set))
        
# ===== NATURAL LANGUAGE EXPERIMENTATION =====
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer(analyzer='word', min_df=0.1, max_df=0.5)
fitted_transformed_data = vectorizer.fit_transform(mbti_posts_list)
print("Fitted Transformed Data:")
print(fitted_transformed_data)
print("\n")

tfidf_transformer = TfidfTransformer()
tfidf_transformed_data =  tfidf_transformer.fit_transform(fitted_transformed_data).toarray()
print("TfIdf Transformed Data:")
print(tfidf_transformed_data)
print("\n")

# ===== MACHINE LEARNING EXPERIMENTATION =====
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Split data into training and testing sets
data_train, data_test, labels_train, labels_test = \
train_test_split(mbti_posts_list, mbti_types_list, test_size=0.2, random_state=32)
print("Training / Testing / Total Data lengths: ")
print(len(data_train), len(data_test), len(data_train) + len(data_test))
print("\n")

# Need labeled and transformed data
lb = LabelEncoder()
data_train_labeled = lb.fit_transform(data_train)
data_test_labeled = lb.fit_transform(data_test)
data_train_labeled = data_train_labeled[:, None]
data_test_labeled = data_test_labeled[:, None]

# Model parameters
MIN_SAMPLES_SPLIT_VALUE = 10
MAX_LEAF_NODE_VALUE = 150

# Decision Tree
print("DECISION TREE CLASSIFIER:")
tree_classifier = tree.DecisionTreeClassifier(min_samples_split=MIN_SAMPLES_SPLIT_VALUE, max_leaf_nodes=MAX_LEAF_NODE_VALUE)
tree_classifier = tree_classifier.fit(data_train_labeled, labels_train)
tree_accuracy = tree_classifier.score(data_test_labeled, labels_test)
print("Accuracy: ", tree_accuracy)
print("\n")

tree_predictions = tree_classifier.predict(data_test_labeled)
print("Classification Report:")
print(classification_report(labels_test, tree_predictions))
print("\n")

# Random Forest
print("RANDOM FOREST CLASSIFIER:")
forest = RandomForestClassifier(n_estimators=10, min_samples_split=MIN_SAMPLES_SPLIT_VALUE, max_leaf_nodes=MAX_LEAF_NODE_VALUE)
forest = forest.fit(data_train_labeled, labels_train)
forest_accuracy = forest.score(data_test_labeled, labels_test)
print("Accuracy: ", forest_accuracy)
print("\n")

forest_predictions = forest.predict(data_test_labeled)
print("Classification Report:")
print(classification_report(labels_test, forest_predictions))