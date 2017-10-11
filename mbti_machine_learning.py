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

# Store posts by MBTI type
# https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words
# TODO: How to handle http links?
mbti_types_list = []
mbti_posts_list = []
stops_set = set(stopwords.words("english")) 
for index,row in mbti_data.iterrows():
    this_type = row['type']
    these_posts = row['posts'].split('|||')
    for post in these_posts:
        # MBTI type
        mbti_types_list.append(this_type)
        
        # Cleanup post
        post_letters_only = re.sub("[^a-zA-Z]", " ", post)
        
        # Convert to lower case, split into individual words
        post_words = post_letters_only.lower().split()
        
        # Remove stop words
        meaningful_words = [w for w in post_words if not w in stops_set]
        
        # Final post for storage
        final_post = " ".join(meaningful_words)
        
        # MBTI post
        mbti_posts_list.append(final_post)
        
# ===== NATURAL LANGUAGE EXPERIMENTATION =====
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=30000)

fitted_transformed_data = vectorizer.fit_transform(mbti_posts_list)
print(fitted_transformed_data)

print("\n")
print('Shape of Sparse Matrix: ', fitted_transformed_data.shape)
print('Amount of Non-Zero occurences: ', fitted_transformed_data.nnz)
print('sparsity: %.2f%%' % (100.0 * fitted_transformed_data.nnz /
                            (fitted_transformed_data.shape[0] * fitted_transformed_data.shape[1])))
print("\n")

# ===== MACHINE LEARNING EXPERIMENTATION =====
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Split data into training and testing sets
data_train, data_test, labels_train, labels_test = \
train_test_split(mbti_posts_list, mbti_types_list, test_size=0.25)
print("Training / Testing / Total Data lengths: ")
print(len(data_train), len(data_test), len(data_train) + len(data_test))
print("\n")

# Need labeled and transformed data
lb = LabelEncoder()
data_train_labeled = lb.fit_transform(data_train)
data_test_labeled = lb.fit_transform(data_test)
data_train_labeled = data_train_labeled[:, None]
data_test_labeled = data_test_labeled[:, None]

tree_classifier = tree.DecisionTreeClassifier(min_samples_split=10, max_leaf_nodes=1024)
tree = tree_classifier.fit(data_train_labeled, labels_train)
accuracy = tree_classifier.score(data_test_labeled, labels_test)
print("Accuracy: ", accuracy)
print("\n")

predictions = tree.predict(data_test_labeled)
print (classification_report(labels_test, predictions))

# TODO: This warning message results in poor scores for algorithm
# UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.