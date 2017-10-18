# -*- coding: utf-8 -*-
"""
Meyers Briggs Machine Learning Algorithm
William McKee
October 2017
"""

# Initial library declarations
import pandas as pd
import string
from nltk.corpus import stopwords

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
print("\n")

# Group by temperament and list percentages
print("MBTI TEMPERAMENTS TABLE\n")
mbti_temperament_counts = mbti_type_counts.groupby('temperament').sum()
print(mbti_temperament_counts)
print("\n")

def clean_up_post(post, stops_set):
    '''Change post contents so that they contain only significant words'''
    # Split up post
    these_posts = post.split("|||")
    final_posts = []
    
    # Loop through posts
    for this_post in these_posts:
        
        # Split into words
        words = this_post.split()
        
        # Loop through words
        final_words = ""
        for word in words:

            # Remove punctuation and tildes
            exclude = set(string.punctuation)
            this_word = ''.join(ch for ch in word if ch not in exclude)
    
            # Convert to lower case
            this_word = this_word.lower()
            
            # Exclude links, numbers, stop words, and MBTI types
            if (word.startswith('http') or word.isdigit() or word in stops_set or word.upper() in mbti_types.keys()):
                continue
            
            # Final post for storage
            final_words = final_words + " " + this_word
        
        # Join words for post
        #final_words = final_words.strip()
        final_posts.append(final_words)
    
    # Join posts
    result_post = ""
    for item in final_posts:
        result_post = result_post + item
    
    return (result_post)
    
# Store posts by MBTI temperment
mbti_types_list = []
mbti_posts_list = []
stops_set = set(stopwords.words("english"))
for index,row in mbti_data.iterrows():
    mbti_types_list.append(mbti_types[row['type']][1])
    mbti_posts_list.append(clean_up_post(row['posts'], stops_set))
    
# ===== NATURAL LANGUAGE PROCESSING =====
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Split data into training and testing sets
RANDOM_STATE = 32
data_train, data_test, labels_train, labels_test = \
    train_test_split(mbti_posts_list, mbti_types_list, test_size=0.2, random_state=RANDOM_STATE)
print("Training / Testing / Total Data lengths: ")
print(len(data_train), len(data_test), len(data_train) + len(data_test))
print("\n")

vectorizer = TfidfVectorizer(analyzer='word', min_df=0.01, stop_words = 'english')
X_train_tfidf = vectorizer.fit_transform(data_train)
X_test_tfidf = vectorizer.transform(data_test)

print("Training Data Matrix Shape:")
print(X_train_tfidf.shape)
print("\n")

print("Testing Data Matrix Shape:")
print(X_test_tfidf.shape)
print("\n")

# ===== MACHINE LEARNING EXPERIMENTATION =====
# TODO: Implement function which cycles through parameters and return highest accuracy (or some other measure)

# Decision Tree
from sklearn import tree
from sklearn.metrics import classification_report
print("DECISION TREE CLASSIFIER:")
tree_classifier = tree.DecisionTreeClassifier(min_samples_split=10, max_leaf_nodes=50, random_state=RANDOM_STATE)
tree_classifier = tree_classifier.fit(X_train_tfidf, labels_train)
tree_accuracy = tree_classifier.score(X_test_tfidf, labels_test)
print("Accuracy: ", tree_accuracy)
print("\n")

tree_predictions = tree_classifier.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(labels_test, tree_predictions))
print("\n")

# Random Forest
from sklearn.ensemble import RandomForestClassifier
print("RANDOM FOREST CLASSIFIER:")
forest_classifier = RandomForestClassifier(n_estimators=20, min_samples_split=5, max_leaf_nodes=75, random_state=RANDOM_STATE)
forest_classifier = forest_classifier.fit(X_train_tfidf, labels_train)
forest_accuracy = forest_classifier.score(X_test_tfidf, labels_test)
print("Accuracy: ", forest_accuracy)
print("\n")

forest_predictions = forest_classifier.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(labels_test, forest_predictions))
print("\n")

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
print("NAIVE BAYES CLASSIFIER:")
nb_classifier = MultinomialNB(alpha=0.01, fit_prior=False)
nb_classifier = nb_classifier.fit(X_train_tfidf, labels_train)
nb_accuracy = nb_classifier.score(X_test_tfidf, labels_test)
print("Accuracy: ", nb_accuracy)
print("\n")

nb_predictions = nb_classifier.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(labels_test, nb_predictions))
print("\n")

# Stochastic Gradient Descent 
from sklearn.linear_model import SGDClassifier
print("STOCHASTIC GRADIENT DESCENT CLASSIFIER:")
SDG_MAX_ITER = 10
sgd_classifier = SGDClassifier(max_iter=SDG_MAX_ITER, random_state=RANDOM_STATE)
sgd_classifier = sgd_classifier.fit(X_train_tfidf, labels_train)
sgd_accuracy = sgd_classifier.score(X_test_tfidf, labels_test)
print("Accuracy: ", sgd_accuracy)
print("\n")

sgd_predictions = sgd_classifier.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(labels_test, sgd_predictions))
print("\n")

# Select K Best
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
print("SELECT K BEST CLASSIFIER:")
kbest_filter = SelectKBest(f_classif, k=2500)
kbest_classifier = SGDClassifier(max_iter=SDG_MAX_ITER, random_state=RANDOM_STATE)
kbest_pipeline = make_pipeline(kbest_filter, kbest_classifier)
kbest_pipeline = kbest_pipeline.fit(X_train_tfidf, labels_train)
kbest_accuracy = kbest_pipeline.score(X_test_tfidf, labels_test)
print("Accuracy: ", kbest_accuracy)
print("\n")

kbest_predictions = kbest_pipeline.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(labels_test, kbest_predictions))
print("\n")