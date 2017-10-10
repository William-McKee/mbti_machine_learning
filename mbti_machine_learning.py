# -*- coding: utf-8 -*-
"""
Meyers Briggs Machine Learning Algorithm
William McKee
October 2017
"""

# Initial library declarations
#import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
#import seaborn as sns

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

print("Data description: ")
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
mbti_type_counts['percent_population'] = mbti_type_counts.index.map(try_it_2)
print(mbti_type_counts)