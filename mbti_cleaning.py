# -*- coding: utf-8 -*-
"""
Meyers Briggs Machine Learning Algorithm
Cleaning Functions
William McKee
October 2017
"""

import string

def clean_up_post(post, stops_set, mbti_types):
    '''
    Change post contents so that they contain only significant words
    post = posts from one line of csv file
    stops_set = set of stop words which do not add any value to NLP prediction
    mbti_types = dictionary with mbti types (such as ENFP, ISTJ) as keys
    
    Returns cleaned up post
    '''
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
        final_posts.append(final_words)
    
    # Join posts
    result_post = ""
    for item in final_posts:
        result_post = result_post + item
    
    return (result_post)