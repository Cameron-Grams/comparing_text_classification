import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import textstat
from collections import defaultdict
import matplotlib.pyplot as plt


# produce main dataframe
def produce_dataframe(path):
    """
    arguments: string path to the collection of sentences

    return: dataframe with the features:
        - original_text
        - tokens
        - lables if formating the training data
    """

    df = pd.read_csv(path)
    df['caps_tokens'] = df.apply(lambda x: word_tokenize(x['original_text']), axis=1)
    df['tokens'] = df.apply(lambda x: [t.lower() for t in x['caps_tokens']], axis=1)
    df['sentence_length'] = df.apply(lambda x: len(x['tokens']), axis=1)
    df = df.drop('caps_tokens', axis=1)
    return df



# the helper function to count the membership of sentence words in the various lists
def dale_chall_check_each_token(token_list, master_list):
    """
    arguments: given the list of tokes from the sentence and the master list containing
    all of the words in the group (Dale-Chall, Concreteness, AoA) 
 
    return: an integer for the sentence: the integer is the number of words from the 
    sentence in the group list

    O(t * m) time...embarrassing
    """

    total_included = 0
 
    for t in token_list:
        if t in master_list:
            total_included += 1

    return  total_included




# format the words in the Dale-Chall file
def create_dale_chall_list(path_):
    """
    given the path to the file returns a list of the words contained
    in the Dale-Chall readability file
    """

    with open(path_, 'r') as file:
        dale_chall_raw = file.readlines()

    return [w.split('\n')[0].strip() for w in dale_chall_raw]

#----------------------------------------------------------------------

def derive_concrete_score(main_df, conc_df):
    """
    concreteness is figured based on concrete words within the total sentence: averaged
    among the words without concrete ratings 

    arguments: main dataframe with all of the sentences, concreteness dataframe with 
        values for the words

    return: float: an average of the mean concreteness ratings and an average
        of the percent known are figured accounting for total sentence length
    """
    
    # index object to store values that have been seen
    concrete_percent = {}
    
    # set of all words with concreteness values
    set_concrete = set(conc_df['Word'])

    def obtain_tuple(word, concrete_percent, conc_df):
        # produces the (concrete mean value, percent recognized) tuple for each word

        try:
            return concrete_percent[word], concrete_percent
        except:
            row = conc_df[conc_df['Word'] == word]
            tup = (list(row['Conc.M'])[0], list(row['Percent_known'])[0])
            concrete_percent[word]= tup
            return tup, concrete_percent



    main_df['set_tuples'] = main_df.apply(lambda x: [obtain_tuple(w, concrete_percent, conc_df)[0] if w in set_concrete else (0, 0) for w in x['tokens'] ], axis=1) 
    main_df['concreteness'] = main_df.apply(lambda x: sum([t[0] for t in x['set_tuples']])/x['sentence_length'], axis=1)
    main_df['recognized'] = main_df.apply(lambda x: sum([t[1] for t in x['set_tuples']])/x['sentence_length'], axis=1)

    main_df = main_df.drop('set_tuples', axis=1)

    return main_df

#----------------------------------------------------

def derive_age_of_acquisition(main_df, AoA_df):
    """
    Adds the features of the estimated aga a word is acquired according to the studey by Kupperman ...
    and the second feature the percent of study participants who recognized the words

    arguments: the main dataframe of words, the Age of Acquistiion dataframe from the Kupperman study

    return: a dataframe with the two additional features 
    """

    aoa_percent = {}

    aoa_df = AoA_df[['Word', 'AoA_Kup', 'Perc_known']]
    aoa_df = aoa_df.dropna()

    # all words left in the collection 
    aoa_words = set(aoa_df['Word'])

    def obtain_tuple(word, aoa_percent, aoa_df):
        # produces the (age of acquistion value, percent recognized) tuple for each word

        try:
            return aoa_percent[word], aoa_percent
        except:
            row = aoa_df[aoa_df['Word'] == word]
            tup = (list(row['AoA_Kup'])[0], list(row['Perc_known'])[0])
            aoa_percent[word]= tup
            return tup, aoa_percent

    main_df['set_tuples'] = main_df.apply(lambda x: [obtain_tuple(w, aoa_percent, aoa_df)[0] if w in aoa_words else (0, 0) for w in x['tokens'] ], axis=1)
    main_df['aoa'] = main_df.apply(lambda x: sum([t[0] for t in x['set_tuples']])/x['sentence_length'], axis=1)
    main_df['perc_known'] = main_df.apply(lambda x: sum([t[1] for t in x['set_tuples']])/x['sentence_length'], axis=1)

    main_df = main_df.drop('set_tuples', axis=1)

    return main_df

#--------------------------------------------

def produce_flesch_kincaid(main_df):
    """
    adds a features for the Flesch-Kincaid readability and grade measures

    arguments: the main dataframe to receive the features 

    return: the dataframe with the 2 additional features added
    """

    main_df['fk_ease'] = main_df.apply(lambda x: textstat.flesch_reading_ease(x['original_text']), axis=1)
    main_df['fk_grade'] = main_df.apply(lambda x: textstat.flesch_kincaid_grade(x['original_text']), axis=1)

    return main_df

#---------------------------------------------

def plot_distribution(df_, title_string):

    df = df_.copy()
    
    # specify character distribution
    df['character_length'] = df.apply(lambda x: len(x['original_text']), axis=1)
    d_ = df.groupby('character_length').count()
    d_ = d_.reset_index()
    d_ = d_.rename({'sentence_length': 'number_sentences'}, axis=1)
    d_ = d_[['character_length', 'number_sentences']]
    X_characters = d_['character_length']
    y_characters = d_['number_sentences']


    # specify token distribution
    df_ = df.groupby('sentence_length').count()
    df_ = df_.reset_index()
    df_ = df_.rename({'tokens': 'number sentences'}, axis=1)
    df_ = df_[['sentence_length', 'number sentences']]
    X_sentences = df_['sentence_length']
    y_sentences = df_['number sentences']

    # produce the plot
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11, 5))
    fig.suptitle(title_string)

    ax1.plot(X_characters, y_characters)
    ax1.set_xlabel('Sentence character length')
    ax1.set_ylabel('Number of sentences')

    ax2.plot(X_sentences, y_sentences)
    ax2.set_xlabel('Sentence token length')

    plt.show()
