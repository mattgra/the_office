import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.getcwd())  # Workaround TODO: fix this
from core.preprocessing import preprocessing_pipeline

# To get rid of false positive for 'SettingWithCopyWarning' based on this post:
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None  # default='warn'


def get_counts_of_sentences_per_character(df, char_col='character', line_col='line'):
    """
    TODO

    :param df:
    :param char_col:
    :param line_col:
    :return:
    """

    # 1) Extract the character & line columns of the dataframe
    df_tf = df[[line_col, char_col]]

    # 2) Add a third column with "1" in every row (this will be used to sum up the pivot result below)
    df_tf['values'] = 1

    # 3) Pivot by character & line and aggregate - fill the rest with 0's
    # TODO: i'm not entirely sure why we use 'count' and not 'sum' in the aggfunc?
    pivot = df_tf.pivot_table(columns=char_col, index=line_col, values='values', aggfunc='count').fillna(0)

    return pivot


def get_counts_per_line_for_specific_characters_vs_others(df, character, char_col='character', line_col='line'):
    """
    TODO

    :param df:
    :param character:
    :param char_col:
    :param line_col:
    :return:
    """
    line_counts_per_character = get_counts_of_sentences_per_character(df, char_col=char_col, line_col=line_col)

    # 1a) Extract the column for the specified character (this is a pd.series)
    character_count = line_counts_per_character[character]

    # 1b) For 'others' -> sum up all the remaining characters (excl. the character of interest) (this is a pd.series)
    others_count = line_counts_per_character[line_counts_per_character.columns.drop(character)].sum(axis=1)

    # 2) Combine the 2 series to a dataframe matched on the same indices (they should have the same anyway!)
    # assert character_count.index == others_count.index, 'Indices of pandas series do not match!'
    df_character = pd.concat([character_count, others_count], axis=1).rename(columns={0: 'Others'})

    # 3) Normalize the line counts to 'counts per 10'000 lines'
    # Otherwise - those who speak more lines naturally have higher counts and a comparison is difficult
    df_character[character + '_norm'] = df_character[character] / df_character[character].sum() * 10000
    df_character['Others_norm'] = df_character['Others'] / df_character['Others'].sum() * 10000

    return df_character


def get_tfidf(df, char_col='character', line_col='line'):
    """
    TODO
    :param df:
    :param character:
    :param char_col:
    :param line_col:
    :return:
    """

    counts_per_char = get_counts_of_sentences_per_character(df, char_col=char_col, line_col=line_col)

    # 1) Term frequency normalized by total number of lines for specific character (N
    # tf = counts_per_char[character] / counts_per_char[character].sum()
    tf = counts_per_char / counts_per_char.sum()

    # 2) Inverse document frequency of term t = N_characters / N_characters that contain term t
    idf = np.log(len(counts_per_char.columns) / counts_per_char.sum(axis=1))

    # 3) Now we want to multiply the tf matrix (tf per term and per character) with the idf vector (idf per term)
    # Pandas is pretty bad at multiplying a matrix with a vector, so we move to numpy for this
    # For this we use a small trick in numpy - we convert the idf vector into a matrix (by replicating the vector)
    # [1,    [[1, 1, 1],
    #  2, =>  [2, 2, 2],
    #  3]     [3, 3, 3]]
    # ... and then use an element-wise multiplication on the tf_matrix and the idf_matrix
    # But before - we want to make sure the indices of both product inputs are the same
    assert all(tf.index == idf.index), 'Indices do not match'
    tf_matrix = tf.values
    idf_matrix = np.repeat(idf.values[:, None], len(tf.columns), axis=1)
    tfidf_matrix = np.multiply(tf_matrix, idf_matrix)

    # Convert back into dataframe
    df_tfidf = pd.DataFrame(tfidf_matrix, columns = tf.columns, index=tf.index)

    # Test by comparing a single column
    tfidf_michael = tf['Michael'] * idf
    assert all(tfidf_michael == df_tfidf['Michael']), 'TFIDF test not passed.'

    return df_tfidf


if __name__ == "__main__":

    df = pd.read_csv('data/the-office_lines.csv', index_col=0)
    df = preprocessing_pipeline(df=df, column='line', verbose=True)
    df_michael = get_counts_per_line_for_specific_characters_vs_others(df, 'Michael')

    tfidf = get_tfidf(df)