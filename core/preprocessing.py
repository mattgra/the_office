import pandas as pd
import logging


def basic_preprocessing(df):
    """
    TODO

    :param df: pandas dataframe
    :return: pandas dataframe with all column headers in lower case
    """
    df.columns = df.columns.str.lower()
    return df


def remove_specific_characters(df, column):
    # TODO: remove things like lines that only consist of ”
    return_df = df[(df[column] != '”')]
    return return_df


def remove_lines_that_only_consist_of_one_space(df, column):
    """
    TODO

    :param df: pandas dataframe containing 1 sentence (and other columns) per row in column 'column'
    :param column: string specifying column name of column containing sentences
    :return: cleaned dataframe without any empty rows in column 'column'
    """

    # Remove those lines that are only an empty string or a space
    return_df = df[(df[column] != '') & (df[column] != ' ')]

    # Quick check
    assert not any(return_df[column] == ''), 'Error: Some lines are only an empty string'
    assert not any(return_df[column] == ' '), 'Error: Some lines are only a white space'

    return return_df


def remove_double_and_more_spaces(df, column):
    """
    TODO

    :param df: pandas dataframe containing 1 sentence (and other columns) per row in column 'column'
    :param column: string specifying column name of column containing sentences
    :return: cleaned dataframe without any multi-spaces and rows with empty strings / only a space
    """

    # Replace all multi-spaces in all dataframe entries with a single space
    df[column] = df[column].str.replace(r'\s+', ' ', regex=True)

    # Quick check that there are no entries left
    assert not any(df[column].str.contains('  ')), 'Error: Some lines contain a double space'

    return df


def split_lines_into_sentences(df, column, split_characters=None):
    """

    :param df:
    :param column:
    :param split_characters:
    :return:
    """

    if split_characters is None:
        split_characters = ['.', '?', '!']
    for split_char in split_characters:
        df[column] = df[column].str.split(split_char)
        df = df.explode(column)

    df = df.reset_index().rename(columns={'index': 'line_id'})

    return df


def preprocessing_pipeline(df, column, verbose=False):
    """
    TODO

    :param df:
    :param column:
    :param verbose:
    :return:
    """

    df = basic_preprocessing(df=df)
    df = split_lines_into_sentences(df=df, column=column)
    df = remove_double_and_more_spaces(df=df, column=column)
    df = remove_lines_that_only_consist_of_one_space(df=df, column=column)

    # Also remove any leading or lagging spaces
    df[column] = df[column].str.strip()

    df = remove_specific_characters(df=df, column=column)

    # Quick check
    assert not any(df[column] == ''), 'Error: Some lines are only an empty string'
    assert not any(df[column] == ' '), 'Error: Some lines are only a white space'
    assert not any(df[column].str.contains('  ')), 'Error: Some lines contain a double space'

    return df


if __name__ == "__main__":

    df = pd.read_csv('data/the-office_lines.csv', index_col=0)
    df = preprocessing_pipeline(df=df, column='line', verbose=True)