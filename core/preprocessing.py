import re
import pandas as pd

# To get rid of false positive for 'SettingWithCopyWarning' based on this post:
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None  # default='warn'


def basic_preprocessing(df):
    """
    TODO

    :param df: pandas dataframe
    :return: pandas dataframe with all column headers in lower case
    """
    df.columns = df.columns.str.lower()
    return df


def remove_specific_lines(df, column, lines=None):
    """
    TODO

    :param df:
    :param column:
    :param lines:
    :return:
    """

    if lines is None:
        lines = ["”", "", " ", "]"]
    for line in lines:
        df = df[(df[column] != line)]

    return df


def remove_text_between_brackets(df, column):
    """
    TODO

    :param df:
    :param column:
    :return:
    """

    # Remove text between any combination of brackets (e.g., {this would be removed])
    df[column] = df[column].str.replace(r"\[.*\]", "", regex=True)
    df[column] = df[column].str.replace(r"\[.*\)", "", regex=True)
    df[column] = df[column].str.replace(r"\[.*\}", "", regex=True)
    df[column] = df[column].str.replace(r"\(.*\]", "", regex=True)
    df[column] = df[column].str.replace(r"\(.*\)", "", regex=True)
    df[column] = df[column].str.replace(r"\(.*\}", "", regex=True)
    df[column] = df[column].str.replace(r"\{.*\]", "", regex=True)
    df[column] = df[column].str.replace(r"\{.*\)", "", regex=True)
    df[column] = df[column].str.replace(r"\{.*\}", "", regex=True)

    # TODO: make this nicer and expand cleaning (e.g., any non-alphanumeric character at beginning of line removed)
    # Remove closing brackets at start of line
    pattern = re.compile(r'^]')
    df[column] = df[column].str.replace(pattern, '')

    # Remove ”+closing bracket at start of line
    pattern = re.compile(r'^”]')
    df[column] = df[column].str.replace(pattern, '')

    # Remove starting ’
    pattern = re.compile(r'^’')
    df[column] = df[column].str.replace(pattern, '')

    # Final cleaning of white spaces
    df[column] = df[column].str.strip()
    return df


def remove_double_and_more_spaces(df, column):
    """
    TODO

    :param df: pandas dataframe containing 1 sentence (and other columns) per row in column 'column'
    :param column: string specifying column name of column containing sentences
    :return: cleaned dataframe without any multi-spaces and rows with empty strings / only a space
    """

    # Replace all multi-spaces in all dataframe entries with a single space
    df[column] = df[column].str.replace(r"\s+", " ", regex=True)

    # Quick check that there are no entries left
    assert not any(df[column].str.contains("  ")), "Error: Some lines contain a double space"

    return df


def split_lines_into_sentences(df, column, split_characters=None):
    """
    TODO

    :param df:
    :param column:
    :param split_characters:
    :return:
    """

    if split_characters is None:
        split_characters = [".", "?", "!"]
    for split_char in split_characters:
        df[column] = df[column].str.split(split_char)
        df = df.explode(column)

    df = df.reset_index().rename(columns={"index": "line_id"})

    return df


def preprocessing_pipeline(df, column='line'):
    """
    TODO

    :param df:
    :param column:
    :param verbose:
    :return:
    """

    df = basic_preprocessing(df=df)
    df = split_lines_into_sentences(df=df, column=column)
    df = remove_text_between_brackets(df=df, column=column)
    df = remove_double_and_more_spaces(df=df, column=column)

    # Also remove any leading or lagging spaces
    df[column] = df[column].str.strip()

    df = remove_specific_lines(df=df, column=column)

    # Quick check
    assert not any(df[column] == ""), "Error: Some lines are only an empty string"
    assert not any(df[column] == " "), "Error: Some lines are only a white space"
    assert not any(df[column].str.contains("  ")), "Error: Some lines contain a double space"

    return df


if __name__ == "__main__":

    df = pd.read_csv("data/the-office_lines.csv", index_col=0)
    df = preprocessing_pipeline(df=df, column="line")
