import re
import pandas as pd

# To get rid of false positive for 'SettingWithCopyWarning' based on this post:
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None  # default='warn'


def basic_preprocessing(df):
    """Function for basic, (ideally) data-unspecific pre-processing (e.g., converting all column names to lower case)

    :param df: pandas dataframe
    :return: pandas dataframe with all column headers in lower case
    """

    df.columns = df.columns.str.lower()
    return df


def remove_specific_lines(df, column, lines=None):
    """Function to remove lines that only consist of irrelevant characters (e.g., empty lines, only a white space, ...) -> these lines are mostly artifacts from other pre-processing.

    :param df: pandas dataframe
    :param column: column name to select column from dataframe
    :param lines: (optional) list of strings representing the lines to be ignored (e.g., [" "])
    :return: pandas dataframe
    """

    if lines is None:
        lines = ["”", "", " ", "]"]
    for line in lines:
        df = df[(df[column] != line)]

    return df


def remove_text_between_brackets(df, column):
    """
    TODO: this function is such a mess and needs to be coded properly...

    :param df: pandas dataframe
    :param column: column name to select column from dataframe
    :return: pandas dataframe
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
    pattern = re.compile(r"^]")  # TODO: is this not matching anything that starts with "]"?
    df[column] = df[column].str.replace(pattern, "")

    # Remove ”+closing bracket at start of line
    pattern = re.compile(r"^”]")
    df[column] = df[column].str.replace(pattern, "")

    # Remove starting ’
    pattern = re.compile(r"^’")
    df[column] = df[column].str.replace(pattern, "")

    # Final cleaning of leading and trailing white spaces
    df[column] = df[column].str.strip()
    return df


def remove_double_and_more_spaces(df, column):
    """Function to replace occurences of multiple white spaces after each other with a single white space."

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
    """Function to split strings based on specified split characters (e.g., input line = "Hi. How are you?" -> Output lines: "Hi", "How are you")
    Note: This each split will add a new row to the input dataframe.

    :param df: pandas dataframe containing spoken text per character
    :param column: string specifying column name of column containing spoken text
    :param split_characters: (optional) list of characters to use for string splitting (e.g., sentence terminators)
    :return: dataframe with 1 row per sentence
    """

    if split_characters is None:
        split_characters = [".", "?", "!"]
    for split_char in split_characters:
        df[column] = df[column].str.split(split_char)
        df = df.explode(column)

    df = df.reset_index().rename(columns={"index": "line_id"})

    return df


def preprocessing_pipeline(df, column="line"):
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
