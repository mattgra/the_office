import pytest
import pandas as pd
from core import preprocessing as prp

df_test = pd.read_csv('tests/test_data/the-office_lines-test_sample.csv', index_col=0)


def test_basic_preprocessing():

    df = df_test.copy()
    df_out = prp.basic_preprocessing(df)

    assert all([x == x.lower() for x in df_out.columns])


def test_remove_specific_lines():

    df = df_test.copy()
    chars_to_test_against = ['”', '', ' ']

    for char in chars_to_test_against:
        new_row = pd.DataFrame({'Character': 'test', 'Line': char, 'Season': 99, 'Episode_Number': 99}, index=[len(df)])
        df = pd.concat([df, new_row])

    df_out = prp.remove_specific_lines(df, column='Line')

    for char in chars_to_test_against:
        assert not any(df_out['Line'] == char)


def test_remove_double_and_more_spaces():
    # TODO
    pass


def test_split_lines_into_sentences():
    # TODO
    pass


def test_preprocessing_pipeline():
    # TODO
    pass