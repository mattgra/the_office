import pandas as pd
from core.preprocessing import preprocessing_pipeline

import matplotlib.pyplot as plt

if __name__ == "__main__":

    df = pd.read_csv('data/the-office_lines.csv', index_col=0)
    df = preprocessing_pipeline(df=df, column='line', verbose=True)

    df_tf = df[['line', 'character']]
    df_tf['values'] = 1
    pivot = df_tf.pivot_table(columns='character', index='line', values='values', aggfunc='count').fillna(0)

    michael = pivot['Michael']
    others = pivot[pivot.columns.drop('Michael')].sum(axis=1)
    df_michael = pd.concat([michael, others], axis=1).rename(columns={0: 'Others'})

    # Normalize to usage per 10000 sentences
    df_michael['Michael_norm'] = df_michael['Michael'] / df_michael['Michael'].sum() * 10000
    df_michael['Others_norm'] = df_michael['Others'] / df_michael['Others'].sum() * 10000

    fig, ax = plt.subplots()
    df_michael.plot.scatter(x='Michael_norm', y='Others_norm', ax=ax)
    # ax[0].set_xlim([0,300])
    # ax[0].set_ylim([0,300])
    # ax[1].set_xlim([0, 300])
    # ax[1].set_ylim([0, 300])