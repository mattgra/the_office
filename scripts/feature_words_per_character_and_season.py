import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core import preprocessing as prp
import os

if __name__ == "__main__":

    # Load data
    df = pd.read_csv("data/the-office_lines.csv", index_col=0)

    # Preprocess data
    df = prp.preprocessing_pipeline(df=df)

    # Select data subset with characters of interest
    characters = ["Michael", "Dwight", "Jim", "Pam"]
    df_sub = df[df["character"].isin(characters)]

    # Count number of lines per season and character and store in dataframe
    dfg = df_sub.groupby(["season", "character"])["line"].count().reset_index()

    # Plot results
    plt.ion()
    fig, ax = plt.subplots()
    sns.barplot(
        x="line", y="season", hue="character", hue_order=characters, orient="h", palette="viridis", data=dfg, ax=ax
    )
    ax.set_xlabel("Count of spoken lines")
    ax.set_ylabel("Season")
    ax.set_title("Count of spoken lines per season and character")

    # Save figure
    fp = "docs/analysis_outputs/count_of_spoken_lines_per_season_and_character.png"
    if not os.path.isfile(fp):
        fig.savefig(fp)

    # Close figure
    plt.close()
