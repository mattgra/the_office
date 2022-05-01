import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from core import preprocessing as prp
from core import nlp
import os


if __name__ == "__main__":

    # Load data
    df = pd.read_csv("data/the-office_lines.csv", index_col=0)

    # Preprocess data
    df = prp.preprocessing_pipeline(df=df)

    # Get tfidf
    df_tfidf = nlp.get_tfidf(df)

    # Select data subset with characters of interest
    characters = ["Michael", "Dwight", "Jim", "Pam"]
    df_sub = df_tfidf[characters]

    # Create a dataframe with the top N entries for selected characters
    number_of_lines_to_plot = 20  # Show top N lines per character
    top_n_df = pd.DataFrame()
    for i, char in enumerate(characters):
        top_n = df_sub[char].sort_values(ascending=False).head(number_of_lines_to_plot).reset_index()
        top_n["rank"] = top_n.index + 1
        top_n["character"] = char
        top_n = top_n.rename(columns={char: "tfidf"})
        top_n_df = pd.concat([top_n_df, top_n])

    # Plot data
    g = sns.catplot(
        x="tfidf",
        y="rank",
        col="character",
        hue="character",
        kind="bar",
        orient="h",
        dodge=False,
        palette="viridis",
        data=top_n_df,
    )
    for i, char in enumerate(characters):
        ax = g.axes_dict[char]
        for j in range(number_of_lines_to_plot):
            line = top_n_df[(top_n_df["character"] == char) & (top_n_df["rank"] == j + 1)]["line"].values[0]
            patch_id = i * number_of_lines_to_plot + j
            x_pos = 0
            y_pos = ax.patches[patch_id].get_y() + ax.patches[patch_id].get_height() / 2
            ax.text(x=x_pos, y=y_pos, s=f"   {line}", va="center", fontsize="8")

    # Format plot axes & titles
    g.set_titles("{col_name}")
    g.set_ylabels("")
    g.set_yticklabels([])
    g.set_xlabels("TF-IDF")
    g.fig.suptitle("Most typical phrases for key characters ordered by TF-IDF")
    plt.tight_layout()

    # Save figure
    fp = "docs/analysis_outputs/tf-idf-analysis.png"
    if not os.path.isfile(fp):
        g.savefig(fp)

    # Close figure
    plt.close()
