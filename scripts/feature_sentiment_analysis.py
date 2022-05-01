import pandas as pd
from transformers import pipeline
import time
import sys, os

sys.path.append(os.getcwd())  # Workaround TODO: fix this
from core import preprocessing as prp
import seaborn as sns
import matplotlib.pyplot as plt


# TODO: clean up
if __name__ == "__main__":

    df = pd.read_csv("data/the-office_lines.csv", index_col=0)
    df = prp.preprocessing_pipeline(df=df, column="line")
    out_fp = "data/processed-sentiment-analysis.csv"
    if not os.path.isfile(out_fp):
        print("Running sentiment analysis ... this might take up to 1h")
        sentiment_pipeline = pipeline("sentiment-analysis")

        df_data = df.sample(50).reset_index()
        df_data = df.copy().reset_index()
        data = df_data["line"].values.tolist()
        start_time = time.time()
        print("Starting inference ...")
        res = sentiment_pipeline(data)
        print(f"Took {time.time()-start_time:.2f} seconds to predict sentiment for {len(data)} lines")
        res = pd.DataFrame(res)
        # out = pd.concat([pd.DataFrame({'line':data}), res], axis=1)
        df_data["label"] = res["label"]
        df_data["score"] = res["score"]
        df_data.set_index("index")

        df_data.to_csv(out_fp)

    df = pd.read_csv(out_fp, index_col=0)
    df["label_numerical"] = df["label"].apply(lambda x: {"NEGATIVE": -1, "POSITIVE": 1}[x])
    df["label_numerical_weighted"] = df["label_numerical"] * df["score"]

    characters = ["Michael", "Dwight", "Jim", "Pam"]
    df = df[df["character"].isin(characters)]

    df_pos = df[df["label_numerical"] > 0]
    df_neg = df[df["label_numerical"] < 0]
    assert len(df_pos) + len(df_neg) == len(df)

    dfg = (
        df.groupby(["season", "episode_number", "label"])
        .sum()[["label_numerical", "label_numerical_weighted"]]
        .sort_index()
        .reset_index()
    )

    dfg["episode_number_temp"] = dfg["episode_number"] - 1

    g = sns.FacetGrid(data=dfg, col="season", sharex=False, col_wrap=3, legend_out=True)
    g.map_dataframe(
        sns.barplot,
        y="label_numerical",
        x="episode_number",
        hue="label",
        palette="viridis",
        alpha=0.5,
        dodge=False,
        ci=None,
    )

    g.map_dataframe(sns.pointplot, y="label_numerical", x="episode_number_temp", ci=None)

    label_fs = 9
    title_fs = 12
    ticklabel_fs = 6
    for ax in g.axes:
        season = ax.get_title()[-1]
        ax.set_title(f"Season {season}", fontsize=title_fs)
        ax.set_xlabel("Episode")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=ticklabel_fs)
        ax.set_ylabel("# lines", fontsize=label_fs)

    g.fig.set_figwidth(12)
    g.fig.set_figheight(6)
    g.add_legend()
    g.fig.suptitle("Sentiment analysis per season and episode (dots show avg of neg and pos)")
    sns.move_legend(g, "upper right", bbox_to_anchor=(1, 1), title="Sentiment")
    plt.tight_layout()

    # Save figure
    fp = "docs/analysis_outputs/sentiment-analysis.png"
    if not os.path.isfile(fp):
        g.savefig(fp)

    # Close figure
    plt.close()
