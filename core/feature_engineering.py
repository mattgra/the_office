import pandas as pd


def get_merged_dataframes(fp_episodes="data/the_office_series.csv", fp_lines="data/processed-sentiment-analysis.csv"):
    """Function to merge ratings reviews per episode with NLP analysis of text spoken during episodes.

    Function to merge 2 specific dataframes.
    (1) contains review data for each episode
    (2) contains each spoken line per character & episode incl. a sentiment classification of the spoken line (i.e., positive or negative sentiment)

    TODO: episode indexing is not correct, some episodes are missing! (check with wikipedia article)

    :param fp_episodes: absolute or relative (to project root) filepath of ratings dataframe
    :param fp_lines: absolute or relative (to project root) filepath of ratings dataframe
    :return: merged dataframe
    """

    # 1. I/O
    df_episodes = pd.read_csv(fp_episodes, index_col=0)
    df_lines = pd.read_csv(fp_lines, index_col=0)

    # 2. Create a matching index for both dataframes to merge them -> "episode_count"
    # 2A) For df_episodes: from 1 to 188
    df_episodes["episode_count"] = df_episodes.reset_index()["index"] + 1

    # 2B) For df_lines: from 1 to 188
    df_lines["episode_count"] = 0

    # Create new index for df_lines
    episodes_before_this_season = 0
    episode_count = []
    for season in df_lines["season"].unique():
        old_episode_numbers = df_lines[df_lines["season"] == season]["episode_number"]
        new_episode_numbers = old_episode_numbers + episodes_before_this_season
        episode_count.extend(new_episode_numbers)
        episodes_before_this_season += old_episode_numbers.nunique()
    df_lines["episode_count"] = episode_count

    # 3) Merge dfs
    df = pd.merge(left=df_episodes, right=df_lines, how="outer", on="episode_count", suffixes=("_ep", "_ln"))
    df = df.drop(columns="Season")

    return df


def extract_features(df):
    """Extracts specified features from a pandas dataframe."

    Extracts features around episodes - e.g., NLP processing of spoken text during shows to classify / predict the rating of an episode.
    NOTE: the unit of analysis is 1 episode - i.e., the df returned will have 1 line entry per episode.

    TODO
    :param df: pandas dataframe containing the-office data
    :return: pandas dataframe with specific features PER episode
    """

    dfg = df.groupby(["character", "label", "episode_count"]).agg(
        {
            "Ratings": "mean",
            "Votes": "mean",
            "Viewership": "mean",
            "Duration": "mean",
            "line": "count",
            "episode_count": "mean",
            "score": "mean",
        }
    )

    dff = pd.DataFrame({"episode": df['episode_count'].unique()})
    for char in ['Michael', 'Dwight', 'Jim', 'Pam']:
        dff["num_negative_lines_" + char] = dfg.loc[char, "NEGATIVE"]['line']
        dff["num_negative_lines_" + char] = dff["num_negative_lines_" + char].fillna(0)
        dff["num_positive_lines_" + char] = dfg.loc[char, "POSITIVE"]['line']
        dff["num_positive_lines_" + char] = dff["num_positive_lines_" + char].fillna(0)

    dff['rating'] = df.groupby('episode_count')['Ratings'].mean()
    return dff


if __name__ == "__main__":

    df = get_merged_dataframes()
    df = extract_features(df)

