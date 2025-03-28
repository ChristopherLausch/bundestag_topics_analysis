import pickle
from gensim import corpora
from gensim.models import LdaModel, LdaMulticore

def get_party_dfs(df):
    return {party: df[(df["party_clean"] == party) & (df["is_president"] == False)] for party in
            df["party_clean"].unique()}

def get_monthly_slices(df):
    return {month: data for month, data in df.groupby("date")}


def lda_per_party(df, num_of_topics=10):
    # Create a dictionary and corpus
    party_topics = {}

    # create a dictionary and corpus
    texts = [text.split() for text in df["processed_text"].dropna()]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train LDA model on the whole dataset
    try:
        lda_model = LdaMulticore(corpus, num_topics=num_of_topics, id2word=dictionary, passes=50, random_state=54321)
        # Store the model
        party_topics["All Parties"] = lda_model.print_topics()
    except:
        print("Problem with party: " + str("All Party Model"))
        return

    party_dfs = get_party_dfs(df)

    for party, party_df in party_dfs.items():
        # Tokenize the processed speech texts
        texts = [text.split() for text in party_df["processed_text"].dropna()]

        # Create a dictionary and corpus
        party_corpus = [dictionary.doc2bow(text) for text in texts]
        # Store results
        try:
            # Get topic distribution for this party
            topic_distributions = [lda_model[doc] for doc in party_corpus]

            # Average topic distribution across all speeches
            avg_topic_dist = {f"Topic_{i}": 0 for i in range(num_of_topics)}
            for doc in topic_distributions:
                for topic_id, prob in doc:
                    avg_topic_dist[f"Topic_{topic_id}"] += prob
            avg_topic_dist = {k: v / len(topic_distributions) for k, v in
                              avg_topic_dist.items()} if topic_distributions else avg_topic_dist
            party_topics[party] = avg_topic_dist

            print(avg_topic_dist)

        except:
            print("Problem with party: " + str(party))

    df_reduced = df[["processed_text", "party_clean"]]

    results = [party_topics, lda_model, dictionary, df_reduced]

    filename = "results/lda_per_party_results_" + str(num_of_topics)

    with open(filename, "wb") as fp:
        pickle.dump(results, fp)

    return lda_model

# ToDo also do one model globally
def monthly_lda_per_party(df, monthly_slices, model, num_of_topics=10):

    monthly_party_analysis = {}

    # create a dictionary and corpus
    texts = [text.split() for text in df["processed_text"].dropna()]
    dictionary = corpora.Dictionary(texts)

    lda_model = model

    # Loop through each month
    for date, monthly_df in monthly_slices.items():
        print(f"Processing {date}...")

        # Dictionary to store results for each party in this month
        party_topics = {}

        party_dfs = {party: monthly_df[(monthly_df["party_clean"] == party) & (monthly_df["is_president"] == False)] for party in monthly_df["party_clean"].unique()}

        for party, party_df in party_dfs.items():
            # Tokenize the processed speech texts
            texts = [text.split() for text in party_df["processed_text"].dropna()]

            # Create a dictionary and corpus
            party_corpus = [dictionary.doc2bow(text) for text in texts]
            # Store results
            try:
                # Get topic distribution for this party
                topic_distributions = [lda_model[doc] for doc in party_corpus]

                # Average topic distribution across all speeches
                avg_topic_dist = {f"Topic_{i}": 0 for i in range(num_of_topics)}
                for doc in topic_distributions:
                    for topic_id, prob in doc:
                        avg_topic_dist[f"Topic_{topic_id}"] += prob
                avg_topic_dist = {k: v / len(topic_distributions) for k, v in
                                  avg_topic_dist.items()} if topic_distributions else avg_topic_dist
                party_topics[party] = avg_topic_dist

                print(avg_topic_dist)

            except:
                print("Problem with party: " + str(party))

        texts = [text.split() for text in monthly_df["processed_text"].dropna()]

        # Create a dictionary and corpus
        party_corpus = [dictionary.doc2bow(text) for text in texts]
        # Store results
        try:
            # Get topic distribution for this party
            topic_distributions = [lda_model[doc] for doc in party_corpus]

            # Average topic distribution across all speeches
            avg_topic_dist = {f"Topic_{i}": 0 for i in range(num_of_topics)}
            for doc in topic_distributions:
                for topic_id, prob in doc:
                    avg_topic_dist[f"Topic_{topic_id}"] += prob
            avg_topic_dist = {k: v / len(topic_distributions) for k, v in
                              avg_topic_dist.items()} if topic_distributions else avg_topic_dist
            party_topics["All Parties"] = avg_topic_dist

            print(avg_topic_dist)

        except:
            print("Problem with party: " + str("All Parties"))

        # Store results for the month
        monthly_party_analysis[date] = party_topics

    df_reduced = df[["processed_text", "party_clean", "date", "year", "month", "day"]]

    results = [monthly_party_analysis, lda_model, dictionary, df_reduced]

    filename = "results/monthly_lda_per_party_results_" + str(num_of_topics)

    with open(filename, "wb") as fp:
        pickle.dump(results, fp)

    return monthly_party_analysis, lda_model