import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
import re

if __name__ == '__main__':

    from preprocess import preprocess
    from lda import lda_per_party, get_monthly_slices, monthly_lda_per_party

    df = pd.read_csv("data/bundestag_wp20_speeches.csv")

    # Check if an already preprocessed dataframe exists and if yes, use it
    try:
        df_preprocessed = pd.read_csv("data/bundestag_wp20_speeches_preprocessed.csv")
        print("loaded preprocessed data")
    except:
        df_preprocessed = preprocess(df)
        print("Preprocessing completed")

    monthly_slices = get_monthly_slices(df_preprocessed)

    topics = [10, 15, 20]

    for num_of_topics in topics:
        print("Full time: " + str(num_of_topics))
        model = lda_per_party(df_preprocessed, num_of_topics)
        monthly_lda_per_party(df_preprocessed, monthly_slices, model, num_of_topics)



