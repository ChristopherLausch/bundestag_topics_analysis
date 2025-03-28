import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
import re

# Load the German spaCy model
nlp = spacy.load("de_core_news_sm")

# Download German stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("german"))

# Regex pattern for introductions/greetings
INTRO_PATTERN = re.compile(
    r"^((?:vielen|herzlichen) dank(?:. – |, ))?"
    r"(((sehr (?:geehrte|verehrte)(?:r)? )?"
    r"(?:frau|herr|glück auf|moin,)?( )?(?:(bundestags|alters)?präsident(?:in)?)?)?"
    r"[,!\s]*"
    r"(((sehr (?:geehrte|verehrte)(?:r)? )"
    r"?(?:frau|herr)?(?: (bundes)kanzler|(bundes)minister|(bundes)regierung(?:in)?)?)?)?"
    r"[,!\s]*"
    r"(?:meine\s+)?(?:sehr\s+)?"
    r"(?:(verehrte|geschätzte|geehrte|liebe|werte)(?:n|r|s)?)?"
    r"\s*"
    r"(und)?"
    r"(?:damen\s+und\s+herren|(kolleginnen\s+und\s+)?kollegen)?"
    r"(abgeordnete)?"
    r"[,!\s]*"
    r"(?:(liebe|sehr\s+geehrte|geschätzte|werte)(?:n|r|s)?)?\s*(und)?"
    r"(?:kolleginnen\s+und\s+kollegen|damen\s+und\s+herren|publikum|präsidium|bürger(innen)|besucher(innen)\s+und\s+besucher)?"
    r"(auf den tribünen)?"
    r"(?:[,!\s]*(?:der\s+demokratischen\s+fraktionen)?)?)*",
    re.IGNORECASE | re.MULTILINE
)
PARENTHESIS_PATTERN = re.compile(r"\(.*?\)") # Matches everything inside ()
PUNCT_NUM_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)  # Matches punctuation
DIGIT_PATTERN = re.compile(r"\d+") # Matches digits


def preprocess(df):
    """ Preprocessing pipeline that handles all the necessary steps"""


    # Splitting of the date column into days/weeks/months/year
    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df = df.sort_values(by="date")

    df["processed_text"] = df["text"].str.lower().str.strip().str.replace(INTRO_PATTERN, "", regex=True)
    df["processed_text"] = df["processed_text"].str.replace(PARENTHESIS_PATTERN, "", regex=True)
    df["processed_text"] = df["processed_text"].str.replace(PUNCT_NUM_PATTERN, "", regex=True)
    df["processed_text"] = df["processed_text"].str.replace(DIGIT_PATTERN, "", regex=True)

    df["processed_text"] = list(nlp_preprocessing(df["processed_text"].tolist()))

    df["processed_text"] = df["processed_text"].astype(str).apply(remove_custom_stopwords)

    df["is_president"] = df["is_president"].astype(bool)  # Ensure it's Boolean (True/False)

    # Apply the cleaning function
    df["party_clean"] = df["speaker_party"].astype(str).apply(clean_party_name)

    df_only_parties = df[df['party_clean'].isin(['Die Linke', 'SPD', 'CDU/CSU', 'Bündnis 90/Die Grünen', 'FDP', 'AfD', 'fraktionslos', 'BSW'])]

    df_only_parties.to_csv("data/bundestag_wp20_speeches_preprocessed.csv")

    return df_only_parties

def nlp_preprocessing(text):
    """Processes a batch of texts using spaCy for faster execution."""
    processed_texts = []

    for doc in nlp.pipe(text, batch_size=50, n_process=-1):  # Enable parallel processing
        tokens = [token.lemma_ for token in doc if
                  token.text not in stop_words and not token.is_punct and not token.is_space]
        processed_texts.append(" ".join(tokens))

    return processed_texts


def remove_custom_stopwords(text):
    """Removes custom stopwords from a lemmatized text."""

    custom_stopwords = {"sagen", "gehen", "geben", "kommen", "müssen", "vieler", "brauchen", "sein", "mal", "ganz",
                        "sagen", "schon", "tun", "Jahr", "Land", "Kollege", "Herr", "heute", "mehr", "Beifall", "ja",
                        "werden", "haben", "Mensch", "Deutschland", "gut", "groß", "neu", "dank", "wichtig", "dafür",
                        "immer", "gerade", "möchten", "natürlich", "dafür", "daran", "insbesondere", "gerne", #here i added some more words -> check with julius
                        "halten", "paar", "seit", "einer", "--", "zwischenfrag", "Kollegin", "dame", "stehen",
                        "machen", "wirklich", "lassen", "deshalb", "deswegen", "Ministerin", "Dame", "letzter",
                        "richtig", "lieb", "nächster", "bei", "schaffen"
                        }

    tokens = text.split()  # Split text into words
    filtered_tokens = [word for word in tokens if word not in custom_stopwords]
    return " ".join(filtered_tokens)


def clean_party_name(party):
    if pd.isna(party):
        return "Unknown"
    party = party.strip()  # Remove leading/trailing spaces
    party = party.replace("\xa0", " ")  # Replace non-breaking spaces with normal spaces
    party = " ".join(party.split())  # Remove extra spaces and newlines
    party = party.replace("\n", " ")  # Replace newlines with spaces
    party = party.replace("DIE LINKE", "Die Linke")  # Standardize "Die Linke"
    party = party.replace("BÜNDNIS 90/DIE GRÜNEN", "Bündnis 90/Die Grünen")  # Standardize
    party = party.replace("Fraktionslos", "fraktionslos")
    party = party.replace("SPDCDU/CSU", "Unknown")  # Fix odd merged entry
    return party
