import pandas as pd
import re
import os

# loading files
buzzfeed_fake = pd.read_csv('data/raw/BuzzFeed_fake_news_content.csv')
buzzfeed_real = pd.read_csv('data/raw/BuzzFeed_real_news_content.csv')
politifact_fake = pd.read_csv('data/raw/PolitiFact_fake_news_content.csv')
politifact_real = pd.read_csv('data/raw/PolitiFact_real_news_content.csv')

# label mapper
buzzfeed_fake['label'] = 'emotional_manipulation'
buzzfeed_real['label'] = 'rational_argument'
politifact_fake['label'] = 'demagogy_tricks'
politifact_real['label'] = 'rational_argument'


# merging
news = pd.concat([buzzfeed_fake, buzzfeed_real,
                 politifact_fake, politifact_real], ignore_index=True)


news['text'] = news['title'].fillna('') + '. ' + news['text'].fillna('')
news = news[['text', 'label']]

# liar dataset
liar_raw = pd.read_csv(
    "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
    sep="\t", header=None,
    names=["filename", "label", "text", "subject", "speaker", "job", "state", "party",
           "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context"],
    on_bad_lines="skip"
)

liar_map = {
    'pants-fire':   'demagogy_tricks',
    'false':        'emotional_manipulation',
    'barely-true':  'emotional_manipulation',
    'half-true':    'emotional_manipulation',
    'mostly-true':  'rational_argument',
    'true':         'rational_argument',
}

liar = liar_raw[['text', 'label']].copy()
liar['label'] = liar['label'].map(liar_map)
liar = liar.dropna(subset=['label', 'text'])

# merging
df = pd.concat([news, liar], ignore_index=True)

# clean text


def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\w\s\.\!\?\,]', '', text)
    return text.strip()


df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.len() > 50]
df = df.drop_duplicates(subset='text')

if __name__ == '__main__':
    df[['text', 'label']].to_csv('data/processed/combined_en.csv', index=False)
    print(df['label'].value_counts())
    print(f'total: {len(df)}')
