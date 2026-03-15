import pandas as pd
import re
import os


def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\w\s\.\!\?\,]', '', text)
    return text.strip()


def weak_label(text):
    text_lower = text.lower()

    fear_words = [
        'threat', 'danger', 'destroy', 'attack', 'enemy',
        'crisis', 'catastrophe', 'disaster', 'terror', 'kill',
        'invasion', 'collapse', 'chaos', 'panic', 'warned', 'warning',
        'end of', 'survival', 'extinction', 'annihilate', 'obliterate',
        'wipe out', 'coming for', 'they want', 'take over', 'takeover',
        'your children', 'your family', 'our children', 'next generation',
        'protect our', 'defend our', 'save our'
    ]

    authority_words = [
        'experts say', 'scientist confirm', 'studies show',
        'according to', 'research proves', 'doctor recommend',
        'officials say', 'report says', 'data shows', 'analysis shows',
        'harvard', 'stanford', 'mit', 'oxford', 'who says', 'cdc says',
        'fbi says', 'cia says', 'pentagon says', 'white house says',
        'peer reviewed', 'published study', 'clinical trial',
        'meta analysis', 'systematic review'
    ]

    demagogy_words = [
        'either', 'only way', 'no choice', 'must choose', 'or else',
        'with us or against us', 'no other option', 'only option',
        'you must', 'we must', 'have to choose', 'forced to',

        'they want', 'they will', 'they are trying', 'they dont want',
        'they lied', 'they hide', 'they control', 'they really',

        'traitor', 'real american', 'true patriot', 'fake news',
        'mainstream media', 'deep state', 'radical left', 'radical right',
        'socialist', 'fascist', 'communist', 'globalist', 'marxist',
        'liberal agenda', 'conservative agenda',

        'what they really mean', 'what he really said',
        'the truth about', 'the real agenda', 'the hidden truth',
        'what really happened', 'the real reason',

        'ordinary people', 'real people', 'working people',
        'elites', 'establishment', 'swamp', 'corrupt politicians',
        'out of touch', 'doesnt care about', 'against the people',

        'everyone knows', 'everybody knows', 'its obvious',
        'common sense', 'any fool can see', 'clearly',
        'of course they', 'naturally they', 'as expected'
    ]

    emotion_words = [
        'shocking', 'unbelievable', 'outrage', 'disgusting', 'outrageous',
        'heartbreaking', 'devastating', 'furious', 'horrifying', 'horrific',
        'incredible', 'explosive', 'bombshell', 'shameful', 'disgrace',
        'breaking', 'urgent', 'alert', 'must read', 'must watch',
        'share this', 'spread the word',
        'how dare', 'wake up', 'open your eyes',

        'sad', 'angry', 'upset', 'worried', 'afraid', 'scared',
        'hate', 'love', 'hope', 'fear', 'proud', 'ashamed',
        'wrong', 'unfair', 'unjust', 'immoral', 'evil', 'corrupt',
        'lies', 'lied', 'lying', 'deceive', 'deceived', 'betrayed',
        'failed', 'failing', 'failure', 'broken', 'ruined'
    ]

    rational_words = [
        'percent', 'per cent', 'percentage', 'statistics', 'data shows',
        'survey found', 'poll shows', 'numbers show',
        'study found', 'research indicates', 'evidence shows',
        'analysis shows', 'findings suggest', 'results show',
        'concluded that', 'demonstrated that',
        'according to data', 'factually', 'verified', 'confirmed by',
        'documented', 'on record', 'historically', 'historically speaking',
        'therefore', 'consequently', 'as a result', 'which means',
        'this indicates', 'this suggests', 'in conclusion'
    ]

    scores = {
        'fear_appeal': sum(1 for w in fear_words if w in text_lower),
        'authority_appeal': sum(1 for w in authority_words if w in text_lower),
        'demagogy_tricks': sum(1 for w in demagogy_words if w in text_lower),
        'emotional_manipulation': sum(1 for w in emotion_words if w in text_lower),
        'rational_argument': sum(1 for w in rational_words if w in text_lower),
    }

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return 'rational_argument'

    return best


# loading files
buzzfeed_fake = pd.read_csv('data/raw/BuzzFeed_fake_news_content.csv')
buzzfeed_real = pd.read_csv('data/raw/BuzzFeed_real_news_content.csv')
politifact_fake = pd.read_csv('data/raw/PolitiFact_fake_news_content.csv')
politifact_real = pd.read_csv('data/raw/PolitiFact_real_news_content.csv')


# merging
news = pd.concat([buzzfeed_fake, buzzfeed_real,
                 politifact_fake, politifact_real], ignore_index=True)


news['text'] = news['title'].fillna('') + '. ' + news['text'].fillna('')
news = news[['text']]

# liar dataset
liar_raw = pd.read_csv(
    'https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv',
    sep='\t', header=None,
    names=['filename', 'label', 'text', 'subject', 'speaker', 'job', 'state', 'party',
           'barely_true', 'false', 'half_true', 'mostly_true', 'pants_fire', 'context'],
    on_bad_lines='skip'
)

liar_valid = pd.read_csv(
    'https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv',
    sep='\t', header=None,
    names=['filename', 'label', 'text', 'subject', 'speaker', 'job', 'state', 'party',
           'barely_true', 'false', 'half_true', 'mostly_true', 'pants_fire', 'context'],
    on_bad_lines='skip'
)

liar_test = pd.read_csv(
    'https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv',
    sep='\t', header=None,
    names=['filename', 'label', 'text', 'subject', 'speaker', 'job', 'state', 'party',
           'barely_true', 'false', 'half_true', 'mostly_true', 'pants_fire', 'context'],
    on_bad_lines='skip'
)

liar_valid = liar_valid[['text']].copy()
liar_test = liar_test[['text']].copy()
liar = liar_raw[['text']].copy()

# merging
df = pd.concat([news, liar, liar_valid, liar_test], ignore_index=True)

df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.len() > 50]
df = df.drop_duplicates(subset='text')

# weak labeling
df['label'] = df['text'].apply(weak_label)

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)

    non_rational = df[df['label'] != 'rational_argument']
    max_rational = len(
        non_rational[non_rational['label'] == 'fear_appeal']) * 4

    rational = df[df['label'] == 'rational_argument'].sample(
        n=max_rational, random_state=42
    )

    df_balanced = pd.concat([non_rational, rational], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42)

    df_balanced[['text', 'label']].to_csv(
        'data/processed/combined_en.csv', index=False)

    print(df_balanced['label'].value_counts())
    print(f'total: {len(df)}')
