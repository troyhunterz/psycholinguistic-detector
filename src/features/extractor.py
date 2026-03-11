import spacy

nlp = spacy.load('en_core_web_sm')


def extract_features(text: str) -> dict:
    """
    Extract psycholinguistic features from text
    Return dict of numerical features
    """
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    sentences = list(doc.sents)
    n_tokens = len(tokens) + 1

    we_words = {'we', 'our', 'us'}
    they_words = {'they', 'them', 'their', 'enemy', 'enemies'}
    modal_words = {'must', 'should', 'need', 'have to', 'ought'}
    logic_words = ['because', 'therefore', 'thus',
                   'however', 'consequently', 'moreover']

    we_ratio = sum(1 for t in tokens if t.lemma_.lower()
                   in we_words) / n_tokens

    they_ratio = sum(1 for t in tokens if t.lemma_.lower()
                     in they_words) / n_tokens

    exclaim_ratio = sum(
        1 for s in sentences if '!' in s.text) / (len(sentences) + 1)

    question_ratio = sum(
        1 for s in sentences if '?' in s.text) / (len(sentences) + 1)

    modal_ratio = sum(1 for t in tokens if t.lemma_.lower()
                      in modal_words) / n_tokens

    logic_count = sum(1 for lw in logic_words if lw in text.lower())
    adj_ratio = sum(1 for t in tokens if t.pos_ == 'ADJ') / n_tokens
    verb_ratio = sum(1 for t in tokens if t.pos_ == 'VERB') / n_tokens
    avg_sent_len = sum(len(list(s)) for s in sentences) / (len(sentences) + 1)
    caps_ratio = sum(1 for t in tokens if t.text.isupper()
                     and len(t.text) > 1) / n_tokens

    return {
        'we_ratio': we_ratio,
        'they_ratio': they_ratio,
        'exclaim_ratio': exclaim_ratio,
        'question_ratio': question_ratio,
        'modal_ratio': modal_ratio,
        'logic_count': logic_count,
        'adj_ratio': adj_ratio,
        'verb_ratio': verb_ratio,
        'avg_sent_len': avg_sent_len,
        'caps_ratio': caps_ratio
    }
