import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

df = pd.read_csv('data/malaysian_sentiment_labeled.csv')

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df['body'].apply(preprocess)

stop_words = set([
    'the', 'a', 'an', 'is', 'it', 'in', 'on', 'at', 'to', 'for', 'of',
    'and', 'or', 'but', 'not', 'this', 'that', 'with', 'are', 'was',
    'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could',
    'should', 'do', 'did', 'does', 'from', 'by', 'as', 'if', 'so', 'we',
    'they', 'he', 'she', 'you', 'i', 'my', 'your', 'our', 'their', 'its',
    'what', 'which', 'who', 'how', 'when', 'where', 'why', 'all', 'more',
    'also', 'just', 'get', 'got', 'go', 'going', 'gone', 'come', 'came',
    'said', 'say', 'says', 'know', 'think', 'want', 'need', 'see', 'look',
    'make', 'made', 'take', 'took', 'give', 'given', 'even', 'one', 'two',
    'much', 'many', 'some', 'any', 'than', 'then', 'now', 'still', 'back',
    'us', 'me', 'him', 'her', 'them', 'up', 'out', 'about', 'into', 'after',
    'before', 'between', 'through', 'during', 'same', 'other', 'new', 'good',
    'well', 'first', 'last', 'long', 'little', 'own', 'right', 'big', 'high',
    'every', 'never', 'always', 'already', 'without', 'because', 'while',
    'let', 'put', 'end', 'point', 'case', 'way', 'part', 'per', 'lot',
    'time', 'year', 'day', 'month', 'thing', 'something', 'anything',
    'everything', 'nothing', 'someone', 'everyone', 'another', 'around',
    'next', 'really', 'actually', 'pretty', 'quite', 'enough', 'better',
    'less', 'mean', 'means', 'sure', 'feel', 'felt', 'keep', 'kept',
    'start', 'stop', 'try', 'tried', 'found', 'find', 'called', 'call',
    'like', 'can', 'there', 'being', 'these', 'those', 'his', 'here',
    'don', 'off', 'over', 'only', 'most', 'such', 'since', 'use', 'used',
    'very', 'few', 'down', 'too', 'old', 'bad', 'again', 'against',
    'under', 'top', 'maybe', 'lol', 'shit', 'person', 'things', 'thing',
    'goes', 'going', 'getting', 'doing', 'working', 'paying', 'being',
    'years', 'today', 'free', 'low', 'high', 'due', 'help', 'life',
    'world', 'may', 'must', 'upon', 'both', 'each', 'few', 'been',
    'into', 'through', 'during', 'having', 'doing', 'naik',
    'non', 'yes', 'didn', 'thats', 'doesn', 'cannot', 'yet', 'ago',
    'till', 'done', 'might', 'whole', 'full', 'hard', 'based', 'least',
    'term', 'fact', 'won', 'ask', 'saying', 'making', 'become', 'change',
    'class', 'private', 'foreign', 'chief', 'court', 'care', 'food',
    'rich', 'poor', 'local', 'paid', 'cut', 'service', 'support',
    'market', 'policy', 'issues', 'epf',
    'yes', 'non', 'full', 'might', 'hard', 'paid', 'once', 'ask',
    'probably', 'however', 'though', 'especially', 'different', 'able',
    'place', 'days', 'months', 'current', 'average', 'based', 'least',
    'lower', 'higher', 'live', 'buy', 'amongst', 'kids', 'home',
    '000', 'rm1', '2025', 'thats', 'doesn', 'cheap', 'best',
    'tidak', 'kepada', 'negara', 'seri', 'datuk',
    'yes', 'best', 'hard', 'done', 'ago', 'home', 'far', 'real',
    'definitely', 'especially', 'however', 'perfect', 'thank',
    'making', 'move', 'gets', 'makes', 'spend', 'months',
    'class', 'term', 'state', 'house', 'car', 'tech', 'step',
    'isn', 'doesn', 'datuk', 'kerajaan', 'kesihatan', 'klinik',
    'mykad', 'lhdn', 'invoice',
    'bring', 'trying', 'using', 'continue', 'understand', 'reason',
    'rather', 'suddenly', 'almost', 'gonna', 'away', 'anyone',
    'majority', 'instead', 'went', 'needs', 'afford', 'remember',
    'hope', 'man', 'guy', 'app', 'boleh', 'gaji', 'harga', 'sabah',
    'times', 'hours', 'leave', 'costs', 'amount', 'report', 'usually',
    'until', 'others', 'children', 'friends', 'compared', 'general',
    'media', 'data', 'level', 'number', 'small', 'post', 'example',
    'experience', 'account', 'told', 'act', 'currently', 'previous',
    'immediately', 'towards', 'total', 'main', 'talk', 'looking',
    'takes', 'worse', 'great', 'easy', 'happy', 'fair', 'despite',
    'global', 'nation', 'improve', 'implement', 'implementation',
    'purchasing', 'trade', 'trump', 'monthly', 'gangster', 'rent',
    'menteri', 'ibrahim', 'socso', 'budi95', 'rm2', 'billion',
    'million', 'school', 'medical', 'insurance', 'land', 'b40',
    '100', 'pkr', 'dap', 'vote', 'election', 'party', 'law',
    'police', 'justice', 'social', 'housing', 'covid', 'loan',
    'hike', 'smuggling', 'development', 'protection', 'driving','dan',
    'yeah', 'wait', 'true', 'huge', 'either', 'literally', 'seems',
    'wants', 'face', 'trust', 'show', 'kind', 'kena', 'week',
    'believe', 'future', 'reduce', 'charge', 'left', 'overseas',
    'stupid', 'fuck', 'source', 'cause', 'voters',
    'above', 'save', 'check', 'later', 'read', 'please', 'matter',
    'whether', 'unless', 'similar', 'according', 'including',
    'cheaper', 'cent', 'area', 'pump', 'kuala', 'lumpur',
    'earning', 'situation', 'financial', 'orang', 'fees',
    'ever', 'whatever', 'unless', 'fully', 'longer', 'agree',
    'consider', 'positive', 'via', 'add', 'check', 'seen',
    'needed', 'optimistic', 'licence', 'raise', 'project',
    'give', 'providing', 'finally', 'added', 'cuts', 'ppl',
    'direction', 'independence', 'terms', 'risk', 'decent',
    'pembelian', 'confidence', 'initiative', 'household',
    'retirement', 'opinion', 'gig', 'foreigners', 'whatever',
    'la', 'lah', 'je', 'je', 'kan', 'pun', 'nak', 'kat', 'tak', 'ada',
    'yang', 'ni', 'tu', 'ke', 'dah', 'nya', 'dia', 'kita', 'mereka',
    'gomen', 'u', 'ur', 'im', 'dont', 'doesnt', 'isnt', 'wasnt',
    'arent', 'didnt', 'wont', 'cant', 'ive', 'youre', 'theyre',
    'https', 'http', 'www', 'com', 'gt', 'amp', 'etc', 'eg',
    'within', 'would', 'could', 'pm', 'gov', 'govt',
    'di', 'untuk', 'dengan', 'dalam', 'dari', 'pada', 'ini', 'itu',
    'juga', 'atau', 'bagi', 'oleh', 'telah', 'akan', 'lebih', 'bila',
    'tak', 'tapi', 'kalau', 'bila', 'saya', 'kami', 'mereka', 'semua',
])

all_words = []
for text in df['cleaned']:
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    all_words.extend(tokens)

word_freq = Counter(all_words)
top20 = word_freq.most_common(20)
words, freqs = zip(*top20)

fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.cm.RdYlBu_r([i/20 for i in range(20)])
bars = ax.barh(list(words)[::-1], list(freqs)[::-1], color=colors[::-1], edgecolor='white')

for bar, val in zip(bars, list(freqs)[::-1]):
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=9, fontweight='bold')

ax.set_title('Top 20 Most Common Words (Stopwords Filtered)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Frequency')
ax.set_xlim(0, max(freqs) * 1.12)
plt.tight_layout()
plt.savefig('notebooks/top20_words.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Top 5: {[w for w,_ in top20[:5]]}')