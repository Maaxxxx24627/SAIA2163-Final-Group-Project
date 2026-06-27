"""
generate_visualizations.py
--------------------------
Run this script from the PROJECT ROOT (where app.py is located):
    python generate_visualizations.py

It will generate all 5 PNG files into notebooks/ folder:
    - notebooks/distribution.png
    - notebooks/wordcloud_negative.png
    - notebooks/wordcloud_neutral.png
    - notebooks/wordcloud_positive.png
    - notebooks/top20_words.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import os

# ── Setup paths ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "malaysian_sentiment_labeled.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "notebooks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")
print(df['sentiment'].value_counts())

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df['body'].apply(preprocess)

# ── Stopwords ─────────────────────────────────────────────────────────────────
stop_words = set([
    'the','a','an','is','it','in','on','at','to','for','of','and','or','but',
    'not','this','that','with','are','was','were','be','been','have','has',
    'had','will','would','could','should','do','did','does','from','by','as',
    'if','so','we','they','he','she','you','i','my','your','our','their','its',
    'what','which','who','how','when','where','why','all','more','also','just',
    'get','got','go','going','gone','come','came','said','say','says','know',
    'think','want','need','see','look','make','made','take','took','give',
    'given','even','one','two','much','many','some','any','than','then','now',
    'still','back','us','me','him','her','them','up','out','about','into',
    'after','before','between','through','during','same','other','new','good',
    'well','first','last','long','little','own','right','big','high','every',
    'never','always','already','without','because','while','let','put','end',
    'point','case','way','part','per','lot','time','year','day','month','thing',
    'something','anything','everything','nothing','someone','everyone','another',
    'around','next','really','actually','pretty','quite','enough','better',
    'less','mean','means','sure','feel','felt','keep','kept','start','stop',
    'try','tried','found','find','called','call',
    # Malay/Manglish
    'la','lah','je','kan','pun','nak','kat','tak','ada','yang','ni','tu','ke',
    'dah','nya','dia','kita','mereka','gomen','u','ur','im','dont','doesnt',
    'isnt','wasnt','arent','didnt','wont','cant','ive','youre','theyre',
    'https','http','www','com','gt','amp','etc','eg','pm','gov','govt',
    'di','untuk','dengan','dalam','dari','pada','ini','itu','juga','atau',
    'bagi','oleh','telah','akan','lebih','bila','tak','tapi','kalau','saya',
    'kami','semua','non','yes','didn','thats','doesn','cannot','yet','ago',
    'till','done','might','whole','full','hard','based','least','term','fact',
    'won','ask','saying','making','become','change','private','foreign',
    'chief','court','care','food','rich','poor','local','paid','cut',
    'service','support','market','policy','issues','epf','bring','trying',
    'using','continue','understand','reason','rather','suddenly','almost',
    'gonna','away','anyone','majority','instead','went','needs','afford',
    'remember','hope','man','guy','app','boleh','gaji','harga','sabah',
    'times','hours','leave','costs','amount','report','usually','until',
    'others','children','friends','compared','general','media','data','level',
    'number','small','post','example','experience','account','told','act',
    'currently','previously','immediately','towards','total','main','talk',
    'looking','takes','worse','great','easy','happy','fair','despite','global',
    'nation','improve','implement','purchasing','trade','trump','monthly',
    'gangster','rent','menteri','ibrahim','socso','budi95','rm2','billion',
    'million','school','medical','insurance','land','b40','100','pkr','dap',
    'vote','election','party','law','police','justice','social','housing',
    'covid','loan','hike','smuggling','development','protection','driving',
    'yeah','wait','true','huge','either','literally','seems','wants','face',
    'trust','show','kind','kena','week','believe','future','reduce','charge',
    'left','overseas','stupid','fuck','source','cause','voters','above','save',
    'check','later','read','please','matter','whether','unless','similar',
    'according','including','cheaper','cent','area','pump','kuala','lumpur',
    'earning','situation','financial','orang','fees','ever','whatever','fully',
    'longer','agree','consider','positive','via','add','seen','needed',
    'optimistic','licence','raise','project','finally','added','cuts','ppl',
    'direction','independence','terms','risk','decent','pembelian','confidence',
    'initiative','household','retirement','opinion','gig','foreigners','anyway',
    'mind','win','sad','imo','sum','cash','stuff','set','ones','bit','worth',
    'wrong','simply','half','exactly','tell','rise','comes','else','taking',
    'throughout','hand','opposition','judicial','fucking','bukan','minyak',
    'former','answer','imagine','plan','pandemic','bumi','zahid','katanya',
    'mrsm','berkata','dan','not','like','can','there','being','these','those',
    'his','here','don','off','over','only','most','such','since','use','used',
    'very','few','down','too','old','bad','again','against','under','top',
    'maybe','lol','shit','person','things','goes','getting','doing','working',
    'paying','years','today','free','low','due','help','life','world','may',
    'must','naik','bring','trying','using','continue','non','yes','didn',
    'thats','doesn','cannot','yet','ago','till','done','might','whole','full',
    'hard','based','least','term','fact','won','ask','saying','making',
    'become','change','private','foreign','chief','court','care','food',
])


# ── 1. DISTRIBUTION ───────────────────────────────────────────────────────────
print("\nGenerating distribution.png...")
counts = df['sentiment'].value_counts().reindex(['neutral', 'negative', 'positive'])
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4878CF', '#2CA4A0', '#6ACC65']
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_title(f'Sentiment Class Distribution (Total: ~{len(df):,} rows)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Sentiment Label')
ax.set_ylabel('Number of Reviews')
ax.set_ylim(0, counts.max() * 1.15)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, 'distribution.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out}")


# ── 2. WORD CLOUDS ────────────────────────────────────────────────────────────
def generate_wordcloud(sentiment, colormap, filename):
    print(f"\nGenerating {filename}...")
    subset = df[df['sentiment'] == sentiment]['cleaned']
    words = []
    for text in subset:
        tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
        words.extend(tokens)
    freq = Counter(words)
    wc = WordCloud(
        width=1000, height=500,
        background_color='white',
        colormap=colormap,
        max_words=120,
        collocations=False,
        min_font_size=10
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Most Frequent Tokens — {sentiment.capitalize()} Class',
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out} | Top 5: {[w for w,_ in freq.most_common(5)]}")

generate_wordcloud('negative', 'Reds',   'wordcloud_negative.png')
generate_wordcloud('neutral',  'Blues',  'wordcloud_neutral.png')
generate_wordcloud('positive', 'Greens', 'wordcloud_positive.png')


# ── 3. TOP 20 WORDS ───────────────────────────────────────────────────────────
print("\nGenerating top20_words.png...")
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
out = os.path.join(OUTPUT_DIR, 'top20_words.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out}")

print("\nAll visualizations generated successfully!")
print(f"Files saved in: {OUTPUT_DIR}")