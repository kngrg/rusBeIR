import re
import pymorphy3
from nltk.corpus import stopwords
import nltk
import json
from tqdm import tqdm

nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))
morph = pymorphy3.MorphAnalyzer()
russian_stopwords.add('который')
russian_stopwords.add('такой')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    processed_words = [lemma for lemma in lemmas if lemma not in russian_stopwords]
    processed_text = ' '.join(processed_words)

    return processed_text

input_file = "/Users/kaengreg/Documents/Работа /НИВЦ/datasets/ria-news/corpus_default.jsonl"
output_file = "/Users/kaengreg/Documents/Работа /НИВЦ/datasets/ria-news/corpus.jsonl"

with open(input_file, 'r', encoding='utf-8') as infile:
    total_lines = sum(1 for _ in infile)

corpus = {}
with open(input_file, 'r') as file:
    for line in tqdm(file, total=total_lines, desc='Preprocessing text'):
        record = json.loads(line)
        record['processed_text'] = preprocess_text(record['text'])
        corpus[record['_id']] = record

with open(output_file, 'w', encoding='utf-8') as outfile:
    for cid in corpus.keys():
      outfile.write(json.dumps(corpus[cid], ensure_ascii=False) + '\n')


