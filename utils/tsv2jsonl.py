import csv
import json
import pandas as pd
from tqdm import tqdm

def tsv_to_jsonl(tsv_file, jsonl_file):
    df = pd.read_csv(tsv_file, sep='\t', header=None, encoding='utf-8')
    df.columns = ['_id', 'text']

    with open(jsonl_file, 'a', encoding='utf-8') as jsonl_f:
        for i in tqdm(range(0, len(df)), desc="Converting tsv to jsonl"):
            #row = {"_id": str(df['_id'][i]), "title": '', "text": df['text'][i]}
            row = {"_id": str(df['_id'][i]),"text": df['text'][i]}
            jsonl_f.write(json.dumps(row, ensure_ascii=False) + '\n')

#tsv_to_jsonl('/Users/kaengreg/Documents/Работа /НИВЦ/datasets/rus-mmarco-google/russian_queries.train.tsv',
#             '/Users/kaengreg/Documents/Работа /НИВЦ/datasets/rus-mmarco-google/queries_default.jsonl')


def count_lines_in_jsonl(jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        count = sum(1 for _ in file)
    return count

file_path = '/Users/kaengreg/Documents/Работа /НИВЦ/datasets/rus-mmarco-google/corpus_default.jsonl'
lines_count = count_lines_in_jsonl(file_path)
print(f'Количество строк в файле: {lines_count}')