import json

def compare_jsonl_files(file1, file2):
    def load_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return {json.loads(line)['_id']: json.loads(line) for line in f}

    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)

    ids_only_in_file1 = set(data1.keys()) - set(data2.keys())
    ids_only_in_file2 = set(data2.keys()) - set(data1.keys())

    print(ids_only_in_file1)

    return len(ids_only_in_file1), len(ids_only_in_file2)


file1 = '/Users/kaengreg/Documents/Работа /НИВЦ/scifact/corpus.jsonl'
file2 = '/Users/kaengreg/Documents/Работа /НИВЦ/scifact/ru_corpus.jsonl'
count_in_file1_only, count_in_file2_only = compare_jsonl_files(file1, file2)
print(f"Количество _id из первого файла, которых нет во втором: {count_in_file1_only}")
print(f"Количество _id из второго файла, которых нет в первом: {count_in_file2_only}")
