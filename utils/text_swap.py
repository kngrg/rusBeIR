import json

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            entry = json.loads(line)
            entry['text'] = entry.pop('processed_text')
            entry = {key: entry[key] for key in ['_id', 'text']}
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

input_file = '/Users/kaengreg/Documents/Работа /НИВЦ/storage/rus-mmarco-processed+default/queries.jsonl'
output_file = '/Users/kaengreg/Documents/Работа /НИВЦ/rus-mmarco-processed/queries.jsonl'

process_jsonl(input_file, output_file)
