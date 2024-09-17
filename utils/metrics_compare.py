import json
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

eng = load_json('/Users/kaengreg/Documents/Работа /НИВЦ/scifact/queries_metrics_eng.json')
rus = load_json('/Users/kaengreg/Documents/Работа /НИВЦ/scifact/queries_metrics_rus.json')

differences = {
    key: abs(eng[key]["ndcg"]["NDCG@10"] - rus[key]["ndcg"]["NDCG@10"])
    for key in eng
}

max_difference_id = max(differences, key=differences.get)
max_difference = differences[max_difference_id]

#print("Разницы между значениями NDCG@10 для одинаковых ключей:", differences)
#print("ID записи с максимальной разницей:", max_difference_id)
#print("Максимальная разница:", max_difference)

sorted_dif = dict(sorted(differences.items(), key=lambda x: x[1], reverse=True))

print(sorted_dif)