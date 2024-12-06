import json
import pandas as pd

# JSONファイルを読み込む関数
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 組み合わせ生成関数（(s, r, o)形式）
def generate_sro_combinations(subjects, relation_name, relation_objects):
    # 1. 5つのsubjectに対して1つのobject
    pattern_1 = [(subject, relation_name, obj) for subject, obj in zip(subjects, relation_objects)]
    # 2. 1つのsubjectに5つのobject
    pattern_2 = [(subjects[0], relation_name, obj) for obj in relation_objects]
    # 3. 1つのobjectを5回更新
    pattern_3 = [(subjects[0], relation_name, relation_objects[i]) for i in range(5)]
    return pattern_1, pattern_2, pattern_3

# JSONファイルのパス
file_path = 'ckndb.json'
# JSONデータの読み込み
data = load_json_file(file_path)
# 結果を保存する辞書
results = {}

# 各評価条件に応じたオブジェクトを(s, r, o)形式で生成し、辞書に保存
for relation_name, relation_data in data["relations_with_multiple_objects"].items():
    if isinstance(relation_data, dict) and "objects" in relation_data:
        pattern_1, pattern_2, pattern_3 = generate_sro_combinations(data["subjects"], relation_name, relation_data["objects"])
        # パターンを辞書に保存
        results[f'{relation_name}_pattern_1'] = pattern_1
        results[f'{relation_name}_pattern_2'] = pattern_2
        results[f'{relation_name}_pattern_3'] = pattern_3

# 上書きする関係についても同様に処理
for relation_name, relation_data in data["relations_with_overwriting"].items():
    if isinstance(relation_data, dict) and "objects" in relation_data:
        pattern_1, pattern_2, pattern_3 = generate_sro_combinations(data["subjects"], relation_name, relation_data["objects"])
        # パターンを辞書に保存
        results[f'{relation_name}_pattern_1'] = pattern_1
        results[f'{relation_name}_pattern_2'] = pattern_2
        results[f'{relation_name}_pattern_3'] = pattern_3

# 辞書をデータフレームに変換するため、リストをフラットにしデータフレームとして扱います。
flattened_results = []

for pattern_name, sro_list in results.items():
    for sro in sro_list:
        flattened_results.append({
            "Pattern": pattern_name,
            "Subject (s)": sro[0],
            "Relation (r)": sro[1],
            "Object (o)": sro[2]
        })

# データフレームに変換
df_results = pd.DataFrame(flattened_results)
# データフレームをCSVファイルとして保存
output_file_path = 'sro_combinations.csv'
df_results.to_csv(output_file_path, index=False)
# ファイルが生成されました。
print(f"データは {output_file_path} に保存されました。")