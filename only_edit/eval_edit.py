import datetime
import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from demo import demo_model_editing
from utils.print_and_save import print_and_save

def extract_data(subject, relation, object_, data):
    """
    指定されたsubject, relation, objectに基づいてデータを抽出し、
    指定されたフォーマットで返します。
    Parameters:
        subject (str): 対象者の名前
        relation (str): 関係の種類
        object_ (str): 関係に関連するオブジェクト
    Returns:
        dict: 指定されたフォーマットの辞書
    """
    # まず、subjectが存在するか確認
    if subject not in data['subjects']:
        raise ValueError(f"Subject '{subject}' はデータセットに存在しません。")
    # 関係がSharedRelationsまたはExclusiveRelationsのどちらに属するかを確認
    if relation in data['SharedRelations']:
        relation_set = 'SharedRelations'
    elif relation in data['ExclusiveRelations']:
        relation_set = 'ExclusiveRelations'
    else:
        raise ValueError(f"Relation '{relation}' はデータセットに存在しません。")
    # TaskDescriptionPromptを除外
    if relation_set == 'SharedRelations' and relation == 'TaskDescriptionPrompt':
        raise ValueError(f"Relation '{relation}' はスキーマ情報であり、個別のデータではありません。")
    if relation_set == 'ExclusiveRelations' and relation == 'TaskDescriptionPrompt':
        raise ValueError(f"Relation '{relation}' はスキーマ情報であり、個別のデータではありません。")
    # 指定されたrelationの詳細を取得
    relation_details = data[relation_set].get(relation)
    if not relation_details:
        raise ValueError(f"Relation '{relation}' の詳細がデータセットに見つかりません。")
    # 指定されたobjectがrelationのobjectsに存在するか確認
    if object_ not in relation_details['objects']:
        raise ValueError(f"Object '{object_}' はRelation '{relation}' のオブジェクトに存在しません。")
    # データを指定されたフォーマットで作成
    formatted_data = {
        'subject': subject,
        'relation': relation,
        'object': object_,
        'objects': relation_details['objects'],
        'prompt': relation_details['prompt'],
        'question': relation_details['question']
    }
    return formatted_data

def evaluate_edit(model, tokenizer, question_prompt, objects):
    # トークン化し、入力としてモデルに渡す
    inputs = tokenizer(question_prompt, return_tensors="pt").to("cuda:0")
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]  # 最後のトークンのlogitを取得
    probabilities = F.softmax(logits, dim=-1)
    object_logits = {}
    object_probs = {}
    for obj in objects:
        tokens = tokenizer.tokenize(obj)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) > 1:
            obj_logit = logits[0, token_ids].mean().item()
            obj_prob = probabilities[0, token_ids].mean().item()
        else:
            obj_logit = logits[0, token_ids[0]].item()
            obj_prob = probabilities[0, token_ids[0]].item()
        object_logits[obj] = obj_logit
        object_probs[obj] = obj_prob
        print(f"Object: {obj}, Logit: {obj_logit}, Probability: {obj_prob}")
    sorted_objects = sorted(object_logits.items(), key=lambda item: item[1], reverse=True)
    return sorted_objects

# モデルのロード
MODEL_NAME = "EleutherAI/gpt-j-6B"
model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda:0"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token

# 現在の日時を取得してファイル名を生成
now = datetime.datetime.now()
formatted_date = now.strftime("%Y%m%d_%H%M%S")
file_path = f"../results/{formatted_date}.txt"
conditions = ["条件(a): Job", "条件(b): 複数のRelations", "条件(c): 共有型 (Shared Relations)", "条件(c): 排他型 (Exclusive Relations)"]
# データセットのjsonをロード
with open('../ckndb.json', 'r') as f:
    json_data = json.load(f)

task_description_prompt = "Select the correct answer to the question from the given options and list the corresponding single digit number.\n###Question###\n[question]\n###Sample Answer###\nA: n\n###Answer Options###\n1. [object1]\n2. [object2]\n3. [object3]\n4. [object4]\n5. [object5]\n###Answer###\nA: "
datasets = [
    # 条件(a): Job
    [
        {
            "subject": "Ryoma Ishigaki",
            "relation": "Job",
            "object": "Engineer"
        },
        {
            "subject": "Jundai Suzuki",
            "relation": "Job",
            "object": "Doctor"
        },
        {
            "subject": "Shun Iwase",
            "relation": "Job",
            "object": "Designer"
        },
        {
            "subject": "Reiya Hiramoto",
            "relation": "Job",
            "object": "Lawyer"
        },
        {
            "subject": "Masato Sekiguchi",
            "relation": "Job",
            "object": "Teacher"
        }
    ],
    # 条件(b): 複数のRelations
    [
        {
            "subject": "Ryoma Ishigaki",
            "relation": "ReadBooks",
            "object": "Harry Potter"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "Health Status",
            "object": "Hypertensive"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "VisitedPlaces",
            "object": "Cairo"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "CurrentLocation",
            "object": "Home"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "LearnedLanguages",
            "object": "French"
        }
    ],
    # 条件(c): 共有型 (Shared Relations)
    [
        {
            "subject": "Ryoma Ishigaki",
            "relation": "VisitedPlaces",
            "object": "Cairo"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "VisitedPlaces",
            "object": "New York"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "VisitedPlaces",
            "object": "Seoul"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "VisitedPlaces",
            "object": "Paris"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "VisitedPlaces",
            "object": "Rome"
        }
    ],
    # 条件(c): 排他型 (Exclusive Relations)
    [
        {
            "subject": "Ryoma Ishigaki",
            "relation": "Health Status",
            "object": "Hypertensive"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "Health Status",
            "object": "Allergic"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "Health Status",
            "object": "Diabetic"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "Health Status",
            "object": "Healthy"
        },
        {
            "subject": "Ryoma Ishigaki",
            "relation": "Health Status",
            "object": "Injured"
        }
    ]
]

for index, dataset in enumerate(datasets):
    print_and_save(conditions[index], file_path)
    print_and_save("各ステップごとの評価", file_path)
    for data in dataset:
        data = extract_data(data["subject"], data["relation"], data["object"], json_data)
        subject = data['subject']
        relation = data['relation']
        new_object = data['object']
        objects = data['objects']
        question = data['question'].replace("[subject]", subject)
        # JSONからgeneration_promptsを取得
        generation_prompts = [task_description_prompt
                        .replace("[question]", question)
                        .replace("[subject]", subject)]
        # JSONからpromptを取得
        object_placeholder_pos = data["prompt"].find("[object]")
        prompt_template = data['prompt'][:object_placeholder_pos]
        prompt = prompt_template.replace("[subject]", "{}")
        # 編集リクエストの作成
        request = [
            {
                "prompt": prompt,
                "subject": subject,
                "target_new": {"str": new_object}
            }
        ]
        # 知識編集の実行
        model_new, orig_weights, probs = demo_model_editing(
            model, tok, request, generation_prompts, file_path=file_path
        )
        # 評価: objectsのlogitが期待通りの順位にいるか確認
        sorted_objects = evaluate_edit(model, tok, question, objects)
        # 評価結果の出力
        # print_and_save(f"subject {subject}, relation {relation}, object {new_object}, question {question}", file_path)
        print_and_save(f"編集結果, relation {relation}: {sorted_objects}", file_path)
    print_and_save("最終状態での評価", file_path)
    for data in dataset:
        data = extract_data(data["subject"], data["relation"], data["object"], json_data)
        subject = data['subject']
        relation = data['relation']
        new_object = data['object']
        objects = data['objects']
        question = data['question'].replace("[subject]", subject)
        # JSONからgeneration_promptsを取得
        generation_prompts = [data['question'].replace("[subject]", subject)]
        # 評価: objectsのlogitが期待通りの順位にいるか確認
        sorted_objects = evaluate_edit(model, tok, question, objects)
        # 評価結果の出力
        # print_and_save(f"subject {subject}, relation {relation}, object {new_object}, question {question}", file_path)
        print_and_save(f"編集結果, relation {relation}: {sorted_objects}", file_path)
    # モデルをGPUから降ろす
    model.to('cpu')
    # モデルの参照を削除
    del model
    # キャッシュをクリア
    torch.cuda.empty_cache()
    if index != (len(datasets)-1):
        # モデルのロード
        MODEL_NAME = "EleutherAI/gpt-j-6B"
        model, tok = (
            AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda:0"),
            AutoTokenizer.from_pretrained(MODEL_NAME),
        )
        tok.pad_token = tok.eos_token
# import os
# import datetime
# import argparse
# import json
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from demo import demo_model_editing
# from utils.print_and_save import print_and_save

# # コマンドライン引数の設定
# parser = argparse.ArgumentParser(description="Knowledge editing based on pattern")
# parser.add_argument('--pattern', type=str, required=True, help='Pattern to use for knowledge editing (e.g., Skill_pattern_1)')
# args = parser.parse_args()

# # CSVファイルの読み込み
# csv_file_path = '../sro_combinations.csv'
# sro_data = pd.read_csv(csv_file_path)

# # JSONデータの読み込み（このJSONには各relationのプロンプト情報が含まれていると仮定）
# with open('../ckndb.json', 'r') as f:
#     json_data = json.load(f)

# # パターンごとにデータをフィルタリングする関数
# def filter_by_pattern(data, pattern_name):
#     return data[data['Pattern'] == pattern_name]

# # モデルのロード
# MODEL_NAME = "EleutherAI/gpt-j-6B"
# model, tok = (
#     AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda:0"),
#     AutoTokenizer.from_pretrained(MODEL_NAME),
# )
# tok.pad_token = tok.eos_token

# # 特定の relation に基づいて JSON からプロンプトを取得し、フォーマットを適用
# def get_prompts_from_json(relation_name, subject, new_object):
#     if relation_name in json_data["relations_with_multiple_objects"]:
#         relation_data = json_data["relations_with_multiple_objects"][relation_name]
#         task_description_prompt = json_data["relations_with_multiple_objects"]["Task_Description_Prompt"]
#     elif relation_name in json_data["relations_with_overwriting"]:
#         relation_data = json_data["relations_with_overwriting"][relation_name]
#         task_description_prompt = json_data["relations_with_overwriting"]["Task_Description_Prompt"]
#     else:
#         raise ValueError(f"Relation '{relation_name}' not found in the JSON data")

#     # [object1], [object2], [object3], などを objects リストの値で置換
#     objects = relation_data["objects"]  # JSON 内の object リストを取得
    
#     # プレースホルダーを動的に置換（存在しない場合は無視）
#     for i, obj in enumerate(objects, 1):  # 1 からスタートして [object1], [object2] に対応
#         task_description_prompt = task_description_prompt.replace(f"[object{i}]", obj)
    
#     # [subject] および [question] を置換
#     generation_prompts = [task_description_prompt
#                         .replace("[question]", relation_data["question"])
#                         .replace("[subject]", subject)]
    
#     # "[object]" より前の部分を取り出して使用
#     object_placeholder_pos = relation_data["prompt"].find("[object]")
#     if object_placeholder_pos == -1:
#         raise ValueError(f"Relation '{relation_name}' prompt does not contain '[object]'")
#     prompt_template = relation_data["prompt"][:object_placeholder_pos]
    
#     return generation_prompts, prompt_template, relation_data["question"].replace("[subject]", subject), objects

# # 引数から編集するパターンを取得
# pattern_to_edit = args.pattern

# # 指定されたパターンに該当するデータをフィルタリング
# filtered_data = filter_by_pattern(sro_data, pattern_to_edit)

# # 編集後のモデルの評価関数
# def evaluate_edit(model, tokenizer, question_prompt, objects):
#     # トークン化し、入力としてモデルに渡す
#     inputs = tokenizer(question_prompt, return_tensors="pt").to("cuda:0")
#     outputs = model(**inputs)
#     logits = outputs.logits[:, -1, :]  # 最後のトークンのlogitを取得

#     # 各objectのlogitを取得し、上位にいるか確認
#     object_logits = {obj: logits[0, tokenizer.convert_tokens_to_ids(obj)].item() for obj in objects}
#     sorted_objects = sorted(object_logits.items(), key=lambda item: item[1], reverse=True)
    
#     return sorted_objects

# # 現在の日時を取得してファイル名を生成
# now = datetime.datetime.now()
# formatted_date = now.strftime("%Y%m%d_%H%M%S")
# file_path = f"../results/{pattern_to_edit}_{formatted_date}.txt"

# # 各編集後の評価と結果の出力
# for index, row in filtered_data.iterrows():
#     # 編集プロンプトの生成と編集リクエストの準備（既存コードと同じ）
#     subject = row['Subject (s)']
#     relation = row['Relation (r)']
#     new_object = row['Object (o)']
    
#     # JSONからプロンプトを取得
#     generation_prompts, prompt_template, question, objects = get_prompts_from_json(relation, subject, new_object)
#     # promptをsubjectに合わせてフォーマット
#     prompt = prompt_template.replace("[subject]", "{}")
#     # 編集リクエストの作成
#     request = [
#         {
#             "prompt": prompt,
#             "subject": subject,
#             "target_new": {"str": new_object},
#         }
#     ]

#     # 知識編集の実行
#     model_new, orig_weights, probs = demo_model_editing(
#         model, tok, request, generation_prompts, file_path=file_path
#     )

#     # 評価: objectsのlogitが期待通りの順位にいるか確認
#     sorted_objects = evaluate_edit(model_new, tok, question, objects)

#     # 評価結果の出力
#     print_and_save(f"編集結果 - index {index+1}, relation {relation}: {sorted_objects}", file_path)
#     # `relations_with_multiple_objects`または`relations_with_overwriting`で順位をチェックする追加処理