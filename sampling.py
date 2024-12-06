import random

# 提供されたデータセット（Pythonの辞書として定義）
data = {
    "subjects": [
        "Ryoma Ishigaki",
        "Jundai Suzuki",
        "Shun Iwase",
        "Reiya Hiramoto",
        "Masato Sekiguchi"
    ],
    "SharedRelations": {
        "TaskDescriptionPrompt": "Select all the correct answers to the question from the given options and list the corresponding numbers.\n###Question###\n[question]\n###Sample Answer###\nA: n, m, k\n###Answer Options###\n1. [object1]\n2. [object2]\n3. [object3]\n4. [object4]\n5. [object5]\n###Answer###\nA: ",
        "Skills": {
            "objects": ["Python", "SQL", "Java", "Excel", "HTML"],
            "prompt": "[subject] is skilled in [object].",
            "question": "Which skills does [subject] have?"
        },
        "Hobbies": {
            "objects": ["Reading", "Hiking", "Cooking", "Gaming", "Photography"],
            "prompt": "[subject] enjoys the hobby of [object].",
            "question": "Which hobbies does [subject] enjoy?"
        },
        "LearnedLanguages": {
            "objects": ["English", "Japanese", "Spanish", "French", "German"],
            "prompt": "[subject] has learned the language [object].",
            "question": "Which languages has [subject] learned?"
        },
        "ReadBooks": {
            "objects": ["The Little Prince", "Harry Potter", "The Lord of the Rings", "The Chronicles of Narnia", "A Bear Called Paddington"],
            "prompt": "[subject] has read the book titled [object].",
            "question": "Which books has [subject] read?"
        },
        "VisitedPlaces": {
            "objects": ["Paris", "New York", "Rome", "Seoul", "Cairo"],
            "prompt": "[subject] has visited [object].",
            "question": "Where has [subject] been?"
        }
    },
    "ExclusiveRelations": {
        "TaskDescriptionPrompt": "Select the correct answer to the question from the given options and list the corresponding single digit number.\n###Question###\n[question]\n###Sample Answer###\nA: n\n###Answer Options###\n1. [object1]\n2. [object2]\n3. [object3]\n4. [object4]\n5. [object5]\n###Answer###\nA: ",
        "Health Status": {
            "objects": ["Healthy", "Injured", "Allergic", "Diabetic", "Hypertensive"],
            "prompt": "[subject] is currently experiencing the health status of [object].",
            "question": "Which health statuses does [subject] have?"
        },
        "Job": {
            "objects": ["Engineer", "Doctor", "Teacher", "Designer", "Lawyer"],
            "prompt": "[subject] currently works as [object].",
            "question": "Which jobs does [subject] have?"
        },
        "Residence": {
            "objects": ["Sapporo", "Tokyo", "Nagoya", "Osaka", "Fukuoka"],
            "prompt": "[subject] resides in [object].",
            "question": "Which residences does [subject] reside in?"
        },
        "CurrentLocation": {
            "objects": ["Home", "Office", "Library", "Restaurant", "Station"],
            "prompt": "[subject] is currently located in [object].",
            "question": "Where is [subject] currently located?"
        },
        "TravelFrequency": {
            "objects": ["Never", "Rarely", "Occasionally", "Frequently", "Very Frequently"],
            "prompt": "[subject] travels [object].",
            "question": "How often does [subject] travel?"
        }
    }
}

# 条件1: 複数のsubjectに対する連続的な編集
def condition_multiple_subjects_single_relation(data):
    # 5つのsubjectを取得
    subjects = data['subjects']
    
    # 使用するrelationを選択（例: "Job"）
    relation_type = 'ExclusiveRelations'
    relation_name = 'Job'
    relation_data = data[relation_type][relation_name]
    
    # 各subjectに対して1つのobjectを割り当てる
    objects = relation_data['objects']
    selected_objects = random.sample(objects, k=5)  # 各subjectに異なるobjectを割り当てる
    
    # 編集内容を作成
    edits = []
    for subject, obj in zip(subjects, selected_objects):
        prompt = relation_data['prompt'].replace('[subject]', subject).replace('[object]', obj)
        question = relation_data['question'].replace('[subject]', subject)
        edits.append({
            'subject': subject,
            'relation': relation_name,
            'object': obj,
            'prompt': prompt,
            'question': question
        })
    return edits

# 条件2: 同一のsubjectに対する複数のrelationの編集
def condition_single_subject_multiple_relations(data):
    # 1つのsubjectを選択
    subject = random.choice(data['subjects'])
    
    # 5つのrelationを選択（SharedRelationsとExclusiveRelationsからランダムに選ぶ）
    relations = []
    relations.extend([(key, 'SharedRelations') for key in data['SharedRelations'] if key != 'TaskDescriptionPrompt'])
    relations.extend([(key, 'ExclusiveRelations') for key in data['ExclusiveRelations'] if key != 'TaskDescriptionPrompt'])
    selected_relations = random.sample(relations, k=5)
    
    # 各relationに対して1つのobjectを割り当てる
    edits = []
    for relation_name, relation_type in selected_relations:
        relation_data = data[relation_type][relation_name]
        obj = random.choice(relation_data['objects'])
        prompt = relation_data['prompt'].replace('[subject]', subject).replace('[object]', obj)
        question = relation_data['question'].replace('[subject]', subject)
        edits.append({
            'subject': subject,
            'relation': relation_name,
            'object': obj,
            'prompt': prompt,
            'question': question
        })
    return edits

# 条件3: 同一の(s, r)ペアに対するobjectの再編集（共有型）
def condition_single_subject_shared_relation_multiple_objects(data):
    # 1つのsubjectを選択
    subject = random.choice(data['subjects'])
    
    # 共有型relationを1つ選択
    relations = [key for key in data['SharedRelations'] if key != 'TaskDescriptionPrompt']
    relation_name = random.choice(relations)
    relation_data = data['SharedRelations'][relation_name]
    
    # 5つのobjectを選択
    objects = relation_data['objects']
    selected_objects = random.sample(objects, k=5)
    
    # 編集内容を作成
    edits = []
    for obj in selected_objects:
        prompt = relation_data['prompt'].replace('[subject]', subject).replace('[object]', obj)
        question = relation_data['question'].replace('[subject]', subject)
        edits.append({
            'subject': subject,
            'relation': relation_name,
            'object': obj,
            'prompt': prompt,
            'question': question
        })
    return edits

# 条件4: 同一の(s, r)ペアに対するobjectの再編集（排他型）
def condition_single_subject_exclusive_relation_multiple_objects(data):
    # 1つのsubjectを選択
    subject = random.choice(data['subjects'])
    
    # 排他型relationを1つ選択
    relations = [key for key in data['ExclusiveRelations'] if key != 'TaskDescriptionPrompt']
    relation_name = random.choice(relations)
    relation_data = data['ExclusiveRelations'][relation_name]
    
    # 5つのobjectを選択
    objects = relation_data['objects']
    selected_objects = random.sample(objects, k=5)
    
    # 編集内容を作成
    edits = []
    for obj in selected_objects:
        prompt = relation_data['prompt'].replace('[subject]', subject).replace('[object]', obj)
        question = relation_data['question'].replace('[subject]', subject)
        edits.append({
            'subject': subject,
            'relation': relation_name,
            'object': obj,
            'prompt': prompt,
            'question': question
        })
    return edits

# メインの実行部分
def main():
    random.seed(42)  # 再現性のためシードを設定

    # 条件1の実行と結果表示
    print("条件1: 複数のsubjectに対する連続的な編集")
    edits1 = condition_multiple_subjects_single_relation(data)
    for edit in edits1:
        print(edit)
    print("\n")

    # 条件2の実行と結果表示
    print("条件2: 同一のsubjectに対する複数のrelationの編集")
    edits2 = condition_single_subject_multiple_relations(data)
    for edit in edits2:
        print(edit)
    print("\n")

    # 条件3の実行と結果表示
    print("条件3: 同一の(s, r)ペアに対するobjectの再編集（共有型）")
    edits3 = condition_single_subject_shared_relation_multiple_objects(data)
    for edit in edits3:
        print(edit)
    print("\n")

    # 条件4の実行と結果表示
    print("条件4: 同一の(s, r)ペアに対するobjectの再編集（排他型）")
    edits4 = condition_single_subject_exclusive_relation_multiple_objects(data)
    for edit in edits4:
        print(edit)
    print("\n")

if __name__ == "__main__":
    main()