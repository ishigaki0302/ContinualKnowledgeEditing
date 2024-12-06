import numpy as np
import re
from collections import OrderedDict
import ast

def softmax(logits):
    """ソフトマックス関数の実装"""
    logits = np.array(logits)
    exps = np.exp(logits - np.max(logits))  # オーバーフロー対策
    return exps / np.sum(exps)

# データをマルチライン文字列として定義
data = """
条件(a): Job
各ステップごとの評価
編集結果, relation Job: [('Lawyer', 2.1896374225616455), ('Engineer', 1.589766263961792), ('Designer', 0.8959683775901794), ('Doctor', 0.8552886247634888), ('Teacher', 0.289331316947937)]
編集結果, relation Job: [('Doctor', 7.255606651306152), ('Designer', 3.5805797576904297), ('Engineer', 2.746816635131836), ('Teacher', 1.6775884628295898), ('Lawyer', 0.6752175092697144)]
編集結果, relation Job: [('Designer', 3.9436371326446533), ('Doctor', 2.777327060699463), ('Engineer', 0.8089579343795776), ('Teacher', 0.5117523670196533), ('Lawyer', -0.11623761057853699)]
編集結果, relation Job: [('Lawyer', 6.777016639709473), ('Teacher', 4.037986755371094), ('Doctor', 2.649456262588501), ('Designer', 1.5044102668762207), ('Engineer', -0.3900315761566162)]
編集結果, relation Job: [('Teacher', 3.665088415145874), ('Doctor', 3.1563849449157715), ('Designer', 2.629030704498291), ('Engineer', 2.2337446212768555), ('Lawyer', 2.1982719898223877)]
最終状態での評価
編集結果, relation Job: [('Lawyer', 2.856492042541504), ('Engineer', 1.6734323501586914), ('Doctor', 1.6650710105895996), ('Designer', 1.0305776596069336), ('Teacher', -0.8252986669540405)]
編集結果, relation Job: [('Doctor', 7.384286880493164), ('Designer', 3.51212739944458), ('Engineer', 2.802394151687622), ('Teacher', 1.9348301887512207), ('Lawyer', 0.7892034649848938)]
編集結果, relation Job: [('Designer', 4.148184299468994), ('Doctor', 3.1407864093780518), ('Engineer', 1.0137419700622559), ('Teacher', 0.5697891712188721), ('Lawyer', -0.09290960431098938)]
編集結果, relation Job: [('Lawyer', 7.290070533752441), ('Teacher', 4.105413913726807), ('Doctor', 3.315387010574341), ('Designer', 1.7193717956542969), ('Engineer', -0.09374022483825684)]
編集結果, relation Job: [('Teacher', 3.665088415145874), ('Doctor', 3.1563849449157715), ('Designer', 2.629030704498291), ('Engineer', 2.2337446212768555), ('Lawyer', 2.1982719898223877)]
条件(b): 複数のRelations
各ステップごとの評価
編集結果, relation ReadBooks: [('Harry Potter', 13.530369758605957), ('The Little Prince', 8.433137893676758), ('The Lord of the Rings', 7.084578037261963), ('A Bear Called Paddington', 5.31906270980835), ('The Chronicles of Narnia', 5.090750217437744)]
編集結果, relation Health Status: [('Hypertensive', 5.4398088455200195), ('Healthy', 4.602020263671875), ('Injured', 1.1853758096694946), ('Diabetic', 0.8749350309371948), ('Allergic', -0.8810122013092041)]
編集結果, relation VisitedPlaces: [('Cairo', 8.228116035461426), ('Paris', 6.1020283699035645), ('Rome', 6.064844131469727), ('New York', 5.6602888107299805), ('Seoul', 3.9677584171295166)]
編集結果, relation CurrentLocation: [('Home', 6.690392017364502), ('Office', 4.393942356109619), ('Library', 3.346142292022705), ('Station', 2.6722395420074463), ('Restaurant', 0.5430010557174683)]
編集結果, relation LearnedLanguages: [('French', 11.888039588928223), ('English', 9.368602752685547), ('German', 7.896204948425293), ('Japanese', 7.5884175300598145), ('Spanish', 6.607028007507324)]
最終状態での評価
編集結果, relation ReadBooks: [('The Little Prince', 8.520033836364746), ('The Lord of the Rings', 7.3563995361328125), ('The Chronicles of Narnia', 5.858844757080078), ('A Bear Called Paddington', 5.411639213562012), ('Harry Potter', 3.0820083618164062)]
編集結果, relation Health Status: [('Healthy', 3.3736624717712402), ('Injured', 3.0216174125671387), ('Hypertensive', 2.724630832672119), ('Allergic', 0.9866428971290588), ('Diabetic', 0.06968644261360168)]
編集結果, relation VisitedPlaces: [('Paris', 7.258829593658447), ('New York', 4.496308326721191), ('Seoul', 3.865238666534424), ('Cairo', 2.9659714698791504), ('Rome', 2.828166961669922)]
編集結果, relation CurrentLocation: [('Home', 5.133467674255371), ('Library', 0.9373608827590942), ('Restaurant', 0.2104603499174118), ('Office', -0.08694514632225037), ('Station', -2.8366646766662598)]
編集結果, relation LearnedLanguages: [('French', 11.888039588928223), ('English', 9.368602752685547), ('German', 7.896204948425293), ('Japanese', 7.5884175300598145), ('Spanish', 6.607028007507324)]
条件(c): 共有型 (Shared Relations)
各ステップごとの評価
編集結果, relation VisitedPlaces: [('Paris', 5.8506598472595215), ('Cairo', 4.674541473388672), ('New York', 4.043600559234619), ('Rome', 3.825284004211426), ('Seoul', 2.437058925628662)]
編集結果, relation VisitedPlaces: [('New York', 8.634759902954102), ('Rome', 5.567094802856445), ('Paris', 5.02165412902832), ('Cairo', 3.7679314613342285), ('Seoul', 3.208742618560791)]
編集結果, relation VisitedPlaces: [('Seoul', 6.16569709777832), ('New York', 5.8932647705078125), ('Rome', 3.494523525238037), ('Cairo', 0.873633861541748), ('Paris', -1.4713504314422607)]
編集結果, relation VisitedPlaces: [('Paris', 14.486228942871094), ('New York', 5.000051021575928), ('Rome', 3.2826602458953857), ('Cairo', 2.819988250732422), ('Seoul', 2.6584157943725586)]
編集結果, relation VisitedPlaces: [('Rome', 4.2503485679626465), ('New York', 4.099995136260986), ('Paris', 2.716451406478882), ('Cairo', 1.7249855995178223), ('Seoul', 0.6867242455482483)]
最終状態での評価
編集結果, relation VisitedPlaces: [('Rome', 4.2503485679626465), ('New York', 4.099995136260986), ('Paris', 2.716451406478882), ('Cairo', 1.7249855995178223), ('Seoul', 0.6867242455482483)]
編集結果, relation VisitedPlaces: [('Rome', 4.2503485679626465), ('New York', 4.099995136260986), ('Paris', 2.716451406478882), ('Cairo', 1.7249855995178223), ('Seoul', 0.6867242455482483)]
編集結果, relation VisitedPlaces: [('Rome', 4.2503485679626465), ('New York', 4.099995136260986), ('Paris', 2.716451406478882), ('Cairo', 1.7249855995178223), ('Seoul', 0.6867242455482483)]
編集結果, relation VisitedPlaces: [('Rome', 4.2503485679626465), ('New York', 4.099995136260986), ('Paris', 2.716451406478882), ('Cairo', 1.7249855995178223), ('Seoul', 0.6867242455482483)]
編集結果, relation VisitedPlaces: [('Rome', 4.2503485679626465), ('New York', 4.099995136260986), ('Paris', 2.716451406478882), ('Cairo', 1.7249855995178223), ('Seoul', 0.6867242455482483)]
条件(c): 排他型 (Exclusive Relations)
各ステップごとの評価
編集結果, relation Health Status: [('Healthy', 4.750846862792969), ('Hypertensive', 3.615534543991089), ('Injured', 2.963477611541748), ('Diabetic', 2.9258780479431152), ('Allergic', 1.659001111984253)]
編集結果, relation Health Status: [('Healthy', 3.955571174621582), ('Allergic', 3.121216297149658), ('Injured', 3.0101583003997803), ('Diabetic', 1.5399974584579468), ('Hypertensive', 0.57059246301651)]
編集結果, relation Health Status: [('Diabetic', 8.59367561340332), ('Healthy', 5.672721862792969), ('Injured', 4.4806318283081055), ('Hypertensive', 3.1591594219207764), ('Allergic', 3.081411123275757)]
編集結果, relation Health Status: [('Healthy', 8.363020896911621), ('Injured', 4.166213512420654), ('Allergic', 3.999351978302002), ('Diabetic', 3.45697021484375), ('Hypertensive', 2.133011817932129)]
編集結果, relation Health Status: [('Injured', 5.7085113525390625), ('Healthy', 4.665274620056152), ('Diabetic', 1.5601634979248047), ('Hypertensive', 1.1214247941970825), ('Allergic', 1.057177186012268)]
最終状態での評価
編集結果, relation Health Status: [('Injured', 5.7085113525390625), ('Healthy', 4.665274620056152), ('Diabetic', 1.5601634979248047), ('Hypertensive', 1.1214247941970825), ('Allergic', 1.057177186012268)]
編集結果, relation Health Status: [('Injured', 5.7085113525390625), ('Healthy', 4.665274620056152), ('Diabetic', 1.5601634979248047), ('Hypertensive', 1.1214247941970825), ('Allergic', 1.057177186012268)]
編集結果, relation Health Status: [('Injured', 5.7085113525390625), ('Healthy', 4.665274620056152), ('Diabetic', 1.5601634979248047), ('Hypertensive', 1.1214247941970825), ('Allergic', 1.057177186012268)]
編集結果, relation Health Status: [('Injured', 5.7085113525390625), ('Healthy', 4.665274620056152), ('Diabetic', 1.5601634979248047), ('Hypertensive', 1.1214247941970825), ('Allergic', 1.057177186012268)]
編集結果, relation Health Status: [('Injured', 5.7085113525390625), ('Healthy', 4.665274620056152), ('Diabetic', 1.5601634979248047), ('Hypertensive', 1.1214247941970825), ('Allergic', 1.057177186012268)]
"""

# 正規表現パターンの定義
pattern = re.compile(r"編集結果, relation ([\w\s]+): \[\((.*?)\)\]")

# データを行ごとに分割
lines = data.strip().split('\n')

# 評価ステップをリストとして保持（順序を維持）
evaluations_ordered = []

for line in lines:
    line = line.strip()
    if not line or line.startswith("条件") or line.startswith("各ステップごとの評価") or line.startswith("最終状態での評価"):
        # 条件や評価のセクションをスキップ
        continue
    match = pattern.match(line)
    if match:
        relation = match.group(1).strip()
        tuples_str = "(" + match.group(2) + ")"
        try:
            # 使用ast.literal_evalでタプルリストを解析
            tuples = ast.literal_eval(f"[{tuples_str}]")
            # 名前とロジットを分ける
            items = [(name, float(logit)) for name, logit in tuples]
            evaluations_ordered.append((relation, items))
        except Exception as e:
            print(f"Error parsing tuples for relation {relation}: {e}")
            continue

# 各評価ステップ内のロジットをソフトマックスで確率に変換し、順序を維持して出力
for idx, (relation, items) in enumerate(evaluations_ordered, 1):
    names, logits = zip(*items)
    probs = softmax(logits)
    prob_dict = OrderedDict(zip(names, probs))
    print(f"Step {idx}: Relation: {relation}")
    for name, prob in prob_dict.items():
        print(f"  {name}: {prob:.4f}")
    print()