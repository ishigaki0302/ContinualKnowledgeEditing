from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# CUDAが利用可能か確認し、デバイスを設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# トークナイザーとモデルのロード
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)  # モデルをGPUに移行
tokenizer.pad_token = tokenizer.eos_token

# 推論に使うテキスト
input_text = """Riku Nakamura is skilled in Python.

Select all the correct answers to the question from the given options and list the corresponding numbers.
###Question###
What skills does Riku Nakamura have?
###Sample Answer###
A: n, m, k
###Answer Options###
1. Python
2. SQL
3. Java
4. Excel
5. HTML
###Answer###
A: """

# テキストをトークナイズし、パディングとattention maskを作成
input_ids = tokenizer.encode(input_text, return_tensors="pt", padding=True, truncation=True).to(device)  # 入力をGPUに移行
attention_mask = torch.ones_like(input_ids).to(device)  # attention_maskをGPUに移行

# 推論を実行（最大50トークンの出力を生成）
output = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_return_sequences=1)

# トークンをテキストにデコード
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 結果を表示
print(output_text)