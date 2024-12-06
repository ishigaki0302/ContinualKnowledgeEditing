from transformers import AutoModelForCausalLM, AutoTokenizer
from demo import demo_model_editing
import datetime

MODEL_NAME = "EleutherAI/gpt-j-6B"

model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
        "cuda:0"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
    # AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False),
)
tok.pad_token = tok.eos_token
print(model.config)

request = [
    {
        "prompt": "{} was the founder of",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
    }
]

generation_prompts = [
        "Shohei Ohtani is",
        "The team Shohei Ohtani plays for is",
        "Your favorite play by Shohei Ohtani is",
        "The most impressive record achieved by Shohei Ohtani is",
        "Shohei Ohtani's greatest strength is",
        "Shohei Ohtani's position is",
    ]

for i in range(1):
    # 現在の日時を取得
    now = datetime.datetime.now()
    # 日時を '年月日_時分秒' の形式でフォーマット
    formatted_date = now.strftime("%Y%m%d_%H%M%S")
    file_path = f"result/edit_output/{formatted_date}.txt"
    # Execute rewrite
    model_new, orig_weight, new_probs = demo_model_editing(
        model, tok, request, generation_prompts, file_path=file_path
    )