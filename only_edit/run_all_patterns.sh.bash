#!/bin/bash

# メインパターンリストを定義
main_patterns=(
    "Skill"
    "Hobbies"
    "Awards"
    "Books Read"
    "Travel Destinations"
    "Health Status"
    "Job"
    "Residence"
    "Phone Plan"
    "Position"
)
# サブパターンの接尾辞
suffixes=("_pattern_1" "_pattern_2" "_pattern_3")

# 全てのメインパターンとサブパターンに対して eval_edit.py を実行
for pattern in "${main_patterns[@]}"; do
    for suffix in "${suffixes[@]}"; do
        full_pattern="${pattern}${suffix}"
        echo "Running eval_edit.py for pattern: $full_pattern"
        python eval_edit.py --pattern "$full_pattern"

        # 実行が失敗した場合はエラーメッセージを表示
        if [ $? -ne 0 ]; then
            echo "Error: Failed to execute eval_edit.py for pattern: $full_pattern"
            exit 1
        fi
    done
done

echo "All patterns processed successfully!"