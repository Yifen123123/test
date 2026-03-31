import os

def merge_lr(input_path, output_path):
    l_lines = []
    r_lines = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('L:'):
                content = line[2:].strip()
                if content:
                    l_lines.append(content)
            elif line.startswith('R:'):
                content = line[2:].strip()
                if content:
                    r_lines.append(content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('L:\n')
        f.write('\n'.join(l_lines))
        f.write('\n\nR:\n')
        f.write('\n'.join(r_lines))

# 資料夾路徑
input_folder = 'input_txts'
output_folder = 'output_txts'

# 確保 output 資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 讀取 input 資料夾內所有 .txt 檔
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)  # 同檔名存到 output
        merge_lr(input_path, output_path)
        print(f"完成：{filename}")

print("全部處理完畢！")
