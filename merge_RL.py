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

    print(f"完成！已儲存至 {output_path}")
