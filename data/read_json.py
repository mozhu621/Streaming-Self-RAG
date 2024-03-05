import json

# 替换为您的JSON文件路径
file_path = 'triviaqa_test_w_gs.jsonl'

# 打开JSON文件并加载数据
with open(file_path, 'r') as file:
    data = json.load(file)

# 假设JSON文件是一个数组结构，并打印第一项
if data and isinstance(data, list):
    print(data[0])
else:
    print("JSON文件不是预期的数组结构或为空")
