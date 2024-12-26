from ollama import chat
from ollama import ChatResponse
import json
import pandas as pd
from tqdm import tqdm

PROMPT = """
您的任务是分析这篇文章反映了市场什么情绪？中性，正向，负向。除输出情绪外，无需输出或生成其他内容：
"""

def call_llm(text, PROMPT):
  response: ChatResponse = chat(model='qwen2.5:7b', 
  messages=[
    {'role': 'system', 'content': PROMPT},
    {
      "role": "user",
      'content': f"这是新闻内容：{text}",
    },
  ],
  options={"stop": ["\n", "这", "该"]})
  res = response['message']['content']
  return res


# 读取CSV文件
file_path = r'2019.csv'
output_file_prefix = 'results_batch_'  # 输出文件前缀
final_file = '2024_with_sentiment_rg_7b.xlsx'
batch_size = 5000  # 每多少条记录保存一次

# 读个CSV文件
df = pd.read_csv(file_path)
# 使用更高效的方式移除问号
df['NewsContent'] = df['NewsContent'].str.replace('?', '', regex=False)
df['Title'] = df['Title'].str.replace('?', '', regex=False)

# 找出所有NewsContent超过4000字符的行，并用Title替换
mask = df['NewsContent'].str.len() > 5000
df.loc[mask, 'NewsContent'] = df.loc[mask, 'Title']


# 初始化进度条
pbar = tqdm(total=len(df), desc="Processing")

# 遍历DataFrame中的每一行，并进行情绪分析
for i, row in df.iterrows():
    text = row['NewsContent']
    result = call_llm(text, PROMPT)
    
    # 更新DataFrame中的情绪列
    df.loc[i, '情绪'] = result
    
    # 更新进度条
    pbar.update(1)
    
    # 每batch_size次或到达末尾时保存进度
    if (i + 1) % batch_size == 0 or (i + 1) == len(df):
        # 创建一个临时的DataFrame来保存当前批次的结果
        start_index = max(0, i + 1 - batch_size)
        current_results_df = df.iloc[start_index:i+1].copy()  # 使用iloc获取包含原字段和情绪字段的子集
        
        # 保存为CSV文件
        output_file = f"{output_file_prefix}{start_index + 1}_to_{i + 1}.csv"
        current_results_df.to_csv(output_file, index_label='Index')
        print(f"Batch saved: {output_file}")

pbar.close()

# 完成所有处理后，将最终结果保存为Excel文件
df.to_excel(final_file, index=False)
print("All data processed and final results saved.")