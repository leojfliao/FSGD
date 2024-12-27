import sys
import json
import pandas as pd
from tqdm import tqdm
from ollama import chat, ChatResponse

PROMPT = """
任务说明：

请分析下述股票文章的内容，并评估其反映的市场情绪。
指示：
极度悲观：市场预期非常负面，通常伴随着重大损失或严重衰退。
较为悲观：存在明显的担忧，但并非完全绝望；有对未来的不确定性。
谨慎乐观：尽管有一些积极因素，但仍保持警惕；对未来持有限度的信心。
乐观：表现出强烈的正面态度，预期未来会有较好的发展。
极度乐观：表现出极大的信心和兴奋，几乎看不到任何负面影响。
中性：没有明显的情绪倾向，或者正负情绪相互抵消。
注意：
输出时仅给出所选的情绪类别，不添加任何额外文字或解释。
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
                                  options={"stop": ["\n", "这", "该", "从", "本", "跟", "以"],
                                           "temperature": 0.1})
    res = response['message']['content']
    return res

def extract_segments(text, segment_length=800):
    """
    从给定的文本中提取三个片段：开头、中间和结尾。

    参数:
    text (str): 要分割的文本。
    segment_length (int): 每个段落的长度，默认为800个字符。

    返回:
    str: 组合后的文本，由三个段落组成。
    """
    if len(text) <= segment_length * 3:
        return text  # 如果文本长度不超过3000，则返回原文本

    start_segment = text[:segment_length]
    end_segment = text[-segment_length:]

    # 计算中间段落的起始位置，确保不会与前面或后面的段落重叠
    mid_start = (len(text) - segment_length) // 2
    mid_end = mid_start + segment_length
    middle_segment = text[mid_start:mid_end]

    # 合并三个段落
    combined_text = f"{start_segment} [MID] {middle_segment} [END] {end_segment}"
    return combined_text

def main(input_file):
    # 提取年份用于生成输出文件名
    year = input_file.split('.')[0]
    output_file_prefix = f'{year}_news_sentiment'
    final_file = f'{output_file_prefix}.csv'
    batch_size = 100  # 每多少条记录保存一次

    # 读取CSV文件并处理坏行
    bad_lines = []
    def handle_bad_lines(line):
        bad_lines.append(line)
        return None

    df = pd.read_csv(input_file, on_bad_lines=handle_bad_lines, engine='python', encoding_errors='ignore')

    # 保存坏行到CSV文件
    if bad_lines:
        bad_lines_df = pd.DataFrame(bad_lines, columns=df.columns)
        bad_lines_df.to_csv(f'{year}_badlines.csv', index=False)
        print(f"Bad lines saved to {year}_badlines.csv")

    # 应用函数到符合条件的行
    over_4000_count = (df['NewsContent'].str.len() > 2500).sum()
    print(f"超过4000字的行数: {over_4000_count}")
    # 找出所有NewsContent超过4000字符的行，并用提取的片段替换
    mask = df['NewsContent'].str.len() > 2500
    df.loc[mask, 'NewsContent'] = df.loc[mask, 'NewsContent'].apply(extract_segments)

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
            output_file = f"{output_file_prefix}_{start_index + 1}_to_{i + 1}.csv"
            current_results_df.to_csv(output_file, index_label='Index')
            print(f"Batch saved: {output_file}")

    pbar.close()

    # 完成所有处理后，将最终结果保存为CSV文件
    df.to_csv(final_file, index=False)
    print(f"All data processed and final results saved to {final_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python llm_test.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)
