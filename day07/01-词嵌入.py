import jieba
import torch
from torch import nn

# 数据集（文本）
text = "北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。"

# 分词
word_list = jieba.lcut(text)
print(word_list)

# 去重
word_list = list(set(word_list))
print(word_list)

# 构建词表 （词和id索引）-> 字典和列表
word2id = {word: i for i, word in enumerate(word_list)}
print(word2id)

# 数量
num_words = len(word_list)
print(num_words)

# 词嵌入
embedding = di
print(embedding)

# 输入
# input_ids = [word2id[word] for word in word_list]
# input_ids = torch.tensor(input_ids)
# print(embedding(input_ids))
for key,value in word2id.items():
    print(key,value)
    print(embedding(torch.tensor([value])))

