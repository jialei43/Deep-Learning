import os

import jieba
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"当前设备：{device}")
def create_vocab():
    """
    创建词汇表

    读取周杰伦歌词文件，使用 jieba 分词构建词汇表，生成词到 ID 的映射关系。
    该函数不接收参数，返回词汇表相关信息供后续模型训练使用。

    :return: tuple，包含以下四个元素：
        - word2id (dict): 词到 ID 的映射字典，key 为词语，value 为对应的索引 ID
        - id2word (list): ID 到词的列表，索引位置即为对应 ID
        - vocab_size (int): 词汇表大小，即唯一词语的总数
        - corpus_id (list): 语料库的 ID 序列（当前实现中存在逻辑错误）
    """
    # 初始化存储所有歌词分词结果的列表
    all_words = []
    # 初始化存储唯一词语（去重后）的列表
    unique_words = []

    # 打开歌词文件，使用 UTF-8 编码读取
    with open('./data/jaychou_lyrics.txt', 'r', encoding='utf-8') as f:
        # 循环读取文件内容直到文件结束
        while True:
            # 逐行读取文件内容
            line = f.readline()
            # 如果读取行为空（到达文件末尾），则跳出循环
            if len(line) == 0:
                break
            # 使用 jieba 对每行歌词进行分词处理
            words = jieba.lcut(line.strip())
            # 将分词结果添加到所有歌词列表中
            all_words.append(words)
            # 遍历当前行的所有词语
            for word in words:
                # 如果该词语不在唯一词语列表中（去重）
                if word not in unique_words:
                    # 将词语添加到唯一词语列表
                    unique_words.append(word)

    # ID 到词的映射：直接使用唯一词语列表，索引即为 ID
    id2word = unique_words
    # 词到 ID 的映射：使用字典推导式，enumerate 生成索引作为 ID
    word2id = {unique_word: i for i, unique_word in enumerate(unique_words)}
    # 计算词汇表大小（唯一词语的数量）
    vocab_size = len(unique_words)
    # 初始化语料库 ID 列表（用于存储转换后的 ID 序列）
    corpus_id = []

    # 遍历所有歌词的分词结果，转换为 ID 序列
    for words in all_words:
        # 初始化当前歌词的 ID 序列
        ids = []
        # 遍历所有歌词（注意：此处应为遍历 words，当前代码存在逻辑错误）
        for word in words:
            # 从词到 ID 映射表中获取词语对应的 ID 并添加
            ids.append(word2id[word])
            # 添加空格标记的 ID（用于句子分隔或填充）
        ids.append(word2id[' '])
        # 将当前歌词的 ID 序列添加到语料库中
        corpus_id.extend(ids)

    # print(corpus_id)

    # 返回词到 ID 映射、ID 到词列表、词汇表大小和语料库 ID 序列
    return word2id, id2word, vocab_size, corpus_id


# 构建数据集
class WordDataset(Dataset):
    """
    歌词数据集类

    继承自 torch.utils.data.Dataset，用于加载歌词序列数据并生成训练样本。
    该类将语料库 ID 序列按指定长度切分为输入 - 输出对，用于语言模型训练。

    :param corpus_id (list): 语料库的 ID 序列，包含所有歌词转换后的整数 ID
    :param seq_len (int): 序列长度，即每个样本包含的时间步数(每个样本的词汇数量)
    :return: 无返回值，返回 Dataset 对象本身
    """

    def __init__(self, corpus_id, seq_len):
        # 调用父类 Dataset 的初始化方法
        super(WordDataset, self).__init__()
        # 保存语料库 ID 序列
        self.corpus_id = corpus_id
        # 保存序列长度参数(每个样本的词汇数量)
        self.seq_len = seq_len
        # 计算可生成的样本数量（整除）
        self.sample_num = len(self.corpus_id) // self.seq_len

    def __len__(self):
        # 返回数据集的总样本数
        return self.sample_num

    def __getitem__(self, id):
        # 计算安全的起始索引，确保不越界且在有效范围内
        start = min(max(id, 0), len(self.corpus_id) - self.seq_len - 2)
        # 截取输入序列：从 start 开始，长度为 seq_len
        x = self.corpus_id[start:start + self.seq_len]
        # 截取目标序列：从 start+1 开始，长度为 seq_len（向后偏移一位作为预测目标）
        y = self.corpus_id[start + 1:start + self.seq_len + 1]
        # 转换为 PyTorch 张量并返回
        return torch.tensor(x), torch.tensor(y)


# 构建模型
class WordModel(nn.Module):
    # 1.定义__init__方法，构造网络层
    def __init__(self, vocab_size):
        # 1.初始化父类成员
        super().__init__()
        # 2.定义网络层
        # 2.1 词嵌入层 生成向量表示
        self.embedding = nn.Embedding(vocab_size, 128)
        # 2.2 RNN层
        self.rnn = nn.RNN(128, 256, num_layers=1, batch_first=True)
        # 2.3 全连接层
        self.linear1 = nn.Linear(256, vocab_size)

    # 2.定义forward方法，前向传播
    def forward(self, x, hidden=None):
        # x:输入的索引序列，shape: (batch_size, seq_len)
        # 数据形状变化：BS - BSE - BSH - B*S,H - B*S,V
        # 1.经过词嵌入层
        # print(f"输入x.shape: {x.shape}")
        x = self.embedding(x)
        # print(f"词嵌入层输出x.shape: {x.shape}")
        # 2.经过RNN层
        output, hidden = self.rnn(x, hidden)
        # print(f"RNN层输出output.shape: {output.shape}")
        # print(f"RNN层输出hidden.shape: {hidden.shape}")
        # output: BSH
        # 3.经过全连接层
        # output: BSH - B*S,H  全连接需要二维数据
        # -1：B*S*H/H
        output = output.reshape(-1, output.shape[-1])
        # print(f"全连接层输入output.shape: {output.shape}")
        logits = self.linear1(output)  # logits: B*S,V
        # print(f"全连接层输出logits.shape: {logits.shape}")
        return logits, hidden

    def init_hidden(self, batch_size):
        # 创建初始隐藏状态
        # 1.创建一个全零张量，形状为 [1, batch_size, 256] # num_layer,bs,hidden_size
        return torch.zeros(1, batch_size, 256)

# 模型训练
def train(model, dataset):
    # 1.定义超参数
    epochs = 100 #
    lr = 0.01 #
    batch_size = 32 #
    weight_decay = 0.0001 # 权重衰减
    # 设备迁移
    model.to(device)
    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 训练循环
    for epoch in range(epochs):
        # 0.设置为训练模式
        model.train()
        # 1.初始化训练的总损失，预测正确的数量，总的样本数
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_tokens = 0
        for batch_id, (x, y) in enumerate(data_loader):
            # x:BS,(batch_size,seq_len)
            # y:BS,(batch_size,seq_len)
            # 0.迁移数据到设备
            x = x.to(device)
            y = y.to(device)
            # 1.前向传播，获取模型的预测结果，logits
            logits, hidden = model(x)
            # print(f"logits.shape: {logits.shape}")
            # print(f"y: {y.shape}")
            # print(f"y.reshape(-1): {y.reshape(-1).shape}")
            # 2.计算损失
            # y:(batch_size,seq_len)2D 转换为 1D(batch_size*seq_len,)
            loss = loss_fn(logits, y.reshape(-1))
            # loss = sum()/(batch_size*seq_len)
            # sum() = loss*(batch_size*seq_len)
            # 3.梯度清零
            optimizer.zero_grad()
            # 4.反向传播
            loss.backward()
            # 5.更新参数
            optimizer.step()
            # 6.更新 训练指标, 总损失，总预测正确的数量，总样本数
            total_loss += loss.item()
            # 2D-1D, B*S
            total_correct += (logits.argmax(dim=-1) == y.reshape(-1)).sum().item()
            total_samples += x.shape[0]
            total_tokens += x.shape[0] * x.shape[1]
            ...
            # 3.打印训练结果
        print(f"epoch: {epoch + 1}/{epochs} | "
              f"train_loss: {total_loss / total_tokens:.4f}, "
              f"train_acc: {total_correct / total_tokens:.4f}")

    # 4.保存模型
    os.makedirs(r"./model", exist_ok=True)
    torch.save(model.state_dict(), r"./model/my_rnn.pth")

# 5.模型预测
@torch.no_grad()  # 关闭梯度计算
def predict(vocab, start_token, sequence_length=40):
    # 1.加载模型
    model = WordModel(len(vocab))
    model.load_state_dict(torch.load("./model/my_rnn.pth"))
    # 迁移模型到 device
    model.to(device)
    # 切换为评估模式，禁用dropout，一些特殊网络层的处理逻辑在两种模式 model.train()和model.eval()下是不同的
    model.eval()
    # 2.获取隐藏状态初始值
    hidden = model.init_hidden(batch_size=1).to(device)
    # 3.开始预测
    # 获取输入的token对应的索引
    start_index = vocab[start_token]
    # 初始化 词索引列表的结果
    result_sequence = [start_index]
    # 初始化 当前时间步的 词索引
    current_index = start_index
    print(f"current_index: {current_index}, len: {len(result_sequence)}")
    for i in range(sequence_length):
        # 前向传播/模型预测
        # 输入需要是形状 [batch_size, sequence_length]，这里为 [1,1]
        y_pred, hidden = model(torch.tensor([[current_index]]).to(device), hidden)
        # 获取预测结果
        y_pred = y_pred.argmax(dim=-1).item()
        # 更新 词索引列表的结果
        result_sequence.append(y_pred)
        # 更新 当前时间步的 词索引，继续预测下一个词
        current_index = y_pred

    # 4.将 词索引列表的结果 转换为 文本，并打印
    for index in result_sequence:
        # 从vocab中 获取 index 对应的 词
        word = [k for k, v in vocab.items() if v == index][0]
        print(word, end="")



if __name__ == '__main__':
    word2id, id2word, vocab_size, corpus_id = create_vocab()
    # print(WordDataset(corpus_id, 10).__getitem__(0))
    # 5代表每个样本的词汇数量
    dataset = WordDataset(corpus_id, 5)
    # # print(vocab_size)
    # x, y = dataset.__getitem__(0)
    # print(f'x:{x}')
    # print(f'y:{y}')
    model = WordModel(vocab_size)
    print(model)
    train(model, dataset)
    # 5.模型预测
    predict(word2id, start_token="周杰伦", sequence_length=200)
