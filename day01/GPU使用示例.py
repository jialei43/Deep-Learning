import torch

# 检查 MPS 是否可用
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 创建一个示例张量并将其移动到 MPS 设备
tensor = torch.randn(3, 3).to(device)

print(f"type: {type(tensor)}")

# 创建一个简单的模型
model = torch.nn.Linear(3, 3).to(device)

# 进行前向计算
output = model(tensor)
print(output)
