import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision import transforms
from dataset import Dataset
from collections import OrderedDict
import time

# 图像预处理变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class ViTWithMLFF(nn.Module):
    def __init__(self, config):
        super(ViTWithMLFF, self).__init__()
        self.vit = ViTModel(config)

        # 定义1x1卷积，用于对不同尺度的特征进行通道数调整
        self.conv1 = nn.Conv2d(768, 256, kernel_size=1)  # 缩放1/4特征
        self.conv2 = nn.Conv2d(768, 256, kernel_size=1)  # 缩放1/2特征
        self.conv3 = nn.Conv2d(768, 256, kernel_size=1)  # 原始特征

        # 最终卷积层
        self.final_conv = nn.Conv2d(256, 1, kernel_size=1)  # 进行二分类

    def forward(self, input1, input2):
        # 从ViT提取出多个隐藏层的输出
        diff = torch.abs(input1 - input2)
        outputs = self.vit(diff, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # 提取所有层的隐藏状态

        # 假设我们选择第6层、第8层和最后一层作为多尺度特征
        feature1 = hidden_states[5][:, 1:, :].transpose(1, 2).view(-1, 768, 14, 14)  # 第6层特征
        feature2 = hidden_states[7][:, 1:, :].transpose(1, 2).view(-1, 768, 14, 14)  # 第8层特征
        feature3 = hidden_states[-1][:, 1:, :].transpose(1, 2).view(-1, 768, 14, 14)  # 最后一层特征

        # 对特征进行通道数调整
        feature1 = self.conv1(feature1)
        feature2 = self.conv2(feature2)
        feature3 = self.conv3(feature3)

        # 将多尺度特征进行融合
        combined_features = feature1 + feature2 + feature3

        # 最终分类卷积
        output = self.final_conv(combined_features)
        output = torch.sigmoid(output.mean([2, 3]))  # 将空间维度上的结果平均，进行二分类

        return output


def frame_difference(prev_frame, curr_frame, threshold):
    diff = torch.abs(prev_frame - curr_frame).sum(dim=(1, 2, 3))  # 直接在GPU上计算差异
    return diff > threshold


def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    loss_fn = nn.BCELoss()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data1, data2, target) in enumerate(train_loader):
        data1, data2, target = data1.to(device), data2.to(device), target.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        predictions = model(data1, data2)
        loss = loss_fn(predictions, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data1.size(0)  # 累加损失
        predicted_labels = (predictions > 0.5).float()
        correct += (predicted_labels == target).sum().item()
        total += target.size(0)

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data1)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_train_loss = train_loss / total
    train_accuracy = 100. * correct / total
    return avg_train_loss, train_accuracy


def test(model, device, test_loader, threshold=0.0159):
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    loss_fn = nn.BCELoss()

    # 初始化计数器
    true_positives = 0  # 实际为0，预测也为0
    false_negatives = 0  # 实际为0，预测为1（漏报）
    false_positives = 0  # 实际为1，预测为0（误报）
    total_predicted_zeros = 0  # 模型预测为0的样本总数
    total_ground_truth_zeros = 0  # 实际为0的样本总数

    # 开始总时间计时
    start_time_total = time.time()
    total_inference_time = 0  # 推理时间累积

    with torch.no_grad():
        for data1, data2, target in test_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device).float()

            # 预设所有预测值为1（负类）
            predicted_labels = torch.ones_like(target).float()

            # 开始推理时间计时
            start_time_inference = time.time()

            # 计算帧差异
            diff_results = frame_difference(data1, data2, threshold)

            # 对于超过阈值的帧进行模型推理
            if diff_results.any():
                predictions = model(data1[diff_results], data2[diff_results])
                predicted_labels[diff_results] = (predictions > 0.5).float().squeeze()

            # 推理时间结束
            end_time_inference = time.time()
            total_inference_time += end_time_inference - start_time_inference

            # 计算准确性
            correct += (predicted_labels == target).sum().item()

            # 统计正类（实际为0，预测为0）
            true_positives += ((predicted_labels == 0) & (target == 0)).sum().item()
            # 统计漏报（实际为0，预测为1）
            false_negatives += ((predicted_labels == 1) & (target == 0)).sum().item()
            # 统计误报（实际为1，预测为0）
            false_positives += ((predicted_labels == 0) & (target == 1)).sum().item()

            # 统计总数
            total_predicted_zeros += (predicted_labels == 0).sum().item()
            total_ground_truth_zeros += (target == 0).sum().item()

            total += target.size(0)

    # 计算帧率 FPS
    fps = total / total_inference_time if total_inference_time > 0 else 0

    # 计算精确度（模型输出0的比例中，实际为0的比例）
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    # 计算召回率（所有实际为0的样本中，预测为0的比例）
    recall = true_positives / total_ground_truth_zeros if total_ground_truth_zeros > 0 else 0

    accuracy = 100 * correct / total if total > 0 else 0

    # 输出评估结果
    print(f'\nTest set: Accuracy: {correct}/{total} ({accuracy:.2f}%), Precision: {precision:.2f}, Recall: {recall:.2f}, FPS: {fps:.2f}\n')

    return accuracy, precision, recall, fps


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_pretrained_weights(model, path):
    model.load_state_dict(torch.load(path))


def run_training(train_loader, test_loader, device, model, optimizer, num_epochs):
    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        avg_train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
        accuracy, precision, recall, fps = test_with_speed(model, device, test_loader)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, "best_difference_vit_mlff_model_test.pth")
            print(f"New best model saved with accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型配置和权重
    config_path = "vit_weight/config.json"
    weights_path = "vit_weight/pytorch_model.bin"

    config = ViTConfig.from_pretrained(config_path)
    model = ViTWithMLFF(config).to(device)

    # 加载模型权重，忽略分类头
    state_dict = torch.load(weights_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith("classifier"):
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=0.0007)

    # 加载数据集
    train_dataset = Dataset(r"transition_dataset/train.txt", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = Dataset(r"transition_dataset/test.txt", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 运行训练
    run_training(train_loader, test_loader, device, model, optimizer, num_epochs=30)

    # 测试最佳模型
    load_pretrained_weights(model, "best_difference_vit_mlff_model.pth")
    test(model, device, test_loader)
