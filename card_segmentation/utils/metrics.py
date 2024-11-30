import torch


def multi_acc(pred, label):
    _, tags = torch.max(pred, dim=1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc

def precision(output, ground_truth):
    smooth = 1e-5
    output = torch.sigmoid(output).data.cpu().numpy()
    output = output > 0.5
    ground_truth = ground_truth.data.cpu().numpy()
    return (output * ground_truth).sum() / (output.sum() + smooth)

def recall(output, ground_truth):
    smooth = 1e-5
    output = torch.sigmoid(output).data.cpu().numpy()
    output = output > 0.5
    ground_truth = ground_truth.data.cpu().numpy()
    return (output * ground_truth).sum() / (ground_truth.sum() + smooth)

def F1_score(output, ground_truth):
    pre = precision(output, ground_truth)
    rec = recall(output, ground_truth)
    return 2 * pre * rec / (pre + rec + 1e-5)

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)
