import torch
import pandas as pd
import os
import torch.nn as nn
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn import metrics
import matplotlib.pyplot as plt

def read_file(path):
    rfile = pd.read_csv(path, header=None, names=None).reset_index(drop=True)
    return rfile

def process_data(df, columns_new):
    df_new = pd.DataFrame(df, columns=columns_new)
    df_new.replace({'FALSE': 0, 'TRUE': 1}, inplace=True)
    return df_new

def write_data(data, path, file_type = 'csv'):
    if file_type == 'csv':
        data.to_csv(path, encoding='utf-8', index=False, header=False, mode='x')

# === [核心修改] 增加数值稳定性保护，防止 NaN ===
def get_label_pred_ddi_binary(model, val_loader, criterion=None):
    # 如果没有传入 criterion，默认使用 BCELoss
    if criterion is None:
        criterion = nn.BCELoss()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred, label, prob = [], [], []
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            batch_data = batch_data.to(device)
            outputs, _ = model(batch_data) 
            
            # 确保维度匹配
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            if batch_data.y.dim() == 1:
                batch_data.y = batch_data.y.unsqueeze(1)
            
            # 将输出限制在 [1e-7, 0.9999999] 之间
            # 这不会影响模型学习，但能 100% 防止 log(0) 导致的 NaN 崩溃
            outputs = torch.clamp(outputs, min=1e-7, max=1.0 - 1e-7)

            # 计算 Loss
            valid_loss = criterion(outputs, batch_data.y)
            
            # 双重保险：检查是否有 NaN，如果有则跳过并打印警告（极罕见情况）
            if torch.isnan(valid_loss):
                print(f"Warning: NaN loss detected at batch {i}. Ignored.")
                continue

            total_loss += valid_loss.item()
            
            prob.extend(outputs.detach().cpu().numpy().flatten())
            # 预测值四舍五入
            pred.extend(torch.round(outputs).detach().cpu().numpy().flatten())
            label.extend(batch_data.y.detach().cpu().numpy().flatten())
    
    # 计算平均 Loss
    if len(val_loader) > 0:
        avg_loss = total_loss / len(val_loader)
    else:
        avg_loss = 0.0

    # 返回 4 个值：标签，预测，概率，平均损失
    return np.array(label), np.array(pred), np.array(prob), avg_loss


def sk_p_r(label, pred):
    """Standard macro-F1 (sklearn definition).

    This matches: sklearn.metrics.f1_score(y_true, y_pred, average='macro').

    Note: zero_division=0 is the standard safe setting (do not reward undefined cases).
    """
    return metrics.f1_score(np.array(label), np.array(pred), average='macro', zero_division=0)


def logging(log_file, logs):
    logfile = open(
        log_file, 'a+'
    )
    logfile.write(logs)
    logfile.close()

def get_logging(log, log_file):
    logging(log_file, log)

def draw_loss_curve(train_losses, valid_losses, path):
    plt.figure()
    plt.plot(train_losses, label='train_loss')
    plt.plot(valid_losses, label='valid_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss Curve')
    save_path = path + 'loss.png'
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    plt.close()

def draw_roc_pr_curve(label_test, prob_test, path):
    plt.figure()
    fpr, tpr, _ = metrics.roc_curve(label_test, prob_test)
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    save_path = path + 'ROC.png'
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    plt.close()

    plt.figure()
    precision, recall, _ = metrics.precision_recall_curve(label_test, prob_test)
    plt.plot(recall, precision, label='PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    save_path = path + 'PR.png'
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    plt.close()

def draw_confusion_matrix(config, cf_matrix, path):
    plt.figure()
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    save_path = path + 'Confusion Matrix.png'
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    plt.close()
