import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import strftime
import os

# Sklearn 库
from sklearn.model_selection import StratifiedKFold, train_test_split # <--- 修改1: 引入 StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# PyTorch & Geometric
from torch_geometric.loader import DataLoader 
from torch.utils.data import Subset 
from dataset import DDIDataset
from model import D1Model
from utils import *

def train_eval(config):
    if config.model_name == 'biosnap':
        dnames_new = ["label", "smile1", "smile2"]
    elif config.model_name == 'drugbank':
        dnames_new = ["label", "smile1", "smile2"]

    print(f"Preparing data for {config.model_name}...")
    
    # === 1. 数据准备 ===
    # 读取完整数据集
    full_dataset = DDIDataset(root=config.train_root, path=config.train_path)
    
    device = torch.device(config.device)
    criterion = nn.BCELoss()

    # Logging 根目录
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    saved_root_path = config.saved_root + current_time + '/'
    if not os.path.exists(saved_root_path):
        os.makedirs(saved_root_path)
        print(f"Directory {saved_root_path} Created")

    # === 2. 准备分层五折 (Stratified 5-Fold) ===
    # 为了做分层，我们需要先提取出所有数据的 label
    print("Extracting labels for stratification...")
    # 假设 dataset[i].y 是一个 tensor scalar，或者 dataset.data.y 存在
    # 这里用一种通用的方式提取标签列表，确保适用于 PyG Dataset
    try:
        # 尝试直接从属性读取 (速度快)
        all_labels = full_dataset.data.y.cpu().numpy()
    except:
        # 如果不支持，则遍历 (速度稍慢但通用)
        all_labels = np.array([data.y.item() for data in full_dataset])

    # 使用 StratifiedKFold 替代 KFold
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = {'auc': [], 'pr': [], 'f1': [], 'acc': []}

    # === 3. 五折循环开始 ===
    # skf.split 需要传入 X 和 y，这里 X 用索引占位，y 用于分层
    all_indices = np.zeros(len(full_dataset)) # 仅占位用
    
    # split 返回 (train_total_idx, test_idx)，确保 Test Set 是分层的
    for fold, (train_total_idx, test_idx) in enumerate(skf.split(all_indices, all_labels)):
        
        saved_path = saved_root_path + f'fold_{fold+1}/'
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        
        print(f"\n[Stratified Five-Fold] >>> Running Fold {fold+1} / {k_folds} <<<")
        print("-----Training-----") 
        
        # --- A. 内层分层切分 (Stratified Train/Val Split) ---
        # 1. 获取当前 80% 数据的标签，用于内层分层
        train_total_labels = all_labels[train_total_idx]
        
        # 2. 切分 Train/Val，加入 stratify 参数
        # 这样 Val Set 的正负比就和 Train Set 完全一致了
        train_idx, val_idx = train_test_split(
            train_total_idx, 
            test_size=0.1, 
            random_state=42, 
            stratify=train_total_labels # <--- 修改2: 关键点，确保验证集分布稳定
        )
        
        # 构建 Subset
        train_subset = Subset(full_dataset, train_idx)
        valid_subset = Subset(full_dataset, val_idx)
        test_subset  = Subset(full_dataset, test_idx)

        # 创建 DataLoader
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, drop_last=True, follow_batch=['pos1', 'pos2'])
        valid_loader = DataLoader(valid_subset, batch_size=config.batch_size, shuffle=False, drop_last=False, follow_batch=['pos1', 'pos2'])
        test_loader  = DataLoader(test_subset,  batch_size=config.batch_size, shuffle=False, drop_last=False, follow_batch=['pos1', 'pos2'])
        
        # --- B. 模型与优化器初始化 ---
        model = D1Model(config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=config.gamma)
        
        best_val_metric = -float('inf') 
        best_epoch = -1
        best_ckpt_path = saved_path + f"{config.model_name}_best.pt"
        
        train_losses = []
        valid_losses = []
        
        starttime = datetime.datetime.now()

        # === 4. 训练循环 ===
        for epoch in range(config.epochs):
            model.train()
            batch_losses = []
            
            for batch_data in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch}"):
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                
                outputs, _ = model(batch_data)
                labels = batch_data.y.float().view(-1, 1)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())

            scheduler.step()
            epoch_train_loss = np.mean(batch_losses)
            train_losses.append(epoch_train_loss)
            
            # === 验证 ===
            label_valid, pred_valid, prob_valid, val_avg_loss = get_label_pred_ddi_binary(model, valid_loader, criterion)
            valid_losses.append(val_avg_loss)
            
            roc_auc_score_val = metrics.roc_auc_score(label_valid, prob_valid)
            pr_auc_score_val = metrics.average_precision_score(label_valid, prob_valid)
            f1_score_val = sk_p_r(label_valid, pred_valid)  
            acc_score_val = metrics.accuracy_score(label_valid, pred_valid)

            # === 保存最佳模型 ===
            if roc_auc_score_val > best_val_metric:
                best_val_metric = roc_auc_score_val
                best_epoch = epoch
                torch.save(model.state_dict(), best_ckpt_path)
                get_logging(
                    f"[BEST] epoch={epoch} val_roc_auc={roc_auc_score_val:.6f} -> saved",
                    saved_path + 'log.txt'
                )

            cls_report_val = metrics.classification_report(label_valid, pred_valid, target_names=['class 0', 'class 1'], zero_division=0)

            # === 终端显示 ===
            print(f"== == training phrase == == ")
            print(f"epoch:{epoch}  train_loss:{epoch_train_loss}")
            print(f"valid_loss:{val_avg_loss}")
            print(f"roc_auc:{roc_auc_score_val}")
            print(f"pr_auc:{pr_auc_score_val}")
            print(f"macro f1: {f1_score_val}")
            print(f"acc:{acc_score_val}")
            print(cls_report_val)

            # === Log 写入 ===
            log = f"== == training phrase == == \n" + \
                  f"epoch:{epoch}  train_loss:{epoch_train_loss}\n" + \
                  f"valid_loss:{val_avg_loss}\n" + \
                  f"roc_auc:{roc_auc_score_val}\n" + \
                  f"pr_auc:{pr_auc_score_val}\n" + \
                  f"macro f1: {f1_score_val}\n" + \
                  f"acc:{acc_score_val}\n" + \
                  cls_report_val + "\n"
            get_logging(log, saved_path + 'log.txt')
            
            endtime = datetime.datetime.now()
            print(f"total run time:  {endtime - starttime}")

        # === 本折测试阶段 ===
        
        draw_loss_curve(train_losses, valid_losses, saved_path)

        if os.path.exists(best_ckpt_path):
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        else:
            get_logging(f"[WARN] No best model found, using last epoch.", saved_path + 'log.txt')

        torch.save(model.state_dict(), saved_path + config.model_name + '.pt')
        get_logging(f"[INFO] Fold {fold+1} Finished. Testing on Strict Test Set.", saved_path + 'log.txt')

        # === Test Phase ===
        label_test, pred_test, prob_test, _ = get_label_pred_ddi_binary(model, test_loader, criterion)
        roc_auc_score_test = metrics.roc_auc_score(label_test, prob_test)
        pr_auc_score_test = metrics.average_precision_score(label_test, prob_test)
        
        draw_roc_pr_curve(label_test, prob_test, saved_path)

        f1_test = sk_p_r(label_test, pred_test) 
        acc_test = metrics.accuracy_score(label_test, pred_test)
        cls_report_test = metrics.classification_report(label_test, pred_test, target_names=['class 0', 'class 1'], zero_division=0)

        print("== == test phrase == == ")
        print('test dataset roc_auc:', roc_auc_score_test)
        print('test dataset pr_auc:', pr_auc_score_test)
        print('test dataset macro f1:', f1_test)
        print('test dataset acc:', acc_test)
        print(cls_report_test)

        log = "== == test phrase == == \n" + \
              f"test dataset roc_auc:{roc_auc_score_test}\n" + \
              f"test dataset pr_auc:{pr_auc_score_test}\n" + \
              f"test dataset macro f1: {f1_test}\n" + \
              f"test dataset acc:{acc_test}\n" + \
              cls_report_test + "\n"
        get_logging(log, saved_path + 'log.txt')

        cf_matrix = confusion_matrix(label_test, pred_test)
        draw_confusion_matrix(config, cf_matrix, saved_path)
        
        fold_results['auc'].append(roc_auc_score_test)
        fold_results['pr'].append(pr_auc_score_test)
        fold_results['f1'].append(f1_test)
        fold_results['acc'].append(acc_test)

    # === 5. 最终汇总 ===
    print("\n" + "="*20 + " 5-Fold Final Report " + "="*20)
    for metric in fold_results:
        mean_val = np.mean(fold_results[metric])
        std_val = np.std(fold_results[metric])
        print(f"Average {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    print("="*60 + "\n")
    
    df_res = pd.DataFrame(fold_results)
    df_res.loc['mean'] = df_res.mean()
    df_res.loc['std'] = df_res.std()
    df_res.to_csv(saved_root_path + '5fold_final_results.csv')