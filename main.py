from train import train_eval
from test import test
from model import D1Model

import torch
import datetime
import argparse

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
starttime = datetime.datetime.now()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', default='./datasets/BIOSNAP/biosnap_train/', type=str, required=False,
                        help='..')
    parser.add_argument('--train_path', default='train_biosnap_smiles_new.csv', type=str, required=False,
                        help='..')
    parser.add_argument('--val_path', type=str, help='validation dataset path')
    parser.add_argument('--test_root', default='./datasets/BIOSNAP/biosnap_test/', type=str, required=False,
                        help='..')
    parser.add_argument('--test_path', default='test_biosnap_smiles_new.csv', type=str, required=False,
                        help='..')
    parser.add_argument('--batch_size', default=8, type=int, required=False,
                        help='..')
    parser.add_argument('--epochs', default=30, type=int, required=False,
                        help='..')
    parser.add_argument('--lr', default=2e-5, type=float, required=False,
                        help='..')
    parser.add_argument('--weight_decay', default=1e-2, type=float, required=False,
                        help='..')
    parser.add_argument('--gamma', default=0.8, type=float, required=False,
                        help='..')
    parser.add_argument('--dropout', default=0.4, type=float, required=False,
                        help='..')
    parser.add_argument('--step_size', default=10, type=int, required=False,
                        help='..')
    parser.add_argument('--num_class', default=2, type=int, required=False,
                        help='..')
    parser.add_argument('--mode', default='train', type=str, required=False,
                        help='train or test')
    parser.add_argument('--shared', action='store_true', help="Enable sharing mode")
    parser.add_argument('--sequence_length', default=512, type=int, required=False,
                        help='..')
    parser.add_argument('--hidden_dim_1d', default=64, type=int, required=False,
                        help='..')
    parser.add_argument('--mid_dim', default=4, type=int, required=False,
                        help='..')
    parser.add_argument('--out_dim', default=1, type=int, required=False,
                        help='..')
    parser.add_argument('--n_gpu', default=4, type=int, required=False,
                        help='..')
    parser.add_argument('--n_cpu', default=128, type=int, required=False,
                        help='..')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str, required=False,
                        help='..')
    parser.add_argument('--model_name', default='biosnap', type=str, required=False,
                        help='..')
    parser.add_argument('--load_model_path', default='./trained_record/biosnap/2023-12-14-10:43:40/', type=str, required=False,
                        help='..')
    parser.add_argument('--saved_root', default='./trained_record/biosnap/', type=str, required=False,
                        help='..')
    parser.add_argument('--project_name', default='PTB-DDI', type=str, required=False,
                        help='..')
    
    # [修改核心参数]
    # 1. vocab_size: 设为 128 (覆盖所有 SMILES 字符)。原先的 60000 是给 BERT 用的。
    parser.add_argument('--vocab_size', default=128, type=int, help='Size of vocab (Character level)')
    
    # 2. embedding_dim: 设为 128 (字符嵌入常用的维度)。
    parser.add_argument('--embedding_dim', default=128, type=int, help='Dim of embedding') 

    args = parser.parse_args()
    if args.mode == 'train':
        train_eval(args)
    elif args.mode == 'test':
        test(args)
    else:
        print("We only have three mode: (1) train, (2) test. Please check your input again.\n")

    endtime = datetime.datetime.now()
    print('Prepocess run time: ', endtime - starttime)
    torch.cuda.empty_cache()