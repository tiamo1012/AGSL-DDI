from utils import read_file
import os
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch

# ================== 自定义 SMILES 字典 (Character-level) ==================
# 包含常见的 SMILES 字符。如果你的数据中有特殊元素(如Br, Cl等)，它们会被当作 pad 或需要添加到这里。
# 也可以改为动态构建字典，但固定字典更稳定。
SMILES_CHARLIST = [
    '<pad>', '<unk>', # 0, 1
    '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '=', '@', 'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'V',
    'Z', '[', '\\', ']', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u'
]

# 构建字符到整数索引的映射
CHAR_TO_INT = {c: i for i, c in enumerate(SMILES_CHARLIST)}
VOCAB_SIZE = len(SMILES_CHARLIST) # 这里的长度通常在 60 左右

def smiles_to_int(smiles, max_len=512):
    """
    将 SMILES 字符串转换为整数列表 (Padding & Truncation)
    """
    if not isinstance(smiles, str):
        smiles = ""
        
    # 1. 截断
    smiles = smiles[:max_len]
    
    # 2. 映射 (遇到不在字典里的字符，给 <unk> 索引，即 1)
    ids = [CHAR_TO_INT.get(char, CHAR_TO_INT['<unk>']) for char in smiles]
    
    # 3. 创建 Mask (1表示有数据，0表示padding)
    mask = [1] * len(ids) + [0] * (max_len - len(ids))
    
    # 4. Padding (补 0)
    ids = ids + [CHAR_TO_INT['<pad>']] * (max_len - len(ids))
    
    return ids, mask
# ==========================================================================

#-------------------------------------Process Drugbank dataset------------------------------
def process_drugbank_dataset():
    path_drugbank = "./datasets/DrugBank/ddi_total.csv"
    path_train = "./datasets/DrugBank/drugbank_train/raw/train_drugbank_smiles.csv"
    path_test = "./datasets/DrugBank/drugbank_test/raw/test_drugbank_smiles.csv"
    path_train_new = "./datasets/DrugBank/drugbank_train/raw/train_drugbank_smiles_new.csv"
    path_test_new = "./datasets/DrugBank/drugbank_test/raw/test_drugbank_smiles_new.csv"

    drugbank_df = pd.read_csv(path_drugbank, header=None, skiprows=1)
    train_df, test_df = train_test_split(drugbank_df, test_size=0.2, random_state=1234, shuffle=True)
    dnames = ['Drug1_ID', 'Drug1_SMILES', 'Drug2_ID', 'Drug2_SMILES', 'Label_Multi', 'label']
    train_df.to_csv(path_train, index=False, header=dnames)
    test_df.to_csv(path_test, index=False, header=dnames)

    # Extract desired columns
    train_extracted = train_df.iloc[:, [5, 1, 3]]
    test_extracted = test_df.iloc[:, [5, 1, 3]]

    # Rename columns
    dnames_new = ["label", "smile1", "smile2"]

    # Save extracted data to new CSV files
    train_extracted.to_csv(path_train_new, index=False, header=dnames_new)
    test_extracted.to_csv(path_test_new, index=False, header=dnames_new)

# process_drugbank_dataset()
#-----------------------------------------------------------------------------------------

class DDIDataset(InMemoryDataset):
    def __init__(self, root='/tmp', path='', transform=None, pre_transform=None):
        self.path = path
        # [修改] 不再加载 BERT Tokenizer
        
        # root is required for save preprocessed data, default is '/tmp'
        super(DDIDataset, self).__init__(root, transform, pre_transform)

        # self.processed_paths comes from processed_file_names property
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process_1D(root)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        # [修改] 更改文件名，强制重新生成处理后的数据 (避免读取旧的BERT数据)
        return ['process_1_char_embedding.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process_1D(self, root):
        df1 = pd.read_csv(root + 'raw/' + self.path)
        data_list = []
        data_len = len(df1)
        
        # 必须与 main.py 中的 sequence_length 一致
        MAX_LEN = 512 

        for i in range(data_len):
            if i % 1000 == 0:
                print('Encoding SMILES to Integers: {}/{}'.format(i + 1, data_len))

            smile1 = df1.loc[i, 'smile1']
            smile2 = df1.loc[i, 'smile2']

            # --------------------------- 1D smiles: smile 1 & 2 ------------------------------------
            # [修改] 使用自定义的 smiles_to_int 替代 tokenizer
            ids1_list, mask1_list = smiles_to_int(smile1, max_len=MAX_LEN)
            ids2_list, mask2_list = smiles_to_int(smile2, max_len=MAX_LEN)

            label = df1.loc[i, 'label']
            label = float(label)
            label = torch.tensor(label)

            # 转换为 LongTensor (Embedding层需要 Long 类型)
            input_ids1 = torch.tensor(ids1_list, dtype=torch.long)
            attention_mask1 = torch.tensor(mask1_list, dtype=torch.long)

            input_ids2 = torch.tensor(ids2_list, dtype=torch.long)
            attention_mask2 = torch.tensor(mask2_list, dtype=torch.long)

            data = DATA(ids1=input_ids1, mask1=attention_mask1,
                        y=label,
                        ids2=input_ids2, mask2=attention_mask2,
                        )

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        
        # save preprocessed data:
        self._process()
        torch.save((data, slices), self.processed_paths[0])

def repartition_biosnap_cold_start_existing_files():
    # ... (保持原有的 repartition 函数内容不变，此处省略以节省空间) ...
    pass

# --- 执行入口 ---
if __name__ == "__main__":
    pass