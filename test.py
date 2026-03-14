from utils import *
from model import *
from dataset import *
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader
import torch
import sklearn

def test(config):

    model = D1Model(config)
    saved_path = config.load_model_path
    model.load_state_dict(torch.load(saved_path + config.model_name + '.pt', map_location=config.device))
    model.eval()
    model.to(config.device)

    test_dataset = DDIDataset(root=config.test_root, path=config.test_path)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
                             follow_batch=['pos1', 'pos2'])


    print('-----Test-----')
    label_test, pred_test, prob_test, _ = get_label_pred_ddi_binary(model, test_loader)

    acc = sklearn.metrics.accuracy_score(label_test, pred_test)
    print('acc:', acc)

    f1 = sk_p_r(label_test, pred_test)
    print('macro f1:', f1)

    roc_auc_score = sklearn.metrics.roc_auc_score(label_test, prob_test)
    print('roc_auc:', roc_auc_score)

    pr_auc_score = sklearn.metrics.average_precision_score(label_test, prob_test)
    print('pr_auc:', pr_auc_score)

    cf_matrix = confusion_matrix(label_test, pred_test)
    print('confusion matrix:\n', cf_matrix)

    draw_roc_pr_curve(label_test, prob_test, saved_path)
    draw_confusion_matrix(config, cf_matrix, saved_path)


