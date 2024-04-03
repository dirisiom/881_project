import json

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm

adj = sp.load_npz('./data_2024/adj.npz')
feat = np.load('./data_2024/features.npy')
labels = np.load('./data_2024/labels.npy')
splits = json.load(open('./data_2024/splits.json'))
idx_train, idx_test = splits['idx_train'], splits['idx_test']

# scaler = StandardScaler()
# feats_normed = scaler.fit_transform(feat)
feats_normed = feat

edge_index, _ = from_scipy_sparse_matrix(adj)
x = torch.tensor(feats_normed, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

print()


class GCN(torch.nn.Module):
    def __init__(self, feat_num, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feat_num, 64)
        self.bn1 = BatchNorm(64)
        self.conv2 = GCNConv(64, 128)
        self.bn2 = BatchNorm(128)
        self.conv3 = GCNConv(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


num_feats = data.num_features
num_classes = len(np.unique(data.y.numpy()))

splits = 5
kfold = KFold(n_splits=splits, shuffle=True, random_state=5757)

val_accs = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(data.y)):
    print(f'Fold {fold + 1}/{kfold.n_splits}')

    model = GCN(num_feats, num_classes)
    lr = 0.008
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True

    acc = -1
    epoch_count = 350
    for epoch in tqdm(range(1, epoch_count + 1), desc=f'Training Fold {fold + 1}'):
        model.train()
        optim.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask.nonzero().squeeze()])
        loss.backward()
        optim.step()

        if epoch % 10 == 0:
            model.eval()
            _, pred = out.max(1)
            correct = pred[val_mask].eq(data.y[val_mask.nonzero().squeeze()]).sum().item()
            acc = correct / val_idx.size
            print(f'Epoch {epoch}, Loss: {loss.item(): .4f}, Validation Acc: {acc: .2f}')
        if epoch == epoch_count:
            torch.save(model.state_dict(), f'model_fold_{fold}.pth')
    val_accs.append(acc)

print(val_accs)

average_val_acc = sum(val_accs) / splits
print(f'Average Validation Accuracy: {average_val_acc:.4f}')

# this is probably not the best way to do this but i was selecting the model
# based on the highest validationn accuracy at the last epoch for each fold
best_fold = val_accs.index(max(val_accs))
best_path = f'model_fold_{best_fold}.pth'

best_model = GCN(num_feats, num_classes)
best_model.load_state_dict(torch.load(best_path))
best_model.eval()

with torch.no_grad():
    out = best_model(data)
    _, test_preds = out.max(dim=1)


# save our results
preds = test_preds[idx_test]
np.savetxt('submission.txt', preds, fmt='%d')
