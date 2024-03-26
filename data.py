import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import scipy.sparse as sp
import numpy as np
import json
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from tqdm import tqdm

adj = sp.load_npz('./data_2024/adj.npz')
feat = np.load('./data_2024/features.npy')
labels = np.load('./data_2024/labels.npy')
splits = json.load(open('./data_2024/splits.json'))
idx_train, idx_test = splits['idx_train'], splits['idx_test']

scaler = StandardScaler()
feats_normed = scaler.fit_transform(feat)

edge_index, _ = from_scipy_sparse_matrix(adj)
x = torch.tensor(feats_normed, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

print()


class GCN(torch.nn.Module):
    def __init__(self, feat_num, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feat_num, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


num_feats = data.num_features
num_classes = len(np.unique(data.y.numpy()))

splits = 5
kfold = KFold(n_splits=splits, shuffle=True, random_state=42)

val_accs = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(data.y)):
    print(f'Fold {fold + 1}/{kfold.n_splits}')

    model = GCN(num_feats, num_classes)
    lr = 0.01
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert idx to tensors for indexing
    train_idx_tensor = torch.tensor(idx_train)[torch.tensor(train_idx)]
    val_idx_tensor = torch.tensor(idx_train)[torch.tensor(val_idx)]

    acc = -1
    for epoch in tqdm(range(1, 201), desc=f'Training Fold {fold + 1}'):
        model.train()
        optim.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_idx_tensor], data.y[train_idx_tensor])
        loss.backward()
        optim.step()

        if epoch % 10 == 0:
            model.eval()
            _, pred = out.max(1)
            correct = pred[val_idx_tensor].eq(data.y[val_idx_tensor]).sum().item()
            acc = correct / val_idx_tensor.size(0)
            print(f'Epoch {epoch}, Loss: {loss.item(): .4f}, Validation Acc: {acc: .4f}')
        if epoch == 200:
            torch.save(model.state_dict(), f'model_fold_{fold}.pth')
    val_accs.append(acc)

print(val_accs)

average_val_acc = sum(val_accs) / splits
print(f'Average Validation Accuracy: {average_val_acc:.4f}')


# save our results
# preds = pred[idx_test]
# np.savetxt('submission.txt', preds, fmt='%d')
