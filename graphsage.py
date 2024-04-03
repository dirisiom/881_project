import json

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm
from sklearn.model_selection import train_test_split

adj = sp.load_npz('./data_2024/adj.npz')
feat = np.load('./data_2024/features.npy')
labels = np.load('./data_2024/labels.npy')
splits = json.load(open('./data_2024/splits.json'))
idx_train, idx_test = splits['idx_train'], splits['idx_test']

scaler = StandardScaler()
feats_normed = scaler.fit_transform(feat)
# feats_normed = feat

edge_index, _ = from_scipy_sparse_matrix(adj)
x = torch.tensor(feats_normed, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

print()


# GraphSAGE Model
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GraphSAGE, self).__init__()
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.bn1 = BatchNorm(hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         self.bn2 = BatchNorm(hidden_channels)
#         self.conv3 = SAGEConv(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         x = F.relu(self.bn1(self.conv1(x, edge_index)))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.bn2(self.conv2(x, edge_index)))
#         x = F.dropout(x, training=self.training)
#         x = self.conv3(x, edge_index)
#         return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.sage2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.sage3(x, edge_index)
        return F.log_softmax(x, dim=1)


num_feats = data.num_features
num_classes = len(np.unique(data.y.numpy()))
# TODO: tune
hidden = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(num_feats, hidden, num_classes).to(device)
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)
data.y = data.y.to(device)
# TODO: try new optims
optim = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=4e-3)
# optim = torch.optim.Adam(model.parameters(), lr=0.0005)
crit = torch.nn.NLLLoss()

idx_train_sort = idx_train[:]
idx_train_sort.sort()

inv_map = {v: i for i, v in enumerate(idx_train_sort)}

idx_train_no_val, idx_val = train_test_split(idx_train, test_size=0.1, random_state=5757)
idx_train_no_val.sort()
idx_val.sort()

train_mask = torch.zeros(data.y.shape[0], dtype=torch.bool)
val_mask = torch.zeros(data.y.shape[0], dtype=torch.bool)

for idx in idx_train_no_val:
    train_mask[inv_map[idx]] = True

for idx in idx_val:
    val_mask[inv_map[idx]] = True

epoch_num = 200
for epoch in tqdm(range(epoch_num)):
    model.train()
    optim.zero_grad()
    out = model(data.x, data.edge_index)
    loss = crit(out[idx_train_no_val], data.y[train_mask.nonzero().squeeze()])
    loss.backward()
    optim.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        _, pred = out.max(dim=1)
        val_loss = crit(out[idx_val], data.y[val_mask.nonzero().squeeze()])
    train_corr = float(pred[idx_train_no_val].eq(data.y[train_mask.nonzero().squeeze()]).sum().item())
    correct = float(pred[idx_val].eq(data.y[val_mask]).sum().item())
    acc = correct / len(idx_val)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}, Validation Acc: {acc}, '
              f'Train Acc: {train_corr / len(idx_train_no_val)}', )

model = model.to('cpu')
torch.save(model.state_dict(), 'sage_no_cv.pth')

model.load_state_dict(torch.load('sage_no_cv.pth'))

model = model.to(device)
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    _, test_preds = out.max(dim=1)

# save our results
preds = test_preds[idx_test]
preds = preds.to('cpu')
np.savetxt('submission_sage.txt', preds, fmt='%d')
