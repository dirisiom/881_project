import json

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix
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
y = torch.tensor(labels, dtype=torch.uint8)
y_full = torch.zeros(feat.shape[0], dtype=torch.uint8)
y_full[idx_train] = y

data = Data(x=x, edge_index=edge_index, y=y_full)

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
hidden = 12

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(num_feats, hidden, num_classes).to(device)
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)
data.y = data.y.to(device)

# TODO: try new optims
optim = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-2)
# optim = torch.optim.Adam(model.parameters(), lr=0.0005)
crit = torch.nn.NLLLoss()

idx_train_no_val, idx_val = train_test_split(idx_train, test_size=0.1, random_state=5757)

train_mask = torch.zeros_like(y_full, dtype=torch.bool)
train_mask[idx_train_no_val] = True
val_mask = torch.zeros_like(y_full, dtype=torch.bool)
val_mask[idx_val] = True

best_loss = float('inf')
best_acc = 0
epoch_num = 400
for epoch in tqdm(range(epoch_num)):
    model.train()
    optim.zero_grad()
    out = model(data.x, data.edge_index)
    loss = crit(out[idx_train_no_val], data.y[idx_train_no_val])
    loss.backward()
    optim.step()

    model.eval()
    found_better = False
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        _, pred = out.max(dim=1)
        val_loss = crit(out[idx_val], data.y[idx_val])
        if val_loss < best_loss:
            found_better = True
            best_loss = val_loss
            model = model.to('cpu')
            torch.save(model.state_dict(), 'sage_no_cv.pth')
            model = model.to(device)
    train_corr = float(pred[idx_train_no_val].eq(data.y[idx_train_no_val]).sum().item())
    correct = float(pred[idx_val].eq(data.y[idx_val]).sum().item())
    acc = correct / len(idx_val)
    if found_better:
        best_acc = acc
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}, Validation Acc: {acc}, '
              f'Validation loss: {val_loss}, Train Acc: {train_corr / len(idx_train_no_val)}', )

# model = model.to('cpu')
# torch.save(model.state_dict(), 'sage_no_cv.pth')
print('\n\n\n')
print(f'Best validation loss: {best_loss}; Corresponding validation accuracy: {best_acc}')
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
