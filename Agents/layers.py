import torch
from torch.nn import init
import numpy as np

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
            layers.append(torch.nn.ReLU())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class BiRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # TODO: add initialization,weight decay
        # define network, not cell
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.05, bidirectional=True)
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.05, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size*2, num_classes,bias=False)  # 1 for bidirection
        self.fc.weight = init.xavier_normal_(self.fc.weight,gain=1)
        # self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Set initial states
        # TODO: change batch size
        h0 = torch.zeros(self.num_layers*2, 256, self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, 256, self.hidden_size).to(device)
        # Forward propagate LSTM
        pred = []
        out, _ = self.lstm(x, (h0, c0))   # out: tensor of shape (seq_id, batch_size, hidden_size*2)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # may not need to pad?
        # out = self.gru(x, h0)
        for i in range(out.size(1)):
          # resi = self.softmax(self.activation(self.fc(out[:,i,:])))
          resi = self.softmax(self.fc(out[:,i,:]))
          pred.append(resi)  # [seq_id,256,1]
        return torch.stack(pred,dim=1) # 横向堆叠 [256,200]


class BiRNN_nopack(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN_nopack, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # TODO: add initialization,weight decay
        # define network, not cell
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.05, bidirectional=True)
        # self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.05, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size*2, num_classes,bias=False)  # 1 for bidirection
        self.fc.weight = init.xavier_normal_(self.fc.weight,gain=1)
        self.activation = torch.nn.Sigmoid()
        # self.activation = nn.ReLU()

    def forward(self, x):
        # Set initial states
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        pred = []
        out, _ = self.lstm(x, (h0, c0))   # out: tensor of shape (seq_id, batch_size, hidden_size*2)
        # print(x.size(),out.size())
        # out = self.gru(x, h0)
        for i in range(x.size(0)):
            pred.append(torch.squeeze(self.activation(self.fc(out[i]))))  # [seq_id,256,1]
        return torch.stack(pred, dim=-1) # 横向堆叠 [256,200]

# use
"""
for i, fea, lab in data_loader(data_path, batch_size=2):
    print(i)
    features = torch.from_numpy(fea).reshape(-1, sequence_length, input_size).to(device)
    lab = torch.from_numpy(lab).reshape(-1, sequence_length).to(device)  # no need trans to [seqlen,batch]
    features = features.permute(1, 0, 2).to(torch.float32)
    outputs = model(features)
    # print(outputs.size(),lab.size())
    with torch.no_grad():
        outputs_ls.append(outputs.cpu().detach().numpy())
    count += 1
    if count > max_count:
        break
"""
"""
torch.save(model.state_dict(), PATH)
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""