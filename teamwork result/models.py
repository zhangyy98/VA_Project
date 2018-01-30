import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(net):
    count = [(name, param.numel()) for name, param in net.named_parameters()]
    total = sum(x[1] for x in count)
    for name, cnt in count:
        print(f'{name:25}{cnt:10}{cnt/total:.2%}')
    print(f'{net.__class__.__name__:25}{total:10}{1:.2%}')


def init_params(module):
    for name, param in module.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal(param)
        elif 'bias' in name:
            nn.init.constant(param, 0.01)


class ResidualRNNBlocks(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.lstms = [nn.LSTM(hidden_size if i else input_size, hidden_size, batch_first=True)
                      for i in range(num_layers)]
        for i, lstm in enumerate(self.lstms): self.add_module(str(i), lstm)

    def forward(self, x):
        x, _ = self.lstms[0](x)

        for lstm in self.lstms[1:]:
            if self.dropout: x = F.dropout(x, p=self.dropout, training=self.training)
            x_r, _ = lstm(x)
            x = x + x_r
        return x


class ResidualConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, fkernel_size, kernel_size, fstride, stride, padding, num_layers=2):
        super().__init__()
        self.convs = [nn.Conv2d(out_channels if i else in_channels,
                                out_channels,
                                kernel_size if i else fkernel_size,
                                stride if i else fstride,
                                padding if i else 0) for i in range(num_layers)]
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.act = nn.LeakyReLU(inplace=True)
        for i, conv in enumerate(self.convs): self.add_module(str(i), conv)

    def forward(self, x):
        x = self.convs[0](x)
        x = self.bn(x)
        x = self.act(x)
        for conv in self.convs[1:]:
            x_r = conv(x)
            x_r = self.bn(x_r)
            x_r = self.act(x_r)
            x = x + x_r
        return x


class ResidualRNNConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfeature, self.afeature, self.feature = 64, 64, 64
        self.vlstm = ResidualRNNBlocks(1024, self.vfeature, dropout=0.5)
        self.alstm = ResidualRNNBlocks(128, self.afeature, dropout=0.5)
        self.olstm = ResidualRNNBlocks(self.vfeature + self.afeature, self.feature, dropout=0.5)

        self.conv = nn.Sequential(ResidualConvBlocks(1, 16, (5, 2), (3, 3), (5, 2), 1, 1),
                                  nn.MaxPool2d(2, 2),
                                  ResidualConvBlocks(16, 32, (3, 2), (3, 3), (3, 2), 1, 1),
                                  nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(
            nn.Linear(32 * 2 * 4, 128, bias=False),
            nn.BatchNorm1d(128, affine=False),
            nn.LeakyReLU(inplace=True),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64, affine=False),
            nn.LeakyReLU(inplace=True),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        init_params(self)

    def forward(self, vfeat, afeat):
        vfeat = self.vlstm(vfeat)
        afeat = self.alstm(afeat)
        x = torch.cat((vfeat, afeat), -1)
        x = self.olstm(x)
        x = self.conv(x.unsqueeze(1))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x.view(x.size(0), -1))

        return x.squeeze()
