import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNetFeature, ResNetFeature_1500


class MainModel(nn.Module):

    def __init__(self, config):
        super(MainModel, self).__init__()

        self.config = config
        self.feature_3000 = ResNetFeature(config)
        self.feature_1500 = ResNetFeature_1500(config)
        self.classifier = PlainLSTM(config, hidden_dim=config['hidden_dim'], num_classes=config['num_classes'])
        self.up_sampling_layers = [
            nn.Sequential(
                torch.nn.Linear(
                    128,
                    128,
                ),
                nn.GELU(),
                torch.nn.Linear(
                    128,
                    128,
                ),
            )
        ]
        self.up_sampling_layers_sm = [
            nn.Sequential(
                torch.nn.Linear(
                    128,
                    128,
                ),
                nn.GELU(),
                torch.nn.Linear(
                    128,
                    128,
                ),
            )
        ]

    def forward(self, x):
        x1 = self.feature_3000(x)  # [128, 470, 128]
        # print(x1.shape)
        # exit()
        x_squeezed = x.squeeze(-1)
        x_1500 = torch.nn.functional.max_pool1d(x_squeezed, kernel_size=2, stride=2)
        x_1500 = x_1500.unsqueeze(-1)
        x2 = self.feature_1500(x_1500)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.up_sampling_layers[0] = self.up_sampling_layers[0].to(device)
        out_low_res = self.up_sampling_layers[0](x2)
        trend_mixing = x1 + out_low_res

        self.up_sampling_layers_sm[0] = self.up_sampling_layers_sm[0].to(device)
        out_high_res = self.up_sampling_layers_sm[0](x1)
        season_mixing = x2 + out_high_res

        mix_feature = 0.5 * trend_mixing + 0.5 * season_mixing

        out = self.classifier(mix_feature)

        return out


class PlainLSTM(nn.Module):
    def __init__(self, config, hidden_dim, num_classes):
        super(PlainLSTM, self).__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.num_classes = num_classes
        self.bidirectional = config['bidirectional']

        self.input_dim = 128

        # architecture
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers,
                            bidirectional=config['bidirectional'])
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def init_hidden(self, x):
        h0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim)).cuda()
        c0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim)).cuda()

        return h0, c0

    def forward(self, x):
        hidden = self.init_hidden(x)
        out, hidden = self.lstm(x, hidden)

        out_f = out[:, -1, :self.hidden_dim]
        out_b = out[:, 0, self.hidden_dim:]
        out = torch.cat((out_f, out_b), dim=1)
        out = self.fc(out)

        return out
