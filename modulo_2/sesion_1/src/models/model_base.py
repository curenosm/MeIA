import torch
import torch.nn as nn
import torch.nn.functional as F

class RecResNet(nn.Module):
    '''
    Build the  Mean Basic model performing a classification with CNN 

    param input_image: list of EEG image [batch_size, n_window, n_channel, h, w]
    param n_classes: number of classes
    return x: output of the last layers after the log softmax
    '''
    def __init__(self, n_classes=4):
        super(RecResNet, self).__init__()

        self.num_layers = 1
        self.hidden_dim = 128

        # Layer 1
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding='valid')
        self.bn1 = nn.BatchNorm3d(16, momentum=0.1)

        # Layer 2
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='valid')
        self.bn2 = nn.BatchNorm3d(16, momentum=0.1)

        #  Residual block ->
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding='same')
        self.bn3 = nn.BatchNorm3d(16, momentum=0.1)

        self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding='same')
        self.bn4 = nn.BatchNorm3d(16, momentum=0.1)

        # Capa Recurrente
        # [32x32 -> 3136]
        # [24x24 -> 1600]
        self.rnn = nn.GRU(input_size=1600, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.5)

        #
        self.fc1 = nn.Linear(128, 128)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 128)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, n_classes)

    
    def forward(self, x):
        # Layer 1
        x = F.tanh(self.bn1(self.conv1(x)))

        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))

        # Residual block
        o = F.relu(self.bn3(self.conv3(x)))
        o = self.bn4(self.conv4(o))
        r = F.relu(x + o)

        # swap axes
        r = torch.swapaxes(r, 1, 2)
        # reshape
        r = torch.reshape(r, (r.shape[0], r.shape[1], r.shape[2] * r.shape[3] * r.shape[4]))
        # RNN
        out, _ = self.rnn(r)

        # redimensionar las salidas en el forma (batch_size, seq_length, hidden_size)
        # tal que esto pueda ser manejable por la capa de salida
        feats_rnn = out[:, -1]

        feats = F.relu(self.fc1(feats_rnn))
        feats = self.drop1(feats)
        feats = F.relu(self.fc2(feats))
        feats = self.drop2(feats)

        out = self.fc3(feats)

        return out


