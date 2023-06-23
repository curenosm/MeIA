import numpy as np
import torch.nn as nn

def split_losocv(X, y, indices):
    """
    Receives the the indices for train and test datasets.
    Outputs the train, validation, and test data and label datasets.
    """

    # get indices to train and test sets
    train_indices = indices[0]
    test_indices = indices[1]

    # obtains train and test sets
    x_train = X[train_indices]
    y_train = np.squeeze(y[train_indices]).astype(np.int32)

    x_test = X[test_indices]
    y_test = np.squeeze(y[test_indices]).astype(np.int32)

    # get list of classes
    y_classes = np.unique(y_train)

    return [(x_train, y_train),
            (x_test, y_test),
            y_classes]


def get_subject_data(X, y, indices):
    """
    Receives the the indices for train and test datasets.
    Outputs the train, validation, and test data and label datasets.
    """

    # get indices to train and test sets
    test_indices = indices[1]

    x_t = X[test_indices]
    y_t = np.squeeze(y[test_indices]).astype(np.int32)

    # get list of classes
    y_classes = np.unique(y_t)

    return [(x_t, y_t),
            y_classes]

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def GRL(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, max_iter=10000):
    super(AdversarialNetwork, self).__init__()

    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.relu1 = nn.ReLU()

    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.relu2 = nn.ReLU()

    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.sigmoid = nn.Sigmoid()

    self.apply(init_weights)

    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):

    if self.training:
        self.iter_num += 1

    # [GRADIENT REVERSAL LAYER]
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(GRL(coeff))

    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)

    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]



def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable
