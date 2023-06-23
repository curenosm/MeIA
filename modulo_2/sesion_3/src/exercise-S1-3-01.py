#!/usr/bin/python3
#   Ejercicio S1-3-01: RecResNet + k-Fold Cross-Validation.
#   Commands to execute this code:
#   python3.8 exercise-S1-3-01.py --dir_resume outputs --seed 223

    
import numpy as np
from utils.modules_pbashivan import load_bashivan_data
from utils.modules_pbashivan import get_subject_indices_kfcv
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
from models.model_base import RecResNet
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dir_pretrain', type=str, default="outputs/recresnet", help='folder used by DNN')
parser.add_argument('--dir_resume', type=str, default="outputs/resume", help='folder for resume')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--seed', type=int, default=223, help='seed')
args = parser.parse_args()


def test(model, test_loader):
    start_test = True
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # get batch data
            samples = data[0].float().cuda()
            labels = data[1].long().cuda()

            # Get features and predictions
            preds = model(samples)
            # probabilidad máxima
            predictions = preds.data.max(1)[1]

            if start_test:
                y_pred = predictions.cpu().numpy()
                y_true = labels.data.cpu().numpy()
                start_test = False
            else:
                y_pred = np.concatenate((y_pred, predictions.cpu().numpy()), 0)
                y_true = np.concatenate((y_true, labels.data.cpu().numpy()), 0)

        # Binarize ytest with shape (n_samples, n_classes)
        labels = np.unique(y_true)
        ytest = label_binarize(y_true, classes=labels)
        ypreds = label_binarize(y_pred, classes=labels)

        # compute utils
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')


    return acc, f1, auc

def kfcv(X, Y, subjects, args):
    """
        K-fold Cross Validation

        Params
            X: dataset containing all subject samples
            Y: dataset containing all subject labels
            subjects: dataset containing pairs between sample indexes and subjects
            args: hyper-parameters to train Custom Domain Adaptation.
    """

    # Extract pairs between indexes and subjects
    subject_indices = get_subject_indices_kfcv(subjects)

    # Iterate over fold_pairs
    for foldNum, fold_indices in enumerate(subject_indices):
        print('Beginning fold {0} out of {1}'.format(foldNum + 1, len(subject_indices)))

        # get data from subject according to fold indices
        x = X[fold_indices]
        y = np.squeeze(Y[fold_indices]).astype(np.int32)
        y_classes = np.unique(y)

        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(x, y)

        list_metrics_clsf = []

        for i, (train_index, test_index) in enumerate(skf.split(x, y)):

            x_train, y_train = x[train_index], y[train_index]
            x_test, y_test = x[test_index], y[test_index]



            # data shape
            print("x_train-shape:", x_train.shape, "y_train-shape:", y_train.shape)
            print("x_test-shape:", x_test.shape, "y_test-shape:", y_test.shape)
            print("classes:", y_classes)


            # [NUMPY TO TENSOR]
            Sx_tensor = torch.tensor(x_train)
            Sy_tensor = torch.tensor(y_train)
            Tx_tensor = torch.tensor(x_test)
            Ty_tensor = torch.tensor(y_test)

            # [CONTAINERS]
            # create container for source labeled data
            source = TensorDataset(Sx_tensor, Sy_tensor)
            # create container for labeled target data
            target = TensorDataset(Tx_tensor, Ty_tensor)

            # [BUILD DATA LOADERS]
            # target
            source_loader = DataLoader(source, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
            target_loader = DataLoader(target, batch_size=20, shuffle=False, num_workers=4)

            # [Build Model]
            model = RecResNet(n_classes=4).cuda()

            # [Optimizers]
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005)
            # [Classification loss]
            criterion = nn.CrossEntropyLoss().cuda()



            for epoch in range(args.epochs):

                # iterator
                iter_train = iter(source_loader)
                list_loss = []

                for c in range(len(source_loader)):
                    # get batch
                    samples, labels = iter_train.next()
                    samples = samples.float().cuda()
                    labels = labels.long().cuda()

                    model.train(True)

                    # get features and labels for source and target domain
                    outputs = model(samples)
                    cls_loss = criterion(outputs, labels)

                    # reset gradients
                    optimizer.zero_grad()

                    # compute gradients
                    cls_loss.backward()

                    # update weights
                    optimizer.step()

                    # append loss
                    list_loss.append(cls_loss.cpu().detach().numpy())

                # evaluation
                model.eval()

                # [DFN | CNN]
                acc_test, f1_test, auc_test = test(model, target_loader)
                avg_loss = np.array(list_loss).mean()
                print('Epoch: %d loss: %4f Acc: %.4f  F1-score: %.4f  AUC: %.4f' % (epoch+1, avg_loss, acc_test, f1_test, auc_test))

            print("\n")
            # add to list
            list_metrics_clsf.append([acc_test, f1_test, auc_test, foldNum+1])

        # To np array
        list_metrics_clsf = np.array(list_metrics_clsf)

        # Save Classification Metrics
        save_file = args.dir_resume + "/kfcv-results.csv"
        f = open(save_file, 'ab')
        np.savetxt(f, list_metrics_clsf, delimiter=",", fmt='%0.4f')
        f.close()


def main(args):
    # set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    #tf.set_random_seed(args.seed)
    print("SEED:", args.seed)


    # Load public cognitive load dataset (Bashivan et al., 2016)
    # We used a window size of 32x32 can be used as in
    # (Bashivan et al., 2016; Jiménez-Guarneros & Gómez-Gil, 2017).
    # The 'generate_images' options must be set to 'True' the first time to generate data samples.
    X, y, subjects = load_bashivan_data("/home/magdiel/Data/COGNITIVE-LOAD/",
                        n_channels=64, n_windows=7, n_bands=3, generate_images=False,
                        size_image=24, visualize=False)

    # run K-Fold Cross Validation
    kfcv(X, y, subjects, args)


# Call main module
main(args)
