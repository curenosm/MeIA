#!/usr/bin/python3
#   Ejercicio S2-2-02:  Architecure criterion.
#                       Domain Specific Batch Normalization (DSBN) + L2-normalization
#   Pythonv3.8

#   Commands to execute this code:
#   python3.8 exercise-S2-2-02.py --dir_resume outputs --seed 223

    
import numpy as np
from utils.modules_pbashivan import load_bashivan_data
from utils.modules_pbashivan import get_subject_indices
from utils.utils import split_losocv
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
from models.model_dsbn import Extractor, Predictor
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings
from losses.cmd import CMD

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--max_iterations', type=int, default=1200)
parser.add_argument('--dir_pretrain', type=str, default="outputs/recresnet", help='folder used by DNN')
parser.add_argument('--dir_resume', type=str, default="outputs/resume", help='folder for resume')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--seed', type=int, default=223, help='seed')
args = parser.parse_args()


def test(network_f, network_h, test_loader):
    start_test = True
    with torch.no_grad():
        for data in test_loader:
            # get batch data
            samples = data[0].float().cuda()
            labels = data[1].long().cuda()

            # Get features and predictions
            preds = network_h(network_f(samples, domain_label=[1]))
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

def losocv(X, Y, subjects, args):
    """
        Leave One-Subject-Out Cross-Validation (LOOCV) on Cognitive Load data

        Params
            X: dataset containing all subject samples
            Y: dataset containing all subject labels
            subjects: dataset containing pairs between sample indexes and subjects
            args: hyper-parameters to train Custom Domain Adaptation.
    """

    # variable used to save accuracy results
    list_metrics_clsf = []

    # Extract pairs between indexes and subjects
    fold_pairs = get_subject_indices(subjects)

    # Iterate over fold_pairs
    for foldNum, fold in enumerate(fold_pairs):
        print('Beginning fold {0} out of {1}'.format(foldNum+1, len(fold_pairs)))

        if foldNum >= 2:# and foldNum + 1 <= 11:
            continue

        # Get source and target datasets
        (x_src, y_src), (x_trg, y_trg), y_classes = split_losocv(X, Y, fold)

        # data shape
        print("x_src-shape:", x_src.shape, "y_src-shape:", y_src.shape)
        print("x_trg-shape:", x_trg.shape, "y_trg-shape:", y_trg.shape)
        print("classes:", y_classes)

        # [NUMPY TO TENSOR]
        Sx_tensor = torch.tensor(x_src)
        Sy_tensor = torch.tensor(y_src)
        Tx_tensor = torch.tensor(x_trg)
        Ty_tensor = torch.tensor(y_trg)

        # [CONTAINERS]
        # create container for source labeled data
        source = TensorDataset(Sx_tensor, Sy_tensor)
        # create container for labeled target data
        target = TensorDataset(Tx_tensor, Ty_tensor)

        # [BUILD DATA LOADERS]
        # target
        source_loader = DataLoader(source, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        target_loader = DataLoader(target, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(target, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # counter
        epoch = 1
        eval_iter = x_src.shape[0] // args.batch_size

        # [Build Model]
        hidden_size = 128
        network_f = Extractor().cuda()
        network_h = Predictor(input_size=hidden_size, n_classes=4).cuda()

        # [OPTIMIZERS]
        optimizer_f = optim.Adam(network_f.parameters(), lr=args.lr, weight_decay=0.0001)
        optimizer_h = optim.Adam(network_h.parameters(), lr=args.lr, weight_decay=0.0001)

        # [Classification loss]
        criterion = nn.CrossEntropyLoss().cuda()

        # [Central moment Discrepancy]
        cmd_loss = CMD(n_moments=2)

        # length of datasets
        len_train_source = len(source_loader)
        len_train_target = len(target_loader)


        # Almacenar pérdida
        list_loss = []

        # [PRELIMINAR TOTAL LOSS]
        lambda_dis = 1.0
        for iter_num in range(0, args.max_iterations + 1):

            network_f.train()
            network_h.train()

            # Update loaders
            if iter_num % len_train_source == 0:
                data_iter_s = iter(source_loader)
            if iter_num % len_train_target == 0:
                data_iter_t = iter(target_loader)

            # get batch
            inputs_source, labels_source = next(data_iter_s)
            inputs_target, _ = next(data_iter_t)

            # TO CUDA
            inputs_source, inputs_target, labels_source = inputs_source.float().cuda(), inputs_target.float().cuda(), labels_source.long().cuda()

            # Extract features and outputs
            # Source domain
            features_source = network_f(inputs_source, domain_label=[0])
            outputs_source = network_h(features_source)

            # Target domain
            features_target = network_f(inputs_target, domain_label=[1])

            # [Classification Loss]
            classifier_loss = criterion(outputs_source, labels_source)

            # [CMD Loss]
            transfer_loss = cmd_loss.forward(features_source, features_target)

            total_loss = classifier_loss + lambda_dis * transfer_loss

            # Reset gradients
            optimizer_f.zero_grad()  # feature extractor
            optimizer_h.zero_grad()  # classifier


            # Barckpropagation
            total_loss.backward()

            # Update weights
            optimizer_f.step()
            optimizer_h.step()


            # append loss
            list_loss.append(total_loss.cpu().detach().numpy())

            # [EVALUATION]
            if iter_num % int(eval_iter) == 0 and iter_num > 0:

                network_f.eval()
                network_h.eval()

                acc, f1, auc = test(network_f, network_h, test_loader)
                avg_loss = np.array(list_loss).mean()
                print('Epoch: %d loss: %4f Acc: %.4f  F1-score: %.4f  AUC: %.4f' % (epoch, avg_loss, acc, f1, auc))
                epoch += 1


        print("\n")
        # add to list
        list_metrics_clsf.append([acc, f1, auc, foldNum+1])

    # To np array
    list_metrics_clsf = np.array(list_metrics_clsf)

    # Save Classification Metrics
    save_file = f"{args.dir_resume}/losocv-results.csv"
    with open(save_file, 'ab') as f:
        np.savetxt(f, list_metrics_clsf, delimiter=",", fmt='%0.4f')


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

    # run Leave One-Subject-Out Cross-Validation (LOSOCV).
    losocv(X, y, subjects, args)


# Call main module
main(args)
