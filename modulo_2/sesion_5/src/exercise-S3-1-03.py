#!/usr/bin/python3
#   Ejercicio S2-2-01: Adversarial Discriminative Domain Adaptation (ADDA)
#   Commands to execute this code:
#   python3.8 exercise-S3-1-03.py --dir_resume outputs --seed 223

import numpy as np
from utils.modules_pbashivan import load_bashivan_data
from utils.modules_pbashivan import get_subject_indices
from utils.utils import split_losocv
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
from models.model_base import RecResNet
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings
from utils.utils import set_requires_grad


warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--max_iterations', type=int, default=1200)
parser.add_argument('--dir_resume', type=str, default="outputs/resume", help='folder for resume')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
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

        # Only Subjects 1 and 2 are executed
        if foldNum + 1 >= 3:
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
        eval_iter = x_src.shape[0] // args.batch_size # evaluate each 'eva_iter'

        # [Build Model]
        source_model = RecResNet(n_classes=4).cuda()
        source_model.load_state_dict(torch.load("./trained_model/source" + str(foldNum+1) + ".pt"))
        source_model.eval()
        set_requires_grad(source_model, requires_grad=False)

        # [Target model]
        target_model = RecResNet(n_classes=4).cuda()
        target_model.load_state_dict(torch.load("./trained_model/source" + str(foldNum+1) + ".pt"))

        # Create adversarial discriminator
        discriminator = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).cuda()

        # [OPTIMIZERS]
        optimizer_disc = torch.optim.SGD(discriminator.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        optimizer_target = torch.optim.SGD(target_model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

        # [Binary Cross-entropy]
        bce_loss = nn.BCEWithLogitsLoss()

        # length of datasets
        len_train_source = len(source_loader)
        len_train_target = len(target_loader)

        # Almacenar pérdida
        list_loss = []
        total_loss = 0
        source_model.eval()

        for iter_num in range(0, args.max_iterations + 1):
            target_model.train()
            discriminator.train()
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

            # Train discriminator
            set_requires_grad(target_model, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)

            # get predictions
            source_preds = source_model(inputs_source)
            target_preds = target_model(inputs_target)

            # combine predictions
            discriminator_x = torch.cat([source_preds, target_preds])
            # generate domain labels
            discriminator_y = torch.cat([torch.ones(inputs_source.shape[0]), torch.zeros(inputs_target.shape[0])]).cuda()
            # apply discriminator
            preds = discriminator(discriminator_x).squeeze()
            # [adversarial loss]
            loss = bce_loss(preds, discriminator_y)


            optimizer_disc.zero_grad()
            loss.backward()
            optimizer_disc.step()

            total_loss += loss.item()

            # Train classifier
            set_requires_grad(target_model, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)

            # generate predictions
            target_preds = target_model(inputs_target)

            # flipped labels
            discriminator_y = torch.ones(inputs_target.shape[0]).cuda()

            # get domain predictions
            preds = discriminator(target_preds).squeeze()

            # [binary cross-entropy loss]
            loss = bce_loss(preds, discriminator_y)

            optimizer_target.zero_grad()
            loss.backward()
            optimizer_target.step()

            # append loss
            list_loss.append(loss.cpu().detach().numpy())

            # [EVALUATION]
            if iter_num % int(eval_iter) == 0 and iter_num > 0:
                target_model.eval()
                discriminator.eval()
                acc, f1, auc = test(target_model, test_loader)

                avg_loss = np.array(list_loss).mean()
                print('Epoch: %d loss: %4f Acc: %.4f  F1-score: %.4f  AUC: %.4f' % (epoch, avg_loss, acc, f1, auc))
                epoch += 1

                total_loss = 0
                list_loss = []

        print("\n")
        # add to list
        list_metrics_clsf.append([acc, f1, auc, foldNum+1])
    
    # To np array
    list_metrics_clsf = np.array(list_metrics_clsf)

    # Save Classification Metrics
    save_file = args.dir_resume+"/losocv-results.csv"
    f=open(save_file, 'ab')
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

    # run Leave One-Subject-Out Cross-Validation (LOSOCV).
    losocv(X, y, subjects, args)




# Call main module
main(args)
