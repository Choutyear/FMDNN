import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

import numpy as np


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)


    fclass = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    fclass.sort()

    class_indices = dict((k, v) for v, k in enumerate(fclass))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in fclass:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        plt.bar(range(len(fclass)), every_class_num, align='center')
        plt.xticks(range(len(fclass)), fclass)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # cumulative loss
    accu_num = torch.zeros(1).to(device)  # cumulative correctly predicted sample count
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    all_labels = []
    all_preds = []
    all_pred_scores = []  # stores the predicted scores for each class
    for step, data in enumerate(data_loader):
        images, labels = data
        all_labels.extend(labels.numpy())
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        all_preds.extend(pred_classes.cpu().numpy())
        all_pred_scores.extend(pred.cpu().detach().numpy())  # detach the predicted scores and convert to numpy array

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        cm = confusion_matrix(all_labels, all_preds)
        tn = cm.sum(axis=1) - np.diag(cm)
        fp = cm.sum(axis=0) - np.diag(cm)
        # recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        specificity = np.mean(tn / (tn + fp + 1e-7))

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, recall: {:.3f}, specificity: {:.3f}, " \
                           "precision: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            recall,
            specificity,
            precision)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, recall, specificity, precision


# Recall + Specificity + Precision
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # cumulative correctly predicted sample count
    accu_loss = torch.zeros(1).to(device)  # cumulative loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    all_labels = []
    all_preds = []
    all_pred_scores = []  # stores the predicted scores for each class
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_pred_scores.extend(pred.cpu().numpy())  # convert the predicted scores to numpy array

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        cm = confusion_matrix(all_labels, all_preds)
        tn = cm.sum(axis=1) - np.diag(cm)
        fp = cm.sum(axis=0) - np.diag(cm)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        specificity = np.mean(tn / (tn + fp + 1e-7))

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, recall: {:.3f}, specificity: {:.3f}, " \
                           "precision: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            recall,
            specificity,
            precision)

    return accu_loss.item() / (
            step + 1), accu_num.item() / sample_num, recall, specificity, precision, all_labels, all_pred_scores


def plot_roc_curve(all_labels, all_pred_scores, num_classes, epoch):
    all_labels = np.array(all_labels)
    all_pred_scores = np.array(all_pred_scores)
    k = epoch
    all_labels = label_binarize(all_labels, classes=[*range(num_classes)])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_pred_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(num_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for class %d' % i)
        plt.legend(loc="lower right")
        plt.savefig('%d_roc_curve_class_%d.png' % (k, i))
        # plt.show()
