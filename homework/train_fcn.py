import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, accuracy
from . import dense_transforms
import torch.utils.tensorboard as tb

def train(args):
    from os import path
    model = FCN()
    cmTrain = ConfusionMatrix()
    cmVal = ConfusionMatrix()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    """

    weight_decay=args.weight_decay
    batch_size=args.batch_size
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = weight_decay)
    loss = torch.nn.CrossEntropyLoss()

    train_data = load_dense_data('dense_data/valid', batch_size=batch_size)
    valid_data = load_dense_data('dense_data/train', batch_size=batch_size)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        acc_vals = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            label_loss = label.long()
            loss_val = loss(logit, label_loss)
            acc_val = accuracy(logit, label_loss)
            cmTrain.add(logit.argmax(1), label_loss)
            #confusionMatrix.class_iou()

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        #avg_acc = torch.cat(cmTrain.global_accuracy(), 0).detach().cpu().numpy()

        #iou=torch.cat(cmTrain.iou(), 0).detach().cpu().numpy()

        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        acc_vals = []
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            acc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
            cmVal.add(logit.argmax(1), label)
        #avg_vacc = torch.cat(cmVal.global_accuracy(), 0).detach().cpu().numpy()

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f iou = %0.3f' % (epoch, epoch, epoch, epoch))
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=12)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()
    train(args)
