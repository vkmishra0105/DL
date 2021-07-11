from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNNClassifier().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loss = torch.nn.CrossEntropyLoss()

    train_data = load_data('data/train', batch_size=16)
    valid_data = load_data('data/valid', batch_size=16)

    global_step = 0
    for epoch in range(35):
        model.train()
        acc_vals = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            acc_val = accuracy(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        avg_acc = sum(acc_vals) / len(acc_vals)

        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        acc_vals = []
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            acc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        avg_vacc = sum(acc_vals) / len(acc_vals)

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        save_model(model)
    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # custom arguments here

    args = parser.parse_args()
    train(args)
