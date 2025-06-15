import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import argparse

from Dataloader.Data import Data
from network import CSFNet
from Metric.mIoU import StreamSegMetrics
from args import ArgumentParser

def parse_args():
    parser = ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()
    return args

def train_main():
    args = parse_args()

    train_data = Data(args.dataset , "train", args.img_train_dir , (args.crop_H, args.crop_W))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size , shuffle=True, num_workers=4)

    val_data = Data(args.dataset , "valid", args.img_test_dir ,None)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size_valid, shuffle=False, num_workers=4)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network = CSFNet(version=args.network , pretrain=args.pretrained, backbone_path=args.backbone_path,
                     dataset=args.dataset , num_classes=args.num_classes)


    if torch.cuda.device_count() > 1:
            if args.num_gpus == 2:
                    print("use 2 gpu")
                    network = nn.DataParallel(network)
            if args.num_gpus == 1:
                    print("use 1 gpu")
                    network = nn.DataParallel(network, device_ids=[0])

    elif torch.cuda.device_count() == 1:
            print("use 1 gpu")
            network = nn.DataParallel(network ,device_ids=[0])
    network = network.to(device)

    from torch.optim import SGD
    optimizer = SGD(network.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lambda1 = lambda epoch: ((1 - (epoch / args.epochs)) ** 0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda1)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    import warnings
    warnings.filterwarnings("ignore")

    if args.weight_path is not None:
         network.load_state_dict(torch.load(args.weight_path))


    metrics = StreamSegMetrics(args.num_classes)
    epoch_losses_train = []
    epoch_losses_val = []
    num_epoch = args.epochs
    z=0
    r=0

    for epoch in range(1, num_epoch+1):
        print("epoch: %d/%d" % (epoch, num_epoch))
        ############################################################################
        # train:
        ############################################################################
        network.train()
        batch_losses = []
        for RGB, X, label in tqdm(train_loader,colour="blue"):
        # for RGB, X, label in train_loader:
            label = label - 1
            RGB = RGB.to(device)
            X = X.to(device)
            label = (label.type(torch.LongTensor)).to(device)
            outputs = network(RGB, X)

            # compute the loss:
            loss = criterion(outputs, label)
            loss_value = loss.data.detach().cpu().numpy()
            batch_losses.append(loss_value)

            # optimization step:
            optimizer.zero_grad()  # (reset gradients)
            loss.backward()  # (compute gradients)
            optimizer.step()  # (perform optimization step)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        print("train loss: %g" % epoch_loss)

        scheduler.step()

        ############################################################################
        # test:
        ############################################################################
        if epoch >= args.eval_epochs_start:
            network.eval()
            batch_losses = []
            for RGB, X, label in tqdm(val_loader,colour="red"):
            # for RGB, X, label in val_loader:
                with torch.no_grad():
                    label = label - 1
                    RGB = RGB.to(device)
                    X = X.to(device)
                    label = (label.type(torch.LongTensor)).to(device)

                    outputs = network(RGB, X)

                    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                    targets = label.cpu().numpy()
                    metrics.update(targets, preds)

                    # compute the loss:
                    loss = criterion(outputs, label)
                    loss_value = loss.data.cpu().numpy()
                    batch_losses.append(loss_value)

            z = metrics.get_results()
            metrics.reset()
            epoch_loss = np.mean(batch_losses)
            epoch_losses_val.append(epoch_loss)
            print("test/val loss: %g" % epoch_loss)

        if z > r:
            if args.dataset == "Cityscapes":
                    torch.save(network.state_dict(), f"Checkpoints/Cityscapes/best_{args.network}_city.pth")

            elif args.dataset == "MFNet":
                    torch.save(network.state_dict(), f"Checkpoints/MFNet/best_{args.network}_MFNet.pth")

            elif args.dataset == "ZJU":
                    torch.save(network.state_dict(), f"Checkpoints/ZJU/best_{args.network}_ZJU.pth")

            elif args.dataset == "FMB":
                    torch.save(network.state_dict(), f"Checkpoints/FMB/best_{args.network}_FMB.pth")

            else:
                    torch.save(network.state_dict(), f"Checkpoints/best_{args.network}_weight.pth")
            print("########################################")
            print("              BEST RESULT               ")
            print("########################################")
            r = z


if __name__ == '__main__':
    train_main()