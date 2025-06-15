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

def eval_main():
    args = parse_args()

    val_data = Data(args.dataset , "valid", args.img_test_dir ,None)

    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size_valid, shuffle=False, num_workers=0)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network = CSFNet(version=args.network , pretrain=None, backbone_path=None,
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

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    import warnings
    warnings.filterwarnings("ignore")

    if args.weight_path is not None:
         network.load_state_dict(torch.load(args.weight_path))

    ############################################################################
    # test:
    ############################################################################
    metrics = StreamSegMetrics(args.num_classes)
    epoch_losses_val = []

    network.eval()
    batch_losses = []
    for RGB, X, label in tqdm(val_loader):
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

    metrics.get_results()
    metrics.reset()
    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    print("test/val loss: %g" % epoch_loss)


if __name__ == '__main__':
    eval_main()