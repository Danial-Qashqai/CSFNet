import argparse

class ArgumentParser(argparse.ArgumentParser):
    def set_common_args(self):

        self.add_argument('--batch_size', type=int, default=8,
                          help='batch size for training')

        self.add_argument('--batch_size_valid', type=int, default=None,
                          help='batch size for validation')


        self.add_argument('--epochs', default=300, type=int, metavar='N',
                          help='number of total epochs to run')

        self.add_argument('--eval_epochs_start', default=1, type=int, metavar='N',
                          help='epochs that evaluation start')

        # training hyper parameters
        self.add_argument('--lr', '--learning-rate', default=0.01,
                          type=float)

        self.add_argument('--num_gpus', type=int, default=1,
                          choices=[1, 2],
                          help='number of gpus')

        # model
        self.add_argument('--network', type=str, default='CSFNet-2',
                          choices=['CSFNet-1', 'CSFNet-2'], help='select model version')


        self.add_argument('--pretrained', type=bool,
                          choices=[True, False], default = False ,
                          help='pretrained backbone on imagenet dataset')

        self.add_argument('--backbone_path', type=str , help='Path to pretrained backbone if pretrained == True')

        self.add_argument('--weight_path', type=str  , help='Path to trained model weight')


        # dataset
        self.add_argument('--dataset', default='Cityscapes',
                          choices=['Cityscapes',
                                   'MFNet',
                                   'ZJU',
                                   'FMB'])

        self.add_argument('--img_train_dir',
                          default=None,
                          help='Path to dataset train image root.')

        self.add_argument('--img_test_dir',
                          default=None,
                          help='Path to dataset test/val image root.')


        self.add_argument('--num_classes', type=int, default=19,
                          help='number of classes')

        self.add_argument('--crop_H', type=int, default=512,
                          help='height of the random crop size in training')
        self.add_argument('--crop_W', type=int, default=1024,
                          help='width of the random crop size in training')
