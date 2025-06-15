import torch
import os
import numpy as np
import cv2
import torchvision
import random


def Data(dataset_name, mode, img_path, crop_size = None):
       if dataset_name == "Cityscapes":
           if mode == "train":
                   trans = torchvision.transforms.Compose([
                                   torchvision.transforms.RandomHorizontalFlip()
                                   , torchvision.transforms.RandomCrop(crop_size, pad_if_needed=True)])

                   return Data_Cityscapes(img_path, transform=trans, mode = mode)
           else:
                   return Data_Cityscapes(img_path, mode=mode)


       if dataset_name == "MFNet":
           if mode == "train":
                   trans = torchvision.transforms.Compose([
                       torchvision.transforms.RandomHorizontalFlip()
                       , torchvision.transforms.RandomCrop(crop_size, pad_if_needed=True)])

                   return Data_MFNet(img_path, transform=trans,  mode=mode)
           else:
                   return Data_MFNet(img_path, mode=mode)


       if dataset_name == "ZJU":
           if mode == "train":
                   trans = torchvision.transforms.Compose([
                       torchvision.transforms.RandomCrop(crop_size, pad_if_needed=True)])

                   return Data_ZJU(img_path, transform=trans,  mode=mode)
           else:
                   return Data_ZJU(img_path,  mode=mode)



       if dataset_name == "FMB":
           if mode == "train":
                   trans = torchvision.transforms.Compose([
                       torchvision.transforms.RandomHorizontalFlip()
                       , torchvision.transforms.RandomCrop(crop_size, pad_if_needed=True)])

                   return Data_FMB(img_path, transform=trans,  mode=mode)
           else:
                   return Data_FMB(img_path, mode=mode)



class Data_Cityscapes (torch.utils.data.Dataset):
        def __init__(self, image_path, transform=None, mode="test"):
            self.transform = transform
            self.mode = mode

            self.w_image = 1024
            self.h_image = 512

            rgb_path = os.path.join(image_path, "rgb")
            depth_path = os.path.join(image_path, "depth_raw")
            label_path = os.path.join(image_path, "labels_19")

            self.examples = []
            folder = os.listdir(rgb_path)

            for folders in folder:
                folders_path = os.path.join(rgb_path, folders)
                img_name = os.listdir(folders_path)
                for imgs in img_name:
                    img_dir = os.path.join(folders_path, imgs)
                    depth_name = imgs.split("_leftImg8bit")[0] + "_depth.npy"
                    depth_dir = os.path.join(depth_path, os.path.join(folders, depth_name))
                    label_name = imgs.split("_leftImg8bit")[0] + "_gtFine_labelIds.png"
                    label_dir = os.path.join(label_path, os.path.join(folders, label_name))

                    sample = {}
                    sample["image_path"] = img_dir
                    sample["depth_path"] = depth_dir
                    sample["label_path"] = label_dir
                    self.examples.append(sample)

        def __getitem__(self, index):
            img_path = self.examples[index]["image_path"]
            dep_path = self.examples[index]["depth_path"]
            lab_path = self.examples[index]["label_path"]

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dep = np.load(dep_path)
            dep = dep.astype('float32')
            dep[dep > 300] = 0
            lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)

            trans_img = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

            trans_gray = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.Normalize(mean=(0.449), std=(0.226))
            ])

            trans_dep = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mean=(31.715617493177906), std=(38.70280704877372))
            ])

            trans_jit = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=(0.9, 1.1))
            ])

            if self.mode == "valid":
                img = cv2.resize(img, (self.w_image, self.h_image), interpolation=cv2.INTER_LINEAR)
                dep = cv2.resize(dep, (self.w_image, self.h_image), interpolation=cv2.INTER_LINEAR)
                lab = cv2.resize(lab, (self.w_image, self.h_image), interpolation=cv2.INTER_NEAREST)

            scales = [0.75, 1, 1.25, 1.5, 1.75]
            if self.mode == "train":
                S = scales[random.randrange(0, 5)]
                width = int(self.w_image * S)
                height = int(self.h_image * S)

                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                dep = cv2.resize(dep, (width, height), interpolation=cv2.INTER_LINEAR)
                lab = cv2.resize(lab, (width, height), interpolation=cv2.INTER_NEAREST)

            lab = lab.reshape((lab.shape[0], lab.shape[1], 1))
            dep = dep.reshape((dep.shape[0], dep.shape[1], 1))

            trans_totensor = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()])
            img = trans_totensor(img)
            dep = trans_totensor(dep)
            lab = trans_totensor(lab) * 255

            if self.mode == "train":
                concat_image = torch.cat((img, dep, lab), dim=0)
                concat_image = self.transform(concat_image)
                img = concat_image[0:3]
                img = trans_jit(img)
                dep = concat_image[3]
                dep = torch.reshape(dep, (1, dep.shape[0], dep.shape[1]))
                lab = concat_image[4]

            else:
                lab = torch.reshape(lab, (lab.shape[1], lab.shape[2]))

            dep = trans_dep(dep)
            gray = trans_gray(img)
            img = trans_img(img)

            dep = torch.cat((gray, dep), dim=0)

            return (img, dep, lab)

        def __len__(self):
            return len(self.examples)

class Data_MFNet(torch.utils.data.Dataset):
    def __init__(self, img_folder, transform=None, mode="test"):
        self.transform = transform
        self.mode = mode

        self.w_image = 640
        self.h_image = 480

        if  self.mode == "train":
             text_path= os.path.join(img_folder,"train.txt")
        else:
             text_path = os.path.join(img_folder, "test.txt")

        image_path= os.path.join(img_folder,"images")
        label_path = os.path.join(img_folder, "labels")

        f = open(text_path, "r")

        self.examples = []
        for item in f:
            if item.__contains__("flip"):
                pass
            else:
                sample = {}
                sample["image_path"] = os.path.join(image_path, (item[0:6] + ".png"))
                sample["label_path"] = os.path.join(label_path, (item[0:6] + ".png"))
                self.examples.append(sample)

    def __getitem__(self, index):
        img_path = self.examples[index]["image_path"]
        lab_path = self.examples[index]["label_path"]

        IMg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = IMg[:, :, 0:3]
        the = IMg[:, :, 3]
        lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE) + 1

        trans_img = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        trans_the = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=(0.449), std=(0.226))
        ])

        trans_jit = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=(0.8, 1.2))
        ])

        scales = [0.5, 0.75, 0.95, 1, 1.25, 1.5, 1.75]
        if self.mode == "train":
            S = scales[random.randrange(0, 7)]
            width = int(self.w_image * S)
            height = int(self.h_image * S)

            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            the = cv2.resize(the, (width, height), interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize(lab, (width, height), interpolation=cv2.INTER_NEAREST)

        lab = lab.reshape((lab.shape[0], lab.shape[1], 1))
        the = the.reshape((the.shape[0], the.shape[1], 1))

        trans_totensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

        img = trans_totensor(img)
        the = trans_totensor(the)
        lab = trans_totensor(lab) * 255

        if self.mode == "train":
            concat_image = torch.cat((img, the, lab), dim=0)
            concat_image = self.transform(concat_image)
            img = concat_image[0:3]
            img = trans_jit(img)
            the = concat_image[3]
            the = trans_jit(the)
            the = torch.reshape(the, (1, the.shape[0], the.shape[1]))
            lab = concat_image[4]

        else:
            lab = torch.reshape(lab, (lab.shape[1], lab.shape[2]))

        the = trans_the(the)
        img = trans_img(img)

        return (img, the, lab)

    def __len__(self):
        return len(self.examples)

class Data_ZJU(torch.utils.data.Dataset):
    def __init__(self, img_folder, transform=None, mode="test"):
        self.transform1 = transform
        self.mode = mode

        self.w_image = 612
        self.h_image = 512

        if self.mode == "train":
            fp_label = os.path.join(img_folder, 'train_label')
        else:
            fp_label = os.path.join(img_folder, 'val_label')

        fp_img0 = os.path.join(img_folder, "0")
        fp_img45 = os.path.join(img_folder, "45")
        fp_img90 = os.path.join(img_folder, "90")
        fp_img135 = os.path.join(img_folder, "135")

        self.examples = []
        files = os.listdir(fp_img0)
        for item in files:
            sample = {}
            sample["image_path0"] = os.path.join(fp_img0, item)
            sample["image_path45"] = os.path.join(fp_img45, (item[0:51] + "_45.png"))
            sample["image_path90"] = os.path.join(fp_img90, (item[0:51] + "_90.png"))
            sample["image_path135"] = os.path.join(fp_img135, (item[0:51] + "_135.png"))
            sample["label_path"] = os.path.join(fp_label, (item[0:51] + ".png"))
            self.examples.append(sample)

    def __getitem__(self, index):
        img_path1 = self.examples[index]["image_path0"]
        img_path2 = self.examples[index]["image_path45"]
        img_path3 = self.examples[index]["image_path90"]
        img_path4 = self.examples[index]["image_path135"]
        lab_path = self.examples[index]["label_path"]

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        img3 = cv2.imread(img_path3)
        img4 = cv2.imread(img_path4)
        lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)

        if self.mode == "train":
            if np.random.rand() > 0.5:
                s1 = img1 - img3
                s2 = img4 - img2
                s1 = s1.astype(np.float32)
                s2 = s2.astype(np.float32)
                s2[s2 == 0] = 0.000000001
                AoLP = 0.5 * np.arctan(np.divide(s1, s2))
                img1 = cv2.flip(img1, 1)
                lab = cv2.flip(lab, 1)
                AoLP = cv2.flip(AoLP, 1)

            else:
                s1 = img1 - img3
                s2 = img2 - img4
                s1 = s1.astype(np.float32)
                s2 = s2.astype(np.float32)
                s2[s2 == 0] = 0.000000001
                AoLP = 0.5 * np.arctan(np.divide(s1, s2))
        else:
            s1 = img1 - img3
            s2 = img2 - img4
            s1 = s1.astype(np.float32)
            s2 = s2.astype(np.float32)
            s2[s2 == 0] = 0.000000001
            AoLP = 0.5 * np.arctan(np.divide(s1, s2))

        trans_img = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        trans_the = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=(0.449), std=(0.226))
        ])

        trans_jit = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=(0.8, 1.2))
        ])

        scales = [0.5, 0.75, 0.98, 1, 1.25, 1.5, 1.75]
        if self.mode == "train":
            S = scales[random.randrange(0, 7)]
            width = int(self.w_image * S)
            height = int(self.h_image * S)

            img = cv2.resize(img1, (width, height), interpolation=cv2.INTER_LINEAR)
            AoLP = cv2.resize(AoLP, (width, height), interpolation=cv2.INTER_NEAREST)
            lab = cv2.resize(lab, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img1, (self.w_image, self.h_image), interpolation=cv2.INTER_LINEAR)
            AoLP = cv2.resize(AoLP, (self.w_image, self.h_image), interpolation=cv2.INTER_NEAREST)
            lab = cv2.resize(lab, (self.w_image, self.h_image), interpolation=cv2.INTER_NEAREST)

        lab = lab.reshape((lab.shape[0], lab.shape[1], 1))

        trans_totensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

        img = trans_totensor(img)
        AoLP = trans_totensor(AoLP)
        lab = trans_totensor(lab) * 255

        if self.mode == "train":
            concat_image = torch.cat((img, AoLP, lab), dim=0)
            concat_image = self.transform1(concat_image)
            img = concat_image[0:3]
            img = trans_jit(img)
            AoLP = concat_image[3:6]
            lab = concat_image[6]

        else:
            lab = torch.reshape(lab, (lab.shape[1], lab.shape[2]))

        AoLP = trans_the(AoLP).type(torch.float)
        img = trans_img(img).type(torch.float)

        return (img, AoLP, lab)

    def __len__(self):
        return len(self.examples)



class Data_FMB(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None, mode="test"):
        self.transform = transform
        self.mode = mode

        self.w_image = 800
        self.h_image = 600

        rgb_path = os.path.join(folder_path, "Visible")
        thermal_path = os.path.join(folder_path, "Infrared")
        label_path = os.path.join(folder_path, "Label")

        self.examples = []
        folder = os.listdir(rgb_path)

        for imgs in folder:
            img_dir = os.path.join(rgb_path, imgs)
            thermal_dir = os.path.join(thermal_path, imgs)
            label_dir = os.path.join(label_path, imgs)

            sample = {}
            sample["image_path"] = img_dir
            sample["thermal_path"] = thermal_dir
            sample["label_path"] = label_dir
            self.examples.append(sample)

    def __getitem__(self, index):
        img_path = self.examples[index]["image_path"]
        the_path = self.examples[index]["thermal_path"]
        lab_path = self.examples[index]["label_path"]

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        the = cv2.imread(the_path, cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)

        trans_img = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        trans_the = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=(0.449), std=(0.226))
        ])

        trans_jit = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=(0.8, 1.2))
        ])

        scales = [0.5, 0.75, 0.95, 1, 1.25, 1.5, 1.75]
        if self.mode == "train":
            S = scales[random.randrange(0, 7)]
            width = int(self.w_image * S)
            height = int(self.h_image * S)

            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            the = cv2.resize(the, (width, height), interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize(lab, (width, height), interpolation=cv2.INTER_NEAREST)

        lab = lab.reshape((lab.shape[0], lab.shape[1], 1))
        the = the.reshape((the.shape[0], the.shape[1], 1))

        trans_totensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

        img = trans_totensor(img)
        the = trans_totensor(the)
        lab = trans_totensor(lab) * 255

        if self.mode == "train":
            concat_image = torch.cat((img, the, lab), dim=0)
            concat_image = self.transform(concat_image)
            img = concat_image[0:3]
            img = trans_jit(img)
            the = concat_image[3]
            the = trans_jit(the)
            the = torch.reshape(the, (1, the.shape[0], the.shape[1]))
            lab = concat_image[4]

        else:
            lab = torch.reshape(lab, (lab.shape[1], lab.shape[2]))

        the = trans_the(the)
        img = trans_img(img)

        return (img, the, lab)

    def __len__(self):
        return len(self.examples)