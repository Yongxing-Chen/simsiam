import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
torch.manual_seed(2022)
np.random.seed(2022)


class FewShotCifar10(torchvision.datasets.CIFAR10):
    def __init__(self, root, mode, transform, n_shot):
        if mode in ("train", "val"):
            train_sign = True
        else:
            train_sign = False
        super().__init__(root, train_sign, transform, download=True)
        self.mode = mode
        if train_sign:
            self.n_shot = n_shot
            self.transform = transform
            # create training and val data
            train_data = []
            train_y = []
            val_data = []
            val_y = []
            indices = np.random.permutation(50000)
            data = self.data[indices]
            targets = np.array(self.targets)
            targets = targets[indices]
            for i in range(10):
                class_data = data[targets == i]
                train_data.append(class_data[:n_shot])
                train_y.append(np.ones((n_shot, )) * i)
                val_data.append(class_data[n_shot:])
                val_y.append(np.ones((5000 - n_shot, )) * i)
            self.train_data = np.concatenate(train_data, axis=0)
            self.val_data = np.concatenate(val_data, axis=0)
            self.train_y = np.concatenate(train_y, axis=0).astype(np.compat.long)
            self.val_y = np.concatenate(val_y, axis=0).astype(np.compat.long)
            indices = np.random.permutation(n_shot * 10)
            self.train_data = self.train_data[indices]
            self.train_y = self.train_y[indices]
            self.train_transform = transform
            self.val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
            ])
            del self.data
            del self.targets


    def __getitem__(self, index):
        if self.mode == "train":
            img, target = self.train_data[index], self.train_y[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.train_transform is not None:
                img = self.train_transform(img)

        elif self.mode == "val":
            img, target = self.val_data[index], self.val_y[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.train_transform is not None:
                img = self.val_transform(img)
        else:
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "val":
            return len(self.val_data)
        else:
            return len(self.data)


class FewShotCifar100(torchvision.datasets.CIFAR100):
    def __init__(self, root, mode, transform, n_shot):
        if mode in ("train", "val"):
            train_sign = True
        else:
            train_sign = False
        super().__init__(root, train_sign, transform, download=True)
        self.mode = mode
        if train_sign:
            self.n_shot = n_shot
            self.transform = transform
            # create training and val data
            train_data = []
            train_y = []
            val_data = []
            val_y = []
            indices = np.random.permutation(50000)
            data = self.data[indices]
            targets = np.array(self.targets)
            targets = targets[indices]
            for i in range(100):
                class_data = data[targets == i]
                train_data.append(class_data[:n_shot])
                train_y.append(np.ones((n_shot, )) * i)
                val_data.append(class_data[n_shot:])
                val_y.append(np.ones((5000 - n_shot, )) * i)
            self.train_data = np.concatenate(train_data, axis=0)
            self.val_data = np.concatenate(val_data, axis=0)
            self.train_y = np.concatenate(train_y, axis=0).astype(np.compat.long)
            self.val_y = np.concatenate(val_y, axis=0).astype(np.compat.long)
            indices = np.random.permutation(n_shot * 10)
            self.train_data = self.train_data[indices]
            self.train_y = self.train_y[indices]
            self.train_transform = transform
            self.val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                 std=[0.2675, 0.2565, 0.2761])
            ])
            del self.data
            del self.targets


    def __getitem__(self, index):
        if self.mode == "train":
            img, target = self.train_data[index], self.train_y[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.train_transform is not None:
                img = self.train_transform(img)

        elif self.mode == "val":
            img, target = self.val_data[index], self.val_y[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.train_transform is not None:
                img = self.val_transform(img)
        else:
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "val":
            return len(self.val_data)
        else:
            return len(self.data)


if __name__ == '__main__':
    import copy
    traindir = "./data/train"
    tf = torchvision.transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])

    dataset = FewShotCifar10(traindir, "train", tf, 10)
    print(dataset.train_y)
    dataset2 = copy.deepcopy(dataset)
    dataset2.mode = "val"
    print(len(dataset))
    print(len(dataset2))
    print(type(dataset[0][0]))
    print(type(dataset[0][1]))