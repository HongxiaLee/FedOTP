# This code is used to generate non-iid data with Feature Skew

import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import Counter

class Datum:
    """Data instance which defines the basic attributes.

    Args:
        data (float): data.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath, label=0, domain=0, classname=""):
        # assert isinstance(impath, str)
        # assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


def prepare_data_domainNet(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    # clipart
    clipart_trainset = DomainNetDataset(data_base_path, 'clipart', transform=transform_train)
    lab2cname = clipart_trainset.lab2cname
    classnames = clipart_trainset.classnames
    clipart_trainset = clipart_trainset.data_detailed
    clipart_testset = DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False).data_detailed
    # infograph
    infograph_trainset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train).data_detailed
    infograph_testset = DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False).data_detailed
    # painting
    painting_trainset = DomainNetDataset(data_base_path, 'painting', transform=transform_train).data_detailed
    painting_testset = DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False).data_detailed
    # quickdraw
    quickdraw_trainset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train).data_detailed
    quickdraw_testset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False).data_detailed
    # real
    real_trainset = DomainNetDataset(data_base_path, 'real', transform=transform_train).data_detailed
    real_testset = DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False).data_detailed
    # sketch
    sketch_trainset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train).data_detailed
    sketch_testset = DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False).data_detailed

    # min_data_len = min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset))
    # print("Train data number: ", min_data_len)
    train_data_num_list = [len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset)]
    test_data_num_list = [len(clipart_testset), len(infograph_testset), len(painting_testset), len(quickdraw_testset), len(real_testset), len(sketch_testset)]
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    train_set = [clipart_trainset, infograph_trainset, painting_trainset, quickdraw_trainset, real_trainset, sketch_trainset]
    test_set = [clipart_testset, infograph_testset, painting_testset, quickdraw_testset, real_testset, sketch_testset]

    return train_set, test_set, classnames, lab2cname


def prepare_data_domainNet_partition_train(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    min_require_size = 5
    n_clients = 5
    print("Clipart: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'clipart', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_domainnet('clipart', cfg.DATASET.BETA, split_test=False, n_parties=cfg.DATASET.USERS, min_require_size=2)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('clipart', train_ratio[0])
    clipart_trainset = DomainNetDataset_sub(data_base_path, 'clipart', net_dataidx_map_train, transform=transform_train)
    clipart_testset = DomainNetDataset_sub(data_base_path, 'clipart', net_dataidx_map_test, transform=transform_test, train=False).data_detailed
    lab2cname = clipart_trainset.lab2cname
    classnames = clipart_trainset.classnames
    clipart_trainset = clipart_trainset.data_detailed
    
    print("Infograph: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet1(data_base_path, 'infograph', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('infograph', train_ratio[0])
    infograph_trainset = DomainNetDataset_sub(data_base_path, 'infograph', net_dataidx_map_train, transform=transform_train).data_detailed
    infograph_testset = DomainNetDataset_sub(data_base_path, 'infograph', net_dataidx_map_test, transform=transform_test, train=False).data_detailed
    
    print("Painting: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet1(data_base_path, 'painting', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('painting', train_ratio[2])
    painting_trainset = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_train, transform=transform_train).data_detailed
    painting_testset = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_test, transform=transform_test, train=False).data_detailed
    
    print("Quickdraw: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet1(data_base_path, 'quickdraw', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('quickdraw', train_ratio[1])
    quickdraw_trainset = DomainNetDataset_sub(data_base_path, 'quickdraw', net_dataidx_map_train, transform=transform_train).data_detailed
    quickdraw_testset = DomainNetDataset_sub(data_base_path, 'quickdraw', net_dataidx_map_test, transform=transform_test, train=False).data_detailed
    
    print("Real")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet1(data_base_path, 'real', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('real', train_ratio[0])
    real_trainset = DomainNetDataset_sub(data_base_path, 'real', net_dataidx_map_train, transform=transform_train).data_detailed
    real_testset = DomainNetDataset_sub(data_base_path, 'real', net_dataidx_map_test, transform=transform_test, train=False).data_detailed
    
    print("Sketch")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet1(data_base_path, 'sketch', cfg.DATASET.BETA, n_parties=cfg.DATASET.USERS, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('sketch', train_ratio[0])
    sketch_trainset = DomainNetDataset_sub(data_base_path, 'sketch', net_dataidx_map_train, transform=transform_train).data_detailed
    sketch_testset = DomainNetDataset_sub(data_base_path, 'sketch', net_dataidx_map_test, transform=transform_test, train=False).data_detailed

    train_data_num_list = [len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset)]
    test_data_num_list = [len(clipart_testset), len(infograph_testset), len(painting_testset), len(quickdraw_testset), len(real_testset), len(sketch_testset)]
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    train_set = [clipart_trainset, infograph_trainset, painting_trainset, quickdraw_trainset, real_trainset, sketch_trainset]
    test_set = [clipart_testset, infograph_testset, painting_testset, quickdraw_testset, real_testset, sketch_testset]

    return train_set, test_set, classnames, lab2cname

def prepare_data_domainNet_partition_client_train(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    min_require_size = 2
    n_clients = 5
    print("Clipart: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'clipart', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_domainnet('clipart', cfg.DATASET.BETA, split_test=False, n_parties=cfg.DATASET.USERS, min_require_size=2)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('clipart', train_ratio[0])
    clipart_trainset = [[] for i in range(n_clients)]
    clipart_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        clipart_trainset[i] = DomainNetDataset_sub(data_base_path, 'clipart', net_dataidx_map_train[i], transform=transform_train)
        clipart_testset[i] = DomainNetDataset_sub(data_base_path, 'clipart', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed
        lab2cname = clipart_trainset[i].lab2cname
        classnames = clipart_trainset[i].classnames
        clipart_trainset[i] = clipart_trainset[i].data_detailed
    
    print("Infograph: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'infograph', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('infograph', train_ratio[0])
    infograph_trainset = [[] for i in range(n_clients)]
    infograph_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        infograph_trainset[i] = DomainNetDataset_sub(data_base_path, 'infograph', net_dataidx_map_train[i], transform=transform_train).data_detailed
        infograph_testset[i] = DomainNetDataset_sub(data_base_path, 'infograph', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed
    
    print("Painting: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'painting', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('painting', train_ratio[2])
    painting_trainset = [[] for i in range(n_clients)]
    painting_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        painting_trainset[i] = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_train[i], transform=transform_train).data_detailed
        painting_testset[i] = DomainNetDataset_sub(data_base_path, 'painting', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed
    
    print("Quickdraw: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'quickdraw', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('quickdraw', train_ratio[1])
    quickdraw_trainset = [[] for i in range(n_clients)]
    quickdraw_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        quickdraw_trainset[i] = DomainNetDataset_sub(data_base_path, 'quickdraw', net_dataidx_map_train[i], transform=transform_train).data_detailed
        quickdraw_testset[i] = DomainNetDataset_sub(data_base_path, 'quickdraw', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed
    
    print("Real")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'real', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('real', train_ratio[0])
    real_trainset = [[] for i in range(n_clients)]
    real_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        real_trainset[i] = DomainNetDataset_sub(data_base_path, 'real', net_dataidx_map_train[i], transform=transform_train).data_detailed
        real_testset[i] = DomainNetDataset_sub(data_base_path, 'real', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed
    
    print("Sketch")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domainnet(data_base_path, 'sketch', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_test = Adjust_test_dataset_domainnet('sketch', train_ratio[0])
    sketch_trainset = [[] for i in range(n_clients)]
    sketch_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        sketch_trainset[i] = DomainNetDataset_sub(data_base_path, 'sketch', net_dataidx_map_train[i], transform=transform_train).data_detailed
        sketch_testset[i] = DomainNetDataset_sub(data_base_path, 'sketch', net_dataidx_map_test[i], transform=transform_test, train=False).data_detailed

    train_data_num_list = []
    test_data_num_list = []
    train_set = []
    test_set = []
    for dataset in [clipart_trainset, infograph_trainset, painting_trainset, quickdraw_trainset, real_trainset, sketch_trainset]:
        for i in range(n_clients):
            train_data_num_list.append(len(dataset[i]))
            train_set.append(dataset[i])
    for dataset in [clipart_testset, infograph_testset, painting_testset, quickdraw_testset, real_testset, sketch_testset]:
        for i in range(n_clients):
            test_data_num_list.append(len(dataset[i]))
            test_set.append(dataset[i])
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    return train_set, test_set, classnames, lab2cname

def prepare_data_office(cfg, data_base_path):
    data_base_path = data_base_path
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    lab2cname = amazon_trainset.lab2cname
    classnames = amazon_trainset.classnames
    amazon_trainset = amazon_trainset.data_detailed
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False).data_detailed
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office).data_detailed
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False).data_detailed
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office).data_detailed
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False).data_detailed
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office).data_detailed
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False).data_detailed

    train_data_num_list = [len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset)]
    test_data_num_list = [len(amazon_testset), len(caltech_testset), len(dslr_testset), len(webcam_testset)]
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    train_set =  [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]
    test_set = [amazon_testset, caltech_testset, dslr_testset, webcam_testset]

    return train_set, test_set, classnames, lab2cname


def prepare_data_office_partition_train(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])

    K = 10
    min_require_size = 2
    n_clients = 3
    # amazon
    print("Amazon: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_office(data_base_path, 'amazon', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # net_dataidx_map_train, _, train_ratio, _ = Dataset_partition_office(data_base_path,'amazon', cfg.DATASET.BETA, split_test=False, n_parties=cfg.DATASET.USERS, min_require_size=min_img_num)
    # net_dataidx_map_test = Adjust_test_dataset_office('amazon', train_ratio[0])
    amazon_trainset = [[] for i in range(n_clients)]
    amazon_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        amazon_trainset[i] = OfficeDataset_sub(data_base_path, 'amazon', net_dataidx_map_train[i], transform=transform_train)
        amazon_testset[i] = OfficeDataset_sub(data_base_path, 'amazon', net_dataidx_map_test[i], train=False, transform=transform_test).data_detailed
        lab2cname = amazon_trainset[i].lab2cname
        classnames = amazon_trainset[i].classnames
        amazon_trainset[i] = amazon_trainset[i].data_detailed

    # caltech
    print("Caltech: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_office(data_base_path, 'caltech', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # caltech_trainset = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_train, transform=transform_train).data_detailed
    # caltech_testset = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_test, transform=transform_train).data_detailed
    caltech_trainset = [[] for i in range(n_clients)]
    caltech_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        caltech_trainset[i] = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_train[i], transform=transform_train).data_detailed
        caltech_testset[i] = OfficeDataset_sub(data_base_path, 'caltech', net_dataidx_map_test[i], train=False, transform=transform_test).data_detailed

    
    # dslr
    print("dslr: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_office(data_base_path, 'dslr', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # dslr_trainset = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_train, transform=transform_train).data_detailed
    # dslr_testset = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_test, transform=transform_train).data_detailed
    dslr_trainset = [[] for i in range(n_clients)]
    dslr_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        dslr_trainset[i] = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_train[i], transform=transform_train).data_detailed
        dslr_testset[i] = OfficeDataset_sub(data_base_path, 'dslr', net_dataidx_map_test[i], train=False, transform=transform_test).data_detailed
    
    
    # webcam
    print("Webcam: ")
    net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_office(data_base_path, 'webcam', cfg.DATASET.BETA, n_parties=n_clients, min_require_size=min_require_size)
    # webcam_trainset = OfficeDataset_sub(data_base_path, 'webcam', net_dataidx_map_train, transform=transform_train).data_detailed
    # webcam_testset = OfficeDataset_sub(data_base_path, 'webcam', net_dataidx_map_test, transform=transform_train).data_detailed
    webcam_trainset = [[] for i in range(n_clients)]
    webcam_testset = [[] for i in range(n_clients)]
    for i in range(n_clients):
        webcam_trainset[i] = OfficeDataset_sub(data_base_path, 'webcam', net_dataidx_map_train[i], transform=transform_train).data_detailed
        webcam_testset[i] = OfficeDataset_sub(data_base_path, 'webcam', net_dataidx_map_test[i], train=False, transform=transform_test).data_detailed

    train_data_num_list = []
    test_data_num_list = []
    train_set = []
    test_set = []
    for dataset in [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]:
        for i in range(n_clients):
            train_data_num_list.append(len(dataset[i]))
            train_set.append(dataset[i])
    for dataset in [amazon_testset, caltech_testset, dslr_testset, webcam_testset]:
        for i in range(n_clients):
            test_data_num_list.append(len(dataset[i]))
            test_set.append(dataset[i])
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    return train_set, test_set, classnames, lab2cname


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        self.base_path = base_path
        if train:
            path = os.path.join(self.base_path, 'office_caltech_10/{}_train.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
        else:
            path = os.path.join(self.base_path, 'office_caltech_10/{}_test.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
            
        self.site_domian = {'amazon':0, 'caltech':1, 'dslr':2, 'webcam':3}
        self.domain = self.site_domian[site]
        self.lab2cname={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.classnames ={'back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector'}
        self.target = [self.lab2cname[text] for text in self.label]
        if train:
            print('Counter({}_train data:)'.format(site), Counter(self.target))
        else:
            print('Counter({}_test data:)'.format(site), Counter(self.target))
        self.label = self.label.tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.target)
    
    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            img_path = os.path.join(self.base_path, self.paths[i])
            data_idx = img_path
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=int(self.domain), classname=label_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.target[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        self.base_path = base_path
        if train:
            path = os.path.join(self.base_path,'DomainNet/{}_train.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
        else:
            path = os.path.join(self.base_path,'DomainNet/{}_test.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
            
        self.site_domian = {'clipart':0, 'infograph':1, 'painting':2, 'quickdraw':3, 'real':4, 'sketch':5}
        self.domain = self.site_domian[site]
        self.lab2cname = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9} 
        self.classnames = {'bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass', 'zebra'}    
        self.target = [self.lab2cname[text] for text in self.label]
        if train:
            print('Counter({}_train data:)'.format(site), Counter(self.target))
        else:
            print('Counter({}_test data:)'.format(site), Counter(self.target))
        self.label = self.label.tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.target)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            img_path = os.path.join(self.base_path, self.paths[i])
            data_idx = img_path
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=int(self.domain), classname=label_idx)
            data_with_label.append(item)
        return data_with_label
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.target[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


def Dataset_partition_domainnet(base_path, site, beta, n_parties=5, min_require_size=2):
    min_size = 0
    K = 10
    train_path = os.path.join(base_path,'DomainNet/{}_train.pkl'.format(site))
    test_path = os.path.join(base_path,'DomainNet/{}_test.pkl'.format(site))
    _, train_text_labels = np.load(train_path, allow_pickle=True)
    _, test_text_labels = np.load(test_path, allow_pickle=True)

    label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9} 
    # train_labels2 = [label_dict[text] for text in train_text_labels]    
    train_labels = np.asarray([label_dict[text] for text in train_text_labels])
    test_labels = np.asarray([label_dict[text] for text in test_text_labels])
    N_train = train_labels.shape[0]
    N_test = test_labels.shape[0]
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        idx_batch_test = [[] for _ in range(n_parties)]
        for k in range(K):
            train_idx_k = np.where(train_labels == k)[0]
            test_idx_k = np.where(test_labels == k)[0]
            np.random.seed(0)
            np.random.shuffle(train_idx_k)
            np.random.shuffle(test_idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
            proportions = proportions / proportions.sum()
            # proportions = proportions * 2
            proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
            train_part_list = np.split(train_idx_k, proportions_train)
            test_part_list = np.split(test_idx_k, proportions_test)
            idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, train_part_list)]
            idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, test_part_list)]   

            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
            min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
            min_size = min(min_size_test,min_size_train)

    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map_train)
    print(site, "Training data split: ",traindata_cls_counts)
    testdata_cls_counts = record_net_data_stats(test_labels, net_dataidx_map_test)
    print(site, "Testing data split: ",testdata_cls_counts)
    return net_dataidx_map_train, net_dataidx_map_test

def Dataset_partition_office(base_path, site, beta, split_test=True, n_parties=3, min_require_size=2):
    min_size = 0
    K = 10
    # np.random.seed(2023)

    train_path = os.path.join(base_path,'office_caltech_10/{}_train.pkl'.format(site))
    test_path = os.path.join(base_path,'office_caltech_10/{}_test.pkl'.format(site))
    _, train_text_labels = np.load(train_path, allow_pickle=True)    
    _, test_text_labels = np.load(test_path, allow_pickle=True)    

    label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}     
    train_labels = np.asarray([label_dict[text] for text in train_text_labels])
    test_labels = np.asarray([label_dict[text] for text in test_text_labels])
    N_train = train_labels.shape[0]
    N_test = test_labels.shape[0]
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        idx_batch_test = [[] for _ in range(n_parties)]
        for k in range(K):
            train_idx_k = np.where(train_labels == k)[0]
            test_idx_k = np.where(test_labels == k)[0]
            np.random.seed(0)
            np.random.shuffle(train_idx_k)
            np.random.shuffle(test_idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
            proportions = proportions / proportions.sum()
            # proportions = proportions * 2
            proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
            train_part_list = np.split(train_idx_k, proportions_train)
            test_part_list = np.split(test_idx_k, proportions_test)
            idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, train_part_list)]
            idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, test_part_list)]   

            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
            min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
            min_size = min(min_size_test,min_size_train)

    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map_train)
    print(site, "Training data split: ",traindata_cls_counts)
    testdata_cls_counts = record_net_data_stats(test_labels, net_dataidx_map_test)
    print(site, "Testing data split: ",testdata_cls_counts)
    return net_dataidx_map_train, net_dataidx_map_test


def Adjust_test_dataset_domainnet(site, class_ratio):
    c_num = 10
    label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
    _, test_text_labels = np.load('/public/home/lihx2/DATA/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)
    test_labels = np.asarray([label_dict[text] for text in test_text_labels])
    unq, unq_cnt = np.unique([test_labels[x] for x in range(len(test_labels))], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    times = [test_class_ratio[x]/class_ratio[x] for x in range(c_num)]
    min_time = min(times)
    right_class_num = [int(min_time*class_ratio[x]) for x in range(c_num)]
    idx = []
    for k in range(c_num):
        test_idx_k = np.where(test_labels == k)[0]
        np.random.shuffle(test_idx_k)
        idx = idx + test_idx_k[:right_class_num[k]].tolist()
    unq, unq_cnt = np.unique([test_labels[x] for x in idx], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    return idx


def Adjust_test_dataset_office(site, class_ratio):
    c_num = 10
    label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
    _, test_text_labels = np.load('data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
    test_labels = np.asarray([label_dict[text] for text in test_text_labels])
    unq, unq_cnt = np.unique([test_labels[x] for x in range(len(test_labels))], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    times = [test_class_ratio[x]/class_ratio[x] for x in range(c_num)]
    min_time = min(times)
    right_class_num = [int(min_time*class_ratio[x]) for x in range(c_num)]
    idx = []
    for k in range(c_num):
        test_idx_k = np.where(test_labels == k)[0]
        np.random.shuffle(test_idx_k)
        idx = idx + test_idx_k[:right_class_num[k]].tolist()
    unq, unq_cnt = np.unique([test_labels[x] for x in idx], return_counts=True)
    test_class_ratio = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    return idx


class OfficeDataset_sub(Dataset):
    def __init__(self, base_path, site, net_dataidx_map, train=True, transform=None):
        self.base_path = base_path
        if train:
            path = os.path.join(self.base_path, 'office_caltech_10/{}_train.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
        else:
            path = os.path.join(self.base_path, 'office_caltech_10/{}_test.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
            
        self.site_domian = {'amazon':0, 'caltech':1, 'dslr':2, 'webcam':3}
        self.domain = self.site_domian[site]
        self.lab2cname={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.classnames ={'back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector'}
        self.target = np.asarray([self.lab2cname[text] for text in self.label])
        self.target = self.target[net_dataidx_map]
        self.label = self.label.tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def second_divide(self, partitions):
        self.target = self.target[partitions]

    def __len__(self):
        return len(self.target)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            img_path = os.path.join(self.base_path, self.paths[i])
            data_idx = img_path
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=int(self.domain), classname=label_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.target[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset_sub(Dataset):
    def __init__(self, base_path, site, net_dataidx_map, train=True, transform=None):
        self.base_path = base_path
        if train:
            path = os.path.join(self.base_path,'DomainNet/{}_train.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
        else:
            path = os.path.join(self.base_path,'DomainNet/{}_test.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
            
        self.site_domian = {'clipart':0, 'infograph':1, 'painting':2, 'quickdraw':3, 'real':4, 'sketch':5}
        self.domain = self.site_domian[site]
        self.lab2cname = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        self.classnames = {'bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass', 'zebra'}
        self.target = np.asarray([self.lab2cname[text] for text in self.label])
        self.target = self.target[net_dataidx_map]
        self.label = self.label.tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def second_divide(self, partitions):
        self.target = self.target[partitions]

    def __len__(self):
        return len(self.target)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            img_path = os.path.join(self.base_path, self.paths[i])
            data_idx = img_path
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=int(self.domain), classname=label_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.target[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

