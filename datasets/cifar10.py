import os

# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from Dassl.dassl.data.datasets import DatasetBase
from utils.datasplit import partition_data

# @DATASET_REGISTRY.register()
class Cifar10():
    dataset_dir = "cifar-10"
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 10

        federated_train_x = [[] for i in range(cfg.DATASET.USERS)]
        federated_test_x = [[] for i in range(cfg.DATASET.USERS)]

        data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts = partition_data(
            'cifar10', self.dataset_dir, cfg.DATASET.PARTITION, cfg.DATASET.USERS, beta=cfg.DATASET.BETA, logdir="./logs/")
        for net_id in range(cfg.DATASET.USERS):
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            for sample in range(len(dataidxs_train)):
                federated_train_x[net_id].append(data_train[dataidxs_train[sample]])
            for sample in range(len(dataidxs_test)):
                federated_test_x[net_id].append(data_test[dataidxs_test[sample]])

        self.federated_train_x = federated_train_x
        self.federated_test_x = federated_test_x
        self.lab2cname = lab2cname
        self.classnames = classnames



