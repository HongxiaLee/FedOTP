import os

# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from Dassl.dassl.data.datasets import DatasetBase
from datasplit import partition_data
from data_utils import prepare_data_office, prepare_data_office_partition_train

# @DATASET_REGISTRY.register()
class Office():
    dataset_dir = "office_caltech_10"
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 10

        if cfg.DATASET.IMBALANCE_TRAIN:
            exp_folder = 'fed_office_label_skew'
            print("label skew in Train")
            print("Dirichlet alpha value: ", cfg.DATASET.BETA)
            print("min image number in each class: ",2)
            print("Divide into %d fold" % cfg.DATASET.USERS)
            train_set, test_set, classnames, lab2cname = prepare_data_office_partition_train(cfg, root)
        else:
            exp_folder = 'fed_office'
            print("No label skew")
            train_set, test_set, classnames, lab2cname = prepare_data_office(cfg, root)

        self.federated_train_x = train_set
        self.federated_test_x = test_set
        self.lab2cname = lab2cname
        self.classnames = classnames



