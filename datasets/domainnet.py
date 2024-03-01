import os

from utils.data_utils import prepare_data_domainNet, prepare_data_domainNet_partition_train, prepare_data_domainNet_partition_client_train

# @DATASET_REGISTRY.register()
class DomainNet():
    dataset_dir = "domainnet"
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 10

        if cfg.DATASET.IMBALANCE_TRAIN:
            exp_folder = 'fed_domainnet_label_skew'
            print("label skew in Train")
            print("Dirichlet alpha value: ", cfg.DATASET.BETA)
            # print("min image number in each class: ", 2)
            print("Divide into %d fold" % cfg.DATASET.USERS)
            if cfg.DATASET.SPLIT_CLIENT:
                train_set, test_set, classnames, lab2cname = prepare_data_domainNet_partition_client_train(cfg, root)
            else:
                train_set, test_set, classnames, lab2cname = prepare_data_domainNet_partition_train(cfg, root)
        else:
            exp_folder = 'fed_domainnet'
            print("No label skew")
            train_set, test_set, classnames, lab2cname = prepare_data_domainNet(cfg, root)
    

        self.federated_train_x = train_set
        self.federated_test_x = test_set
        self.lab2cname = lab2cname
        self.classnames = classnames



