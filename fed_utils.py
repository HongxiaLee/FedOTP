import torch
import copy
from prettytable import PrettyTable

def average_weights(w,idxs_users,datanumber_client,islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])
    
    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points
        
        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg


def count_parameters(model,model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:
            # if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params