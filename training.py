import os
import pandas as pd
import numpy as np
import torch
import torch_scatter
import random

from torch_geometric.data import Dataset
from torch_geometric.loader import NodeLoader, DataLoader
import json
import argparse

from GNN_network import *
from training_utils import *
from data_utils import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_path = 'datasets/dataset_rndshapSM_rndbc_fixobj_0.65_0.07/gnn_skipBCFC_GP_12l_32l_1500/cfg.json'

with open(cfg_path , 'r') as f:
    cfg = json.load(f)


dataset_cfg = cfg['dataset_cfg']
nn_cfg = cfg['nn_cfg']
train_cfg = cfg['train_cfg']

if 'out_dir' in cfg:
    out_dir = cfg['out_dir']

fc_in_dim = nn_cfg['fc_in_dim']
fc_out_dim = nn_cfg['fc_out_dim']
out_dim = nn_cfg['out_dim']
edge_feature_dims = nn_cfg['edge_feature_dims']
num_filters = nn_cfg['num_filters']

train_ratio = train_cfg['train_ratio'] 
valid_ratio = train_cfg['valid_ratio']
num_epochs = train_cfg['num_epochs']

decay_factor = train_cfg['decay_factor']
decay_step = train_cfg['decay_step']
batch_size = train_cfg['batch_size']

skipcon = nn_cfg['skipcon_indx']
edge_skipcon = nn_cfg['edge_skipcon_indx']
fc_skipcon = nn_cfg['fc_skipcon_indx']
hidden_size = nn_cfg["hidden_size"]

batchnorm_layers = nn_cfg["batchnorm"]
activations_layers = nn_cfg["activation"]
msg_layers = nn_cfg["msg_norm"]
selu = nn_cfg["selu"]
loss = nn_cfg["loss"]


datasetT = makeDataset(dataset_cfg['dataset_dir'],dataset_cfg['dataset_source_len'],  
                      do_read = True, dataset_name=dataset_cfg['dataset_name'])

datasetV = makeDataset(dataset_cfg['val_dataset_dir'], dataset_cfg['val_dataset_source_len'],  
                      do_read = True, dataset_name=dataset_cfg['val_dataset_name'])

train_size = int(dataset_cfg['dataset_source_len'] * train_ratio)
val_size  = int(dataset_cfg['dataset_source_len'] * valid_ratio)
train_dataset = datasetT[:train_size]
val_dataset = datasetV[:val_size]
test_dataset = datasetV[val_size:]

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

#gnn_model = FlowGNN_withFC(edge_feat_dims=edge_feature_dims, num_filters=num_filters,fc_in_dim=fc_in_dim, 
                          # fc_out_dim=fc_out_dim, out_dim=4)
print(train_dataset[0])
#print(next(iter(train_dataloader)))    

#gnn_model = FlowGNN_skipBC(edge_feat_dims=edge_feature_dims, num_filters=num_filters, skipcon_indx = skipcon, out_dim=4)


# num_layers = nn_cfg["num_layers"]
# gnn_model = FlowGNN_with_FC_BN_RC(hidden_size, num_layers, fc_in_dim=fc_in_dim,  
#                             fc_out_dim=fc_out_dim, fcskipcon_indx = skipcon, out_dim=4)


# gnn_model = FlowGNN_deepmod(edge_feature_dims, num_filters, fc_in_dim, fc_out_dim,  SC_list=skipcon, 
#                  fc_skip_indx=fc_skipcon, batchnorm = batchnorm_layers, activations=activations_layers, selu = selu, msg=msg_layers, loss_func=loss)

gnn_model = FlowGNN_deepmod(edge_feature_dims, num_filters, fc_in_dim, fc_out_dim,  SC_list=skipcon, SC_ed_list = edge_skipcon,
                 fc_skip_indx=fc_skipcon, batchnorm = batchnorm_layers, activations=activations_layers, msg=msg_layers, selu=selu, loss_func=loss)


# gnn_model = FlowGNN_with_FC_deep(edge_feat_dims=edge_feature_dims, num_filters=num_filters,fc_in_dim=fc_in_dim,  
#                             fc_out_dim=fc_out_dim, fcskipcon_indx = fcskipcon, out_dim=4)

# gnn_model = FlowGNN_with_FC_pool(edge_feat_dims=edge_feature_dims, num_filters=num_filters,fc_in_dim=fc_in_dim, 
#                             fc_out_dim=fc_out_dim, out_dim=4)
#gnn_model = FlowGNN(edge_feat_dims=edge_feature_dims, num_filters=num_filters, out_dim=3)

# model_save_pth = '/home/vlad/gnn_diplom/shapeopt-fields_predict-gnn_laminar_flow/GNN_PytorchGeo/datasets/dataset_IrregMesh_rndshap_rndbc_fixobj_0.65_0.07/gnn_FC/best_model.pt'
# gnn_model.load_state_dict(torch.load(model_save_pth))


optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = decay_step, gamma = decay_factor)
train_loss_hist, val_loss_hist, grads = train_model(gnn_model, train_dataloader, val_dataloader, device, 2, optimizer, scheduler, cfg, collect_grads=True)
print(grads)