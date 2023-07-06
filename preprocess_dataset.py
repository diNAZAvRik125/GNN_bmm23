import os.path as osp

from data_utils import make_dataset

if __name__ == "__main__":
    dataset_dir = 'datasets/dataset_rndshapSM_rndbc_fixobj_0.65_0.07'
    data_list = [
        [osp.join(dataset_dir, 'train'), 'train.pt'],
        [osp.join(dataset_dir, 'val'), 'val.pt']
    ]
    nodes_dim = 3  # x, y, obj

    for data_dir, dataset_name in data_list:
        dataset = make_dataset(data_dir, 3000, with_bc=True,
                               norm_flow=False, norm_coord=True, save=True, dataset_name=dataset_name,
                               nodes_dim=nodes_dim)
