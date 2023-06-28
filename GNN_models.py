import torch
from torch import nn

from torch_geometric.nn.pool.glob import global_mean_pool
from torch_geometric.nn import EdgePooling, DeepGCNLayer, GENConv, MessageNorm

from GNN_layers import ProcessorLayer, SmoothingLayer


class FlowGNN_original(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, geom_in_dim=2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_original, self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.processor = nn.ModuleList()

        self.decoder = nn.LazyLinear(out_dim)

        for ef, nf in zip(self.edge_feat_dims, self.num_filters):
            self.processor.append(ProcessorLayer(ef, nf, hidden_nodes))
            self.processor.append(SmoothingLayer())

    def forward(self, data):

        x, edge_index, edge_attr = data.x[:, :7], data.edge_index, data.edge_attr

        skip_info = x[:, :self.geom_in_dim]

        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred

    def loss(self, pred, inp):
        true_flow = inp.flow
        print(true_flow.shape, pred.shape)
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)


class FlowGNN_original_skipBC(nn.Module):  # напрямую прокидываютс параметры гран условий в вершины

    def __init__(self, edge_feat_dims, num_filters, skipcon_indx=[], geom_in_dim=2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_original_skipBC, self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.skipcon_indx = skipcon_indx

        self.processor = nn.ModuleList()

        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef, nf, hidden_nodes, idx=i))
            self.processor.append(SmoothingLayer(idx=i))

    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,
                                   :3], data.edge_index, data.edge_attr,  # здесь были data.bc, убрал для прямых гран.усл.

        bc = data.x[:, 3:]
        x = torch.cat([x, bc], 1)

        skip_info = x[:, :self.geom_in_dim]

        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)

                # if layer.name == 'smoothing' and layer.idx == 2:
                #     x = torch.cat([x, bc], 1)

                # if layer.name == 'smoothing' and layer.idx == 5:
                #     x = torch.cat([x, bc], 1)
                # if layer.idx%3 == 0:
                if layer.idx in self.skipcon_indx:  # if layer.idx%3 ==0
                    x = torch.cat([x, bc], 1)

        pred = self.decoder(x)
        return pred

    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)


class FlowGNN_conv_block(nn.Module):

    def __init__(self, edge_dims, node_dims, hidden_size=128, skipcon_nodes_idx=None, skipcon_edges_idx=None,
                 fc_skipcon=False, batchnorm=False, idx=0, selu=False):
        super(FlowGNN_conv_block, self).__init__()
        self.conv = ProcessorLayer(edge_dims, node_dims, hidden_size, selu)
        self.smooth = SmoothingLayer()

        self.node_norm_layer = None
        if batchnorm:
            self.node_norm_layer = nn.BatchNorm1d(node_dims)

        self.skipcon_nodes_idx = skipcon_nodes_idx  # int, layer indx
        self.skipcon_edges_idx = skipcon_edges_idx
        self.fc_skipcon = fc_skipcon  # bool, 
        self.idx = idx

    # TODO: rename arguments
    def forward(self, node_attr, edge_index, edge_attr, skip_connec=None, skip_connec_ed=None, fc_skipconnec=None,
                skip_info=None):

        if skip_connec is not None:
            node_attr = torch.add(node_attr, skip_connec)

        if skip_connec_ed is not None:
            edge_attr = torch.add(edge_attr, skip_connec_ed)

        if skip_info is not None:
            node_attr = torch.cat([node_attr, skip_info], 1)

        if self.fc_skipcon:
            node_attr = torch.cat([node_attr, fc_skipconnec], 1)

        node_attr, edge_attr = self.conv(node_attr, edge_index, edge_attr)
        node_attr, edge_attr = self.smooth(node_attr, edge_index, edge_attr)

        if self.node_norm_layer is not None:
            node_attr = self.node_norm_layer(node_attr)

        return node_attr, edge_attr


class FlowGNN_fc_block(nn.Module):

    # TODO: hidden_layers_list = [8, 16, 32]
    def __init__(self, out_dim, hidden_dim=32, layers_num=2):
        super(FlowGNN_fc_block, self).__init__()
        self.out_dim = out_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()

        for i in range(self.layers_num - 1):
            self.layers.append(nn.LazyLinear(self.hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.LazyLinear(self.out_dim))

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            # print('FC hui')
            # print('FC layer num', i)
            x = layer(x)
        # print('FC', )
        return x


class FlowGNN(nn.Module):  # универсальная модель

    def __init__(self, edge_filters, node_filters, fc_in_dim, fc_out_dim, SC_list=None, SC_ed_list=None,
                 fc_skip_indx=None, batchnorm=None, selu=None, loss_func='mae',
                 hidden_size=128, geom_in_dim=2, out_dim=4):
        super(FlowGNN, self).__init__()

        self.edge_filters = edge_filters
        self.node_filters = node_filters
        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim

        # каждому слою сопоставляется множество слоев из которых приходят SC
        self.SC_ed_list = SC_ed_list
        self.SC_list = SC_list  # [None, None, None, [0,1,2],...] В блок № 3 прокинуты скип конеккшены из блоков №№0-2

        self.fc_skip_indx = fc_skip_indx  # [1,3,4,6] слои куда заходит fc слой
        self.batchnorm = batchnorm
        self.selu = selu
        self.hidden_size = hidden_size
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.loss_func = loss_func

        self.layer_list = nn.ModuleList()
        self.FC_list = nn.ModuleList()

        for i, (ef, nf) in enumerate(zip(self.edge_filters, self.node_filters)):

            if i in self.fc_skip_indx:
                self.layer_list.append(
                    FlowGNN_conv_block(ef, nf, skipcon_nodes_idx=self.SC_list[i], skipcon_edges_idx=self.SC_ed_list[i],
                                       fc_skipcon=True, batchnorm=self.batchnorm[i], idx=i, selu=self.selu[i]))
                self.FC_list.append(FlowGNN_fc_block(self.fc_out_dim, hidden_dim=32, layers_num=2))
            else:
                self.layer_list.append(
                    FlowGNN_conv_block(ef, nf, skipcon_nodes_idx=self.SC_list[i], skipcon_edges_idx=self.SC_ed_list[i],
                                       batchnorm=self.batchnorm[i], idx=i, selu=self.selu[i]))

        self.decoder = nn.LazyLinear(self.out_dim)

# TODO: move [:, :self.geom_in_dim+1] to dataset load
    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:, :self.geom_in_dim+1], data.edge_index, data.edge_attr, data.bc, data.batch

        x_outs = {}
        edge_outs = {}
        skipcon_nodes = None
        skipcon_ed = None
        skip_info = x[:, :self.geom_in_dim]
        fc_out = None
        if self.fc_skip_indx is not None:
            fc_out = self.FC_list[0](bc)
        fc_count = 1

        for i, layer in enumerate(self.layer_list):

            if layer.skipcon_nodes_idx is not None:
                skipcon_nodes = x_outs[layer.skipcon_nodes_idx[0]]
                for i in layer.skipcon_nodes_idx[1:]:
                    skipcon_nodes += x_outs[i]

            if layer.skipcon_edges_idx is not None:
                skipcon_ed = edge_outs[layer.skipcon_edges_idx[0]]
                for i in layer.skipcon_edges_idx[1:]:
                    skipcon_ed += edge_outs[i]

            if layer.idx in self.fc_skip_indx[1:]:
                graph_pool = global_mean_pool(x, batch)
                graph_pool = graph_pool[batch]
                fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out, graph_pool], 1))
                fc_count += 1
            x, edge_attr = layer(x, edge_index, edge_attr, skipcon_nodes, skipcon_ed, fc_out, skip_info)

            skipcon_nodes = None
            skipcon_ed = None
            x_outs[i] = x
            edge_outs[i] = edge_attr

        pred = self.decoder(x)

        return pred

    # TODO: move from model
    def loss(self, pred, inp):
        true_flow = inp.flow
        if self.loss_func == 'mae':
            error = torch.mean(torch.abs(true_flow - pred), 1)
        elif self.loss_func == 'mse':
            error = torch.mean(torch.square(true_flow - pred), 1)
        return torch.mean(error)
