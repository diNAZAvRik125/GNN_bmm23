import torch
from torch import nn

import pandas
import torch_scatter
from torch_geometric.data import Dataset
from torch_geometric.nn.pool.glob import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import EdgePooling, DeepGCNLayer, GENConv, MessageNorm
import os


class FlowGNN(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN,self).__init__()


        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        
        self.processor = nn.ModuleList()
        
        self.decoder = nn.LazyLinear(out_dim)

        for ef, nf in zip(self.edge_feat_dims, self.num_filters):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes))
            self.processor.append(SmoothingLayer()) 

    
    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,:7], data.edge_index, data.edge_attr

        skip_info = x[:,:self.geom_in_dim]

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



    

class FlowGNN_with_FC_skipcon(nn.Module):   # Модель из отчета росатома с двумя skip_connection
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_skipcon,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc

        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)
            
            if layer.name == 'smoothing' and layer.idx == 6:
                x = x + x_out[1][0]

                x = torch.cat([x, skip_info], 1)
            if layer.name == 'smoothing' and layer.idx == 7:
                x = x + x_out[0][0]
                x = torch.cat([x, skip_info], 1)


            if layer.name == 'smoothing' and layer.idx == 2:
                fc_out1 = self.FC1(bc)
                #print(x.shape, fc_out1.shape)
                x = torch.cat([x, fc_out1], 1)
                #(print(x.shape, layer.idx, 'after FC1'))

            if layer.name == 'smoothing' and layer.idx == 5:
                x = x + x_out[2][0]
                fc2_inp = torch.cat([bc, fc_out1], 1)
                fc_out2 = self.FC2(fc2_inp)
                x = torch.cat([x, fc_out2], 1)
                #(print(x.shape, layer.idx, 'after FC2'))


        # x1, edge_attr1 = layer(x, edge_index, edge_attr)
        
        # x2, edge_attr2 = layer(x1, edge_index, edge_attr1)
        # x3, edge_attr3 = layer(x2, edge_index, edge_attr2)
        # fc1_out = self.FC1(bc)



        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    


    
class FlowGNN_with_FC_skipcon_deep(nn.Module):  # глубокая версия с полносвязными слоями на каждый сверточный слой, а также двумя skip connection
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_skipcon_deep,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC_list = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            

            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

                if layer.idx == 1:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)

                elif layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    #print(self.FC_list[0], fc_id)
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)

            if layer.name == 'smoothing' and layer.idx == 8:
                x = x + x_out[4][0]
                x = torch.cat([x, skip_info], 1)
            if layer.name == 'smoothing' and layer.idx == 10:
                x = x + x_out[2][0]
                x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    

class FlowGNN_with_FC_deep(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, fcskipcon_indx = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_deep,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.fcskipcon_indx = fcskipcon_indx
        self.processor = nn.ModuleList()

        self.FC_list = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i in fcskipcon_indx):      # i%2 == 1
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        #x_out = []
        fc_counter = 0
        for layer in self.processor:
            

            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                #x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

                if layer.idx == self.fcskipcon_indx[0]:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)
                    fc_counter += 1
                    
                elif layer.idx in self.fcskipcon_indx:
                    fc_out = self.FC_list[fc_counter](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
                    fc_counter += 1

            # if layer.name == 'smoothing' and layer.idx == 8:
            #     x = x + x_out[4][0]
            #     x = torch.cat([x, skip_info], 1)
            # if layer.name == 'smoothing' and layer.idx == 10:
            #     x = x + x_out[2][0]
            #     x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
       
class FlowGNN_withFC(nn.Module):  # модель из отчета росатома. Два полносвязный слоя предают выходы в вершины в сдвух местах
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_withFC,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))

        self.FC_list = nn.Modulelist()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc

        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

            if layer.name == 'smoothing' and layer.idx == 2:
                fc_out1 = self.FC1(bc)
                x = torch.cat([x, fc_out1], 1)
                #(print(x.shape, layer.idx, 'after FC1'))

            if layer.name == 'smoothing' and layer.idx == 5:
                fc_out2 = self.FC2(bc)
                x = torch.cat([x, fc_out2], 1)
                #(print(x.shape, layer.idx, 'after FC2'))


        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)    
    
class FlowGNN_skipBC(nn.Module):    # напрямую прокидываютс параметры гран условий в вершины
    
    def __init__(self, edge_feat_dims, num_filters, skipcon_indx  = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_skipBC,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.skipcon_indx = skipcon_indx
        
        self.processor = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,:3], data.edge_index, data.edge_attr,  # здесь были data.bc, убрал для прямых гран.усл.
        
        bc = data.x[:,3:]
        x = torch.cat([x, bc], 1)

        skip_info = x[:,:self.geom_in_dim]

        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                
            # if layer.name == 'smoothing' and layer.idx == 2:
            #     x = torch.cat([x, bc], 1)

            # if layer.name == 'smoothing' and layer.idx == 5:
            #     x = torch.cat([x, bc], 1)
                #if layer.idx%3 == 0:
                if layer.idx in self.skipcon_indx:   #if layer.idx%3 ==0
                    x = torch.cat([x, bc], 1)
            

        pred = self.decoder(x)
        return pred
    
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)

class FlowGNN_with_FC_pool(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_pool,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.Pool1 = EdgePooling(in_channels=64, edge_score_method=EdgePooling.compute_edge_score_softmax)
        #self.Pool2 = EdgePooling(in_channels=128, edge_score_method=EdgePooling.compute_edge_score_softmax)

            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:,:3], data.edge_index, data.edge_attr, data.bc, data.batch
        skip_info = x[:,:self.geom_in_dim]

        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)

                if layer.idx == 2:
                    fc_out1 = self.FC1(bc)
                    x = torch.cat([x, fc_out1], 1)
                    x, edge_index, batch, unpool = self.Pool1(x, edge_index, batch)

                if layer.idx == 4:
                    x, edge_index, batch = self.Pool1.unpool(x, unpool)
                if layer.idx == 5:
                    fc_out2 = self.FC2(bc)
                    x = torch.cat([x, fc_out2], 1)

        pred = self.decoder(x)
        return pred

class FlowGNN_BN_block(nn.Module):
    
    def __init__(self, edge_dims, node_dims, hidden_size = 128,  skipcon_idx = None, skipcon_ed_idx=None,
                 fc_skipcon = False, batchnorm=False, act=False, msg=False, geom_in_dim=2, idx = 0, selu=False):
        super(FlowGNN_BN_block,self).__init__()
        self.conv = ProcessorLayer(edge_dims, node_dims, hidden_size, msg, selu)
        self.smooth = SmoothingLayer()

        self.node_norm_layer = None
        self.act = None

        if batchnorm:
            #self.node_norm_layer = nn.LayerNorm(node_dims)
            self.node_norm_layer = nn.BatchNorm1d(node_dims)
           # self.edge_norm_layer = nn.LayerNorm(edge_dims)
        if act:
            self.act = nn.ReLU()
        
        self.skipcon_idx = skipcon_idx   # int, layer indx  
        self.skipcon_ed_idx = skipcon_ed_idx
        self.fc_skipcon = fc_skipcon  # bool, 
        self.idx = idx

    def forward(self, x, edge_index, edge_attr, skip_connec=None, skip_connec_ed=None, fc_skipconnec=None, skip_info = None):


        if skip_connec is not None:
            x = torch.add(x, skip_connec)
        
        if skip_connec_ed is not None:
            edge_attr = torch.add(edge_attr, skip_connec_ed)

        if skip_info is not None:
            x = torch.cat([x, skip_info],1)

        if self.fc_skipcon:
            x = torch.cat([x, fc_skipconnec],1)

        x, edge_attr = self.conv(x, edge_index, edge_attr)

        if self.act is not None:
            x , edge_attr = self.act(x), self.act(edge_attr)
        
        x , edge_attr = self.smooth(x, edge_index, edge_attr)
        
        if self.node_norm_layer is not None:
            x = self.node_norm_layer(x)
            #edge_attr = self.edge_norm_layer(edge_attr)

        return x, edge_attr

class FlowGNN_BC_FC_block(nn.Module):

    def __init__(self, out_dim, hidden_dim = 32, layers_num = 2):
        super(FlowGNN_BC_FC_block,self).__init__()
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
            x = layer(x)
        return x
        
        

class FlowGNN_deepmod(nn.Module):

    def __init__(self, edge_filters, node_filters, fc_in_dim, fc_out_dim, SC_list=None, SC_ed_list=None,
                 fc_skip_indx=None, batchnorm = None, activations = None, msg = None, selu=None, loss_func = 'mae', hidden_size = 128, geom_in_dim=2,out_dim = 4):
        super(FlowGNN_deepmod,self).__init__()

        self.edge_filters = edge_filters
        self.node_filters = node_filters
        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.SC_ed_list = SC_ed_list
        self.SC_list = SC_list     #[[1,2], None, None, [2,3,4],...]  каждому слою сопоставляется множество слоев из которых приходят SC
        self.fc_skip_indx = fc_skip_indx # [1,3,4,6] слои куда заходит fc слой
        self.batchnorm = batchnorm
        self.activation = activations
        self.selu = selu
        self.msg = msg
        self.hidden_size = hidden_size
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.loss_func = loss_func

        self.layer_list = nn.ModuleList()
        self.FC_list = nn.ModuleList()

        for i, (ef, nf) in enumerate(zip(self.edge_filters, self.node_filters)):

            if i in self.fc_skip_indx:
                self.layer_list.append(FlowGNN_BN_block(ef,nf, skipcon_idx = self.SC_list[i],skipcon_ed_idx=self.SC_ed_list[i], fc_skipcon=True, batchnorm = self.batchnorm[i],
                act = self.activation[i], msg = self.msg[i], idx = i, selu = self.selu[i]))
                self.FC_list.append(FlowGNN_BC_FC_block(self.fc_out_dim,hidden_dim=32, layers_num=2))
            else:  
                self.layer_list.append(FlowGNN_BN_block(ef,nf, skipcon_idx = self.SC_list[i], skipcon_ed_idx=self.SC_ed_list[i], batchnorm = self.batchnorm[i],
                                    act = self.activation[i], msg = self.msg[i],idx = i, selu = self.selu[i]))
            
                
        self.decoder = nn.LazyLinear(self.out_dim)
                    
    
    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:,:3], data.edge_index, data.edge_attr, data.bc, data.batch

        x_outs = {}
        edge_outs = {}
        skipcon=None
        skipcon_ed = None
        skip_info = x[:,:self.geom_in_dim]
        fc_out=None
        if self.fc_skip_indx is not None:
            fc_out = self.FC_list[0](bc)
        fc_count = 1

        for i, layer in enumerate(self.layer_list):
            
            #print(i, 'hui')
            if layer.skipcon_idx is not None:
                #print(skipcon)
                skipcon = x_outs[layer.skipcon_idx[0]]
                for i in layer.skipcon_idx[1:]:
                    skipcon += x_outs[i]

            if layer.skipcon_ed_idx is not None:
                skipcon_ed = edge_outs[layer.skipcon_ed_idx[0]]
                for i in layer.skipcon_ed_idx[1:]:
                    skipcon_ed += edge_outs[i]
                #print(skipcon.shape, 'hui2')

            if layer.idx in self.fc_skip_indx[1:]:

                graph_pool = global_mean_pool(x, batch)
                graph_pool = graph_pool[batch]
                fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out, graph_pool],1))
                #fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out],1))
                fc_count += 1
            #print(skipcon_ed, 'hui')
            x, edge_attr = layer(x, edge_index, edge_attr, skipcon, skipcon_ed, fc_out, skip_info)

            skipcon = None
            skipcon_ed = None
            x_outs[i] = x
            edge_outs[i] = edge_attr
        
        pred = self.decoder(x)

        return pred
                        
    def loss(self,pred, inp):
        true_flow = inp.flow
        if self.loss_func == 'mae':
            error = torch.mean(torch.abs(true_flow - pred), 1)
        elif self.loss_func == 'mse':
            error = torch.mean(torch.square(true_flow - pred), 1)
        return torch.mean(error) 


                
class FlowGNN_with_FC_BN_RC(nn.Module):  #DeepGCN

    def __init__(self, hidden_size, num_layers, fc_in_dim, fc_out_dim, fcskipcon_indx = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super().__init__()
        self.hidden = hidden_size
        self.hidden_nodes = hidden_nodes
        self.num_layers = num_layers
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.fcskipcon_idx = fcskipcon_indx

        self.node_encoder = nn.LazyLinear(16)
        self.edge_encoder = nn.LazyLinear(16)

        self.node_decoder = nn.LazyLinear(self.out_dim)

        self.conv1 = ProcessorLayer(64,64, self.hidden_nodes)
        self.smooth1 = SmoothingLayer()

        self.layers = nn.ModuleList()

        for i in range(1,self.num_layers-1):

            input_dim = self.hidden + self.geom_in_dim
            if i in  self.fcskipcon_idx:
                input_dim += self.fc_out_dim 
            #print(input_dim)
            conv = GENConv(input_dim, self.hidden, aggr='softmax', 
                        t=1.0, learn_t=True, num_layers=2, norm='batch', msg_norm=True, ckpt_grad = False)
            norm = nn.BatchNorm1d(self.hidden, affine = True)
            act = nn.ReLU()
            layer = DeepGCNLayer(conv, norm, act, block='res')
            self.layers.append(layer)
        
        self.convn = ProcessorLayer(8,16,self.hidden)
        self.smoothn = SmoothingLayer()

    def forward(self, data):
        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        skip_info = x[:,:self.geom_in_dim]
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.smooth1(x, edge_index, edge_attr)
        
        prev_x = x
        for i, layer in enumerate(self.layers, start = 1):
            
            x = torch.cat([prev_x, skip_info], 1)
            #print(x.shape, 1)
            if i in self.fcskipcon_idx:
                x = torch.cat([x, bc], 1) 
                #print(x.shape, 2)
            x = layer(x, prev_x, edge_index, edge_attr)
            #print(x.shape, 3)
            prev_x = x
            # x = torch.cat([x, skip_info], 1)
            #           
        x, edge_attr = self.convn(x, edge_index, edge_attr)
        x, edge_attr = self.smoothn(x, edge_index, edge_attr)

        out = self.node_decoder(x)
        return out

    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)         





class FlowGNN_3Decoder(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, decoder_edge_feat_dims,decoder_nodes_feat_dims, geom_in_dim = 2, hidden_nodes=128):
        super(FlowGNN_3Decoder,self).__init__()

        self.encoder_edge_feat_dims = edge_feat_dims
        self.encoder_num_filters = num_filters
        self.decoder_edges = decoder_edge_feat_dims
        self.decoder_nodes = decoder_nodes_feat_dims
        self.geom_in_dim = geom_in_dim
        #self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        
        self.encoder = nn.ModuleList()
        self.FC_list = nn.ModuleList()
        for i, (ef, nf) in enumerate(zip(self.encoder_edge_feat_dims, self.encoder_num_filters)):
            self.encoder.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.encoder.append(SmoothingLayer(idx = i))

            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

        self.decoder_1 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_2 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_3 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_4 = Decoder(self.decoder_edges, self.decoder_nodes)
        

    def forward(self, data):
        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.encoder:
            x, edge_attr = layer(x, edge_index, edge_attr)

            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])

                if layer.idx == 1:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)

                elif layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    #print(self.FC_list[0], fc_id)
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
        
        x1 = self.decoder_1(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x2 = self.decoder_2(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x3 = self.decoder_3(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x4 = self.decoder_4(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        
        out = torch.cat([x1,x2,x3, x4], 1)
        return out
    
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    

class Decoder(nn.Module):



    def __init__(self, edge_feat_dims, num_filters, hidden_nodes = 128, fc_out_dim = 4, out_dim = 1):
        super(Decoder,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.hidden_nodes = hidden_nodes
        self.fc_out_dim = fc_out_dim

        self.processor= nn.ModuleList()
        self.FC_list = nn.ModuleList()
        self.FC_decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, self.hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, x, edge_attr, edge_index, bc, x_out, skip_info, fc_out = None):

        #x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        

        for layer in self.processor:

            x, edge_attr = layer(x, edge_index, edge_attr)

            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])

                if layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
                
        x = self.FC_decoder(x)
        return x





class ProcessorLayer(MessagePassing):

    def __init__(self, edge_feats, node_feats, hidden_state, msg = False, idx = 0, selu = False):
        super(ProcessorLayer, self).__init__()

        self.name = 'processor'
        self.idx = idx
        activation = nn.ReLU()

        self.msg = msg
        if msg:
            self.messagenorm = MessageNorm(learn_scale = True)

        if selu:
            activation = nn.SELU()

        self.edge_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      #nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(edge_feats)
                                      #activation,
                                      )
                                      
        self.node_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      #nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(node_feats),
                                      activation,
                                      )
        
        
    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr)

        if self.msg:
            out = self.messagenorm(x, out)
            
        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges


    def message(self, x_i, x_j, edge_attr):
        #print(x_i.shape, x_j.shape,edge_attr.shape)

        updated_edges = torch.cat([torch.div(x_i + x_j,2), torch.abs(x_i - x_j)/2, edge_attr], 1)
        #print(updated_edges.shape, 'hui message')
        updated_edges = self.edge_mlp(updated_edges)
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        #print(updated_edges.shape, torch.max(edge_index[0, :]), edge_index[0, :].shape[0], torch.min(edge_index[0, :]), 'hui')
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'sum')
        return out, updated_edges
    

class SmoothingLayer(MessagePassing):

    def __init__(self, idx=0):
        super(SmoothingLayer, self).__init__()
        
        self.name = 'smoothing'
        self.idx = idx

    def forward(self, x, edge_index, edge_attr):

        out_nodes, out_edges = self.propagate(edge_index, x=x, edge_attr = edge_attr)
        return out_nodes, out_edges
  
    def message(self, x_i, x_j):
        updated_edges = (x_i + x_j)/2
        return updated_edges

    def aggregate(self, updated_edges, edge_index):

        node_dim = 0
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'mean')
        return out, updated_edges
import torch
from torch import nn

import pandas
import torch_scatter
from torch_geometric.data import Dataset
from torch_geometric.nn.pool.glob import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import EdgePooling, DeepGCNLayer, GENConv, MessageNorm
import os


class FlowGNN(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN,self).__init__()


        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        
        self.processor = nn.ModuleList()
        
        self.decoder = nn.LazyLinear(out_dim)

        for ef, nf in zip(self.edge_feat_dims, self.num_filters):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes))
            self.processor.append(SmoothingLayer()) 

    
    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,:7], data.edge_index, data.edge_attr

        skip_info = x[:,:self.geom_in_dim]

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



    

class FlowGNN_with_FC_skipcon(nn.Module):   # Модель из отчета росатома с двумя skip_connection
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_skipcon,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc

        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)
            
            if layer.name == 'smoothing' and layer.idx == 6:
                x = x + x_out[1][0]

                x = torch.cat([x, skip_info], 1)
            if layer.name == 'smoothing' and layer.idx == 7:
                x = x + x_out[0][0]
                x = torch.cat([x, skip_info], 1)


            if layer.name == 'smoothing' and layer.idx == 2:
                fc_out1 = self.FC1(bc)
                #print(x.shape, fc_out1.shape)
                x = torch.cat([x, fc_out1], 1)
                #(print(x.shape, layer.idx, 'after FC1'))

            if layer.name == 'smoothing' and layer.idx == 5:
                x = x + x_out[2][0]
                fc2_inp = torch.cat([bc, fc_out1], 1)
                fc_out2 = self.FC2(fc2_inp)
                x = torch.cat([x, fc_out2], 1)
                #(print(x.shape, layer.idx, 'after FC2'))


        # x1, edge_attr1 = layer(x, edge_index, edge_attr)
        
        # x2, edge_attr2 = layer(x1, edge_index, edge_attr1)
        # x3, edge_attr3 = layer(x2, edge_index, edge_attr2)
        # fc1_out = self.FC1(bc)



        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    


    
class FlowGNN_with_FC_skipcon_deep(nn.Module):  # глубокая версия с полносвязными слоями на каждый сверточный слой, а также двумя skip connection
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_skipcon_deep,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC_list = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            

            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

                if layer.idx == 1:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)

                elif layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    #print(self.FC_list[0], fc_id)
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)

            if layer.name == 'smoothing' and layer.idx == 8:
                x = x + x_out[4][0]
                x = torch.cat([x, skip_info], 1)
            if layer.name == 'smoothing' and layer.idx == 10:
                x = x + x_out[2][0]
                x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    

class FlowGNN_with_FC_deep(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, fcskipcon_indx = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_deep,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.fcskipcon_indx = fcskipcon_indx
        self.processor = nn.ModuleList()

        self.FC_list = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i in fcskipcon_indx):      # i%2 == 1
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        #x_out = []
        fc_counter = 0
        for layer in self.processor:
            

            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                #x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

                if layer.idx == self.fcskipcon_indx[0]:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)
                    fc_counter += 1
                    
                elif layer.idx in self.fcskipcon_indx:
                    fc_out = self.FC_list[fc_counter](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
                    fc_counter += 1

            # if layer.name == 'smoothing' and layer.idx == 8:
            #     x = x + x_out[4][0]
            #     x = torch.cat([x, skip_info], 1)
            # if layer.name == 'smoothing' and layer.idx == 10:
            #     x = x + x_out[2][0]
            #     x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
       
class FlowGNN_withFC(nn.Module):  # модель из отчета росатома. Два полносвязный слоя предают выходы в вершины в сдвух местах
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_withFC,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))

        self.FC_list = nn.Modulelist()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc

        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

            if layer.name == 'smoothing' and layer.idx == 2:
                fc_out1 = self.FC1(bc)
                x = torch.cat([x, fc_out1], 1)
                #(print(x.shape, layer.idx, 'after FC1'))

            if layer.name == 'smoothing' and layer.idx == 5:
                fc_out2 = self.FC2(bc)
                x = torch.cat([x, fc_out2], 1)
                #(print(x.shape, layer.idx, 'after FC2'))


        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)    
    
class FlowGNN_skipBC(nn.Module):    # напрямую прокидываютс параметры гран условий в вершины
    
    def __init__(self, edge_feat_dims, num_filters, skipcon_indx  = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_skipBC,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.skipcon_indx = skipcon_indx
        
        self.processor = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,:3], data.edge_index, data.edge_attr,  # здесь были data.bc, убрал для прямых гран.усл.
        
        bc = data.x[:,3:]
        x = torch.cat([x, bc], 1)

        skip_info = x[:,:self.geom_in_dim]

        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                
            # if layer.name == 'smoothing' and layer.idx == 2:
            #     x = torch.cat([x, bc], 1)

            # if layer.name == 'smoothing' and layer.idx == 5:
            #     x = torch.cat([x, bc], 1)
                #if layer.idx%3 == 0:
                if layer.idx in self.skipcon_indx:   #if layer.idx%3 ==0
                    x = torch.cat([x, bc], 1)
            

        pred = self.decoder(x)
        return pred
    
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)

class FlowGNN_with_FC_pool(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_pool,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.Pool1 = EdgePooling(in_channels=64, edge_score_method=EdgePooling.compute_edge_score_softmax)
        #self.Pool2 = EdgePooling(in_channels=128, edge_score_method=EdgePooling.compute_edge_score_softmax)

            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:,:3], data.edge_index, data.edge_attr, data.bc, data.batch
        skip_info = x[:,:self.geom_in_dim]

        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)

                if layer.idx == 2:
                    fc_out1 = self.FC1(bc)
                    x = torch.cat([x, fc_out1], 1)
                    x, edge_index, batch, unpool = self.Pool1(x, edge_index, batch)

                if layer.idx == 4:
                    x, edge_index, batch = self.Pool1.unpool(x, unpool)
                if layer.idx == 5:
                    fc_out2 = self.FC2(bc)
                    x = torch.cat([x, fc_out2], 1)

        pred = self.decoder(x)
        return pred

class FlowGNN_BN_block(nn.Module):
    
    def __init__(self, edge_dims, node_dims, hidden_size = 128,  skipcon_idx = None, skipcon_ed_idx=None,
                 fc_skipcon = False, batchnorm=False, act=False, msg=False, geom_in_dim=2, idx = 0, selu=False):
        super(FlowGNN_BN_block,self).__init__()
        self.conv = ProcessorLayer(edge_dims, node_dims, hidden_size, msg, selu)
        self.smooth = SmoothingLayer()

        self.node_norm_layer = None
        self.act = None

        if batchnorm:
            #self.node_norm_layer = nn.LayerNorm(node_dims)
            self.node_norm_layer = nn.BatchNorm1d(node_dims)
           # self.edge_norm_layer = nn.LayerNorm(edge_dims)
        if act:
            self.act = nn.ReLU()
        
        self.skipcon_idx = skipcon_idx   # int, layer indx  
        self.skipcon_ed_idx = skipcon_ed_idx
        self.fc_skipcon = fc_skipcon  # bool, 
        self.idx = idx

    def forward(self, x, edge_index, edge_attr, skip_connec=None, skip_connec_ed=None, fc_skipconnec=None, skip_info = None):


        if skip_connec is not None:
            x = torch.add(x, skip_connec)
        
        if skip_connec_ed is not None:
            edge_attr = torch.add(edge_attr, skip_connec_ed)

        if skip_info is not None:
            x = torch.cat([x, skip_info],1)

        if self.fc_skipcon:
            x = torch.cat([x, fc_skipconnec],1)

        x, edge_attr = self.conv(x, edge_index, edge_attr)

        if self.act is not None:
            x , edge_attr = self.act(x), self.act(edge_attr)
        
        x , edge_attr = self.smooth(x, edge_index, edge_attr)
        
        if self.node_norm_layer is not None:
            x = self.node_norm_layer(x)
            #edge_attr = self.edge_norm_layer(edge_attr)

        return x, edge_attr

class FlowGNN_BC_FC_block(nn.Module):

    def __init__(self, out_dim, hidden_dim = 32, layers_num = 2):
        super(FlowGNN_BC_FC_block,self).__init__()
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
            x = layer(x)
        return x
        
        

class FlowGNN_deepmod(nn.Module):

    def __init__(self, edge_filters, node_filters, fc_in_dim, fc_out_dim, SC_list=None, SC_ed_list=None,
                 fc_skip_indx=None, batchnorm = None, activations = None, msg = None, selu=None, loss_func = 'mae', hidden_size = 128, geom_in_dim=2,out_dim = 4):
        super(FlowGNN_deepmod,self).__init__()

        self.edge_filters = edge_filters
        self.node_filters = node_filters
        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.SC_ed_list = SC_ed_list
        self.SC_list = SC_list     #[[1,2], None, None, [2,3,4],...]  каждому слою сопоставляется множество слоев из которых приходят SC
        self.fc_skip_indx = fc_skip_indx # [1,3,4,6] слои куда заходит fc слой
        self.batchnorm = batchnorm
        self.activation = activations
        self.selu = selu
        self.msg = msg
        self.hidden_size = hidden_size
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.loss_func = loss_func

        self.layer_list = nn.ModuleList()
        self.FC_list = nn.ModuleList()

        for i, (ef, nf) in enumerate(zip(self.edge_filters, self.node_filters)):

            if i in self.fc_skip_indx:
                self.layer_list.append(FlowGNN_BN_block(ef,nf, skipcon_idx = self.SC_list[i],skipcon_ed_idx=self.SC_ed_list[i], fc_skipcon=True, batchnorm = self.batchnorm[i],
                act = self.activation[i], msg = self.msg[i], idx = i, selu = self.selu[i]))
                self.FC_list.append(FlowGNN_BC_FC_block(self.fc_out_dim,hidden_dim=32, layers_num=2))
            else:  
                self.layer_list.append(FlowGNN_BN_block(ef,nf, skipcon_idx = self.SC_list[i], skipcon_ed_idx=self.SC_ed_list[i], batchnorm = self.batchnorm[i],
                                    act = self.activation[i], msg = self.msg[i],idx = i, selu = self.selu[i]))
            
                
        self.decoder = nn.LazyLinear(self.out_dim)
                    
    
    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:,:3], data.edge_index, data.edge_attr, data.bc, data.batch

        x_outs = {}
        edge_outs = {}
        skipcon=None
        skipcon_ed = None
        skip_info = x[:,:self.geom_in_dim]
        fc_out=None
        if self.fc_skip_indx is not None:
            fc_out = self.FC_list[0](bc)
        fc_count = 1

        for i, layer in enumerate(self.layer_list):
            
            #print(i, 'hui')
            if layer.skipcon_idx is not None:
                #print(skipcon)
                skipcon = x_outs[layer.skipcon_idx[0]]
                for i in layer.skipcon_idx[1:]:
                    skipcon += x_outs[i]

            if layer.skipcon_ed_idx is not None:
                skipcon_ed = edge_outs[layer.skipcon_ed_idx[0]]
                for i in layer.skipcon_ed_idx[1:]:
                    skipcon_ed += edge_outs[i]
                #print(skipcon.shape, 'hui2')

            if layer.idx in self.fc_skip_indx[1:]:

                graph_pool = global_mean_pool(x, batch)
                graph_pool = graph_pool[batch]
                fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out, graph_pool],1))
                #fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out],1))
                fc_count += 1
            #print(skipcon_ed, 'hui')
            x, edge_attr = layer(x, edge_index, edge_attr, skipcon, skipcon_ed, fc_out, skip_info)

            skipcon = None
            skipcon_ed = None
            x_outs[i] = x
            edge_outs[i] = edge_attr
        
        pred = self.decoder(x)

        return pred
                        
    def loss(self,pred, inp):
        true_flow = inp.flow
        if self.loss_func == 'mae':
            error = torch.mean(torch.abs(true_flow - pred), 1)
        elif self.loss_func == 'mse':
            error = torch.mean(torch.square(true_flow - pred), 1)
        return torch.mean(error) 


                
class FlowGNN_with_FC_BN_RC(nn.Module):  #DeepGCN

    def __init__(self, hidden_size, num_layers, fc_in_dim, fc_out_dim, fcskipcon_indx = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super().__init__()
        self.hidden = hidden_size
        self.hidden_nodes = hidden_nodes
        self.num_layers = num_layers
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.fcskipcon_idx = fcskipcon_indx

        self.node_encoder = nn.LazyLinear(16)
        self.edge_encoder = nn.LazyLinear(16)

        self.node_decoder = nn.LazyLinear(self.out_dim)

        self.conv1 = ProcessorLayer(64,64, self.hidden_nodes)
        self.smooth1 = SmoothingLayer()

        self.layers = nn.ModuleList()

        for i in range(1,self.num_layers-1):

            input_dim = self.hidden + self.geom_in_dim
            if i in  self.fcskipcon_idx:
                input_dim += self.fc_out_dim 
            #print(input_dim)
            conv = GENConv(input_dim, self.hidden, aggr='softmax', 
                        t=1.0, learn_t=True, num_layers=2, norm='batch', msg_norm=True, ckpt_grad = False)
            norm = nn.BatchNorm1d(self.hidden, affine = True)
            act = nn.ReLU()
            layer = DeepGCNLayer(conv, norm, act, block='res')
            self.layers.append(layer)
        
        self.convn = ProcessorLayer(8,16,self.hidden)
        self.smoothn = SmoothingLayer()

    def forward(self, data):
        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        skip_info = x[:,:self.geom_in_dim]
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.smooth1(x, edge_index, edge_attr)
        
        prev_x = x
        for i, layer in enumerate(self.layers, start = 1):
            
            x = torch.cat([prev_x, skip_info], 1)
            #print(x.shape, 1)
            if i in self.fcskipcon_idx:
                x = torch.cat([x, bc], 1) 
                #print(x.shape, 2)
            x = layer(x, prev_x, edge_index, edge_attr)
            #print(x.shape, 3)
            prev_x = x
            # x = torch.cat([x, skip_info], 1)
            #           
        x, edge_attr = self.convn(x, edge_index, edge_attr)
        x, edge_attr = self.smoothn(x, edge_index, edge_attr)

        out = self.node_decoder(x)
        return out

    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)         





class FlowGNN_3Decoder(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, decoder_edge_feat_dims,decoder_nodes_feat_dims, geom_in_dim = 2, hidden_nodes=128):
        super(FlowGNN_3Decoder,self).__init__()

        self.encoder_edge_feat_dims = edge_feat_dims
        self.encoder_num_filters = num_filters
        self.decoder_edges = decoder_edge_feat_dims
        self.decoder_nodes = decoder_nodes_feat_dims
        self.geom_in_dim = geom_in_dim
        #self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        
        self.encoder = nn.ModuleList()
        self.FC_list = nn.ModuleList()
        for i, (ef, nf) in enumerate(zip(self.encoder_edge_feat_dims, self.encoder_num_filters)):
            self.encoder.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.encoder.append(SmoothingLayer(idx = i))

            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

        self.decoder_1 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_2 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_3 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_4 = Decoder(self.decoder_edges, self.decoder_nodes)
        

    def forward(self, data):
        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.encoder:
            x, edge_attr = layer(x, edge_index, edge_attr)

            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])

                if layer.idx == 1:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)

                elif layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    #print(self.FC_list[0], fc_id)
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
        
        x1 = self.decoder_1(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x2 = self.decoder_2(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x3 = self.decoder_3(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x4 = self.decoder_4(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        
        out = torch.cat([x1,x2,x3, x4], 1)
        return out
    
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    

class Decoder(nn.Module):



    def __init__(self, edge_feat_dims, num_filters, hidden_nodes = 128, fc_out_dim = 4, out_dim = 1):
        super(Decoder,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.hidden_nodes = hidden_nodes
        self.fc_out_dim = fc_out_dim

        self.processor= nn.ModuleList()
        self.FC_list = nn.ModuleList()
        self.FC_decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, self.hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, x, edge_attr, edge_index, bc, x_out, skip_info, fc_out = None):

        #x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        

        for layer in self.processor:

            x, edge_attr = layer(x, edge_index, edge_attr)

            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])

                if layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
                
        x = self.FC_decoder(x)
        return x





class ProcessorLayer(MessagePassing):

    def __init__(self, edge_feats, node_feats, hidden_state, msg = False, idx = 0, selu = False):
        super(ProcessorLayer, self).__init__()

        self.name = 'processor'
        self.idx = idx
        activation = nn.ReLU()

        self.msg = msg
        if msg:
            self.messagenorm = MessageNorm(learn_scale = True)

        if selu:
            activation = nn.SELU()

        self.edge_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      #nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(edge_feats)
                                      #activation,
                                      )
                                      
        self.node_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      #nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(node_feats),
                                      activation,
                                      )
        
        
    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr)

        if self.msg:
            out = self.messagenorm(x, out)
            
        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges


    def message(self, x_i, x_j, edge_attr):
        #print(x_i.shape, x_j.shape,edge_attr.shape)

        updated_edges = torch.cat([torch.div(x_i + x_j,2), torch.abs(x_i - x_j)/2, edge_attr], 1)
        #print(updated_edges.shape, 'hui message')
        updated_edges = self.edge_mlp(updated_edges)
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        #print(updated_edges.shape, torch.max(edge_index[0, :]), edge_index[0, :].shape[0], torch.min(edge_index[0, :]), 'hui')
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'sum')
        return out, updated_edges
    

class SmoothingLayer(MessagePassing):

    def __init__(self, idx=0):
        super(SmoothingLayer, self).__init__()
        
        self.name = 'smoothing'
        self.idx = idx

    def forward(self, x, edge_index, edge_attr):

        out_nodes, out_edges = self.propagate(edge_index, x=x, edge_attr = edge_attr)
        return out_nodes, out_edges
  
    def message(self, x_i, x_j):
        updated_edges = (x_i + x_j)/2
        return updated_edges

    def aggregate(self, updated_edges, edge_index):

        node_dim = 0
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'mean')
        return out, updated_edges
import torch
from torch import nn

import pandas
import torch_scatter
from torch_geometric.data import Dataset
from torch_geometric.nn.pool.glob import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import EdgePooling, DeepGCNLayer, GENConv, MessageNorm
import os


class FlowGNN(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN,self).__init__()


        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        
        self.processor = nn.ModuleList()
        
        self.decoder = nn.LazyLinear(out_dim)

        for ef, nf in zip(self.edge_feat_dims, self.num_filters):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes))
            self.processor.append(SmoothingLayer()) 

    
    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,:7], data.edge_index, data.edge_attr

        skip_info = x[:,:self.geom_in_dim]

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



    

class FlowGNN_with_FC_skipcon(nn.Module):   # Модель из отчета росатома с двумя skip_connection
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_skipcon,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc

        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)
            
            if layer.name == 'smoothing' and layer.idx == 6:
                x = x + x_out[1][0]

                x = torch.cat([x, skip_info], 1)
            if layer.name == 'smoothing' and layer.idx == 7:
                x = x + x_out[0][0]
                x = torch.cat([x, skip_info], 1)


            if layer.name == 'smoothing' and layer.idx == 2:
                fc_out1 = self.FC1(bc)
                #print(x.shape, fc_out1.shape)
                x = torch.cat([x, fc_out1], 1)
                #(print(x.shape, layer.idx, 'after FC1'))

            if layer.name == 'smoothing' and layer.idx == 5:
                x = x + x_out[2][0]
                fc2_inp = torch.cat([bc, fc_out1], 1)
                fc_out2 = self.FC2(fc2_inp)
                x = torch.cat([x, fc_out2], 1)
                #(print(x.shape, layer.idx, 'after FC2'))


        # x1, edge_attr1 = layer(x, edge_index, edge_attr)
        
        # x2, edge_attr2 = layer(x1, edge_index, edge_attr1)
        # x3, edge_attr3 = layer(x2, edge_index, edge_attr2)
        # fc1_out = self.FC1(bc)



        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    


    
class FlowGNN_with_FC_skipcon_deep(nn.Module):  # глубокая версия с полносвязными слоями на каждый сверточный слой, а также двумя skip connection
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_skipcon_deep,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC_list = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            

            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

                if layer.idx == 1:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)

                elif layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    #print(self.FC_list[0], fc_id)
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)

            if layer.name == 'smoothing' and layer.idx == 8:
                x = x + x_out[4][0]
                x = torch.cat([x, skip_info], 1)
            if layer.name == 'smoothing' and layer.idx == 10:
                x = x + x_out[2][0]
                x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    

class FlowGNN_with_FC_deep(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, fcskipcon_indx = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_deep,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.fcskipcon_indx = fcskipcon_indx
        self.processor = nn.ModuleList()

        self.FC_list = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i in fcskipcon_indx):      # i%2 == 1
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        #x_out = []
        fc_counter = 0
        for layer in self.processor:
            

            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                #x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

                if layer.idx == self.fcskipcon_indx[0]:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)
                    fc_counter += 1
                    
                elif layer.idx in self.fcskipcon_indx:
                    fc_out = self.FC_list[fc_counter](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
                    fc_counter += 1

            # if layer.name == 'smoothing' and layer.idx == 8:
            #     x = x + x_out[4][0]
            #     x = torch.cat([x, skip_info], 1)
            # if layer.name == 'smoothing' and layer.idx == 10:
            #     x = x + x_out[2][0]
            #     x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
       
class FlowGNN_withFC(nn.Module):  # модель из отчета росатома. Два полносвязный слоя предают выходы в вершины в сдвух местах
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_withFC,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))

        self.FC_list = nn.Modulelist()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc

        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

            if layer.name == 'smoothing' and layer.idx == 2:
                fc_out1 = self.FC1(bc)
                x = torch.cat([x, fc_out1], 1)
                #(print(x.shape, layer.idx, 'after FC1'))

            if layer.name == 'smoothing' and layer.idx == 5:
                fc_out2 = self.FC2(bc)
                x = torch.cat([x, fc_out2], 1)
                #(print(x.shape, layer.idx, 'after FC2'))


        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)    
    
class FlowGNN_skipBC(nn.Module):    # напрямую прокидываютс параметры гран условий в вершины
    
    def __init__(self, edge_feat_dims, num_filters, skipcon_indx  = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_skipBC,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.skipcon_indx = skipcon_indx
        
        self.processor = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,:3], data.edge_index, data.edge_attr,  # здесь были data.bc, убрал для прямых гран.усл.
        
        bc = data.x[:,3:]
        x = torch.cat([x, bc], 1)

        skip_info = x[:,:self.geom_in_dim]

        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                
            # if layer.name == 'smoothing' and layer.idx == 2:
            #     x = torch.cat([x, bc], 1)

            # if layer.name == 'smoothing' and layer.idx == 5:
            #     x = torch.cat([x, bc], 1)
                #if layer.idx%3 == 0:
                if layer.idx in self.skipcon_indx:   #if layer.idx%3 ==0
                    x = torch.cat([x, bc], 1)
            

        pred = self.decoder(x)
        return pred
    
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)

class FlowGNN_with_FC_pool(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_pool,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.Pool1 = EdgePooling(in_channels=64, edge_score_method=EdgePooling.compute_edge_score_softmax)
        #self.Pool2 = EdgePooling(in_channels=128, edge_score_method=EdgePooling.compute_edge_score_softmax)

            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:,:3], data.edge_index, data.edge_attr, data.bc, data.batch
        skip_info = x[:,:self.geom_in_dim]

        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)

                if layer.idx == 2:
                    fc_out1 = self.FC1(bc)
                    x = torch.cat([x, fc_out1], 1)
                    x, edge_index, batch, unpool = self.Pool1(x, edge_index, batch)

                if layer.idx == 4:
                    x, edge_index, batch = self.Pool1.unpool(x, unpool)
                if layer.idx == 5:
                    fc_out2 = self.FC2(bc)
                    x = torch.cat([x, fc_out2], 1)

        pred = self.decoder(x)
        return pred

class FlowGNN_BN_block(nn.Module):
    
    def __init__(self, edge_dims, node_dims, hidden_size = 128,  skipcon_idx = None, skipcon_ed_idx=None,
                 fc_skipcon = False, batchnorm=False, act=False, msg=False, geom_in_dim=2, idx = 0, selu=False):
        super(FlowGNN_BN_block,self).__init__()
        self.conv = ProcessorLayer(edge_dims, node_dims, hidden_size, msg, selu)
        self.smooth = SmoothingLayer()

        self.node_norm_layer = None
        self.act = None

        if batchnorm:
            #self.node_norm_layer = nn.LayerNorm(node_dims)
            self.node_norm_layer = nn.BatchNorm1d(node_dims)
           # self.edge_norm_layer = nn.LayerNorm(edge_dims)
        if act:
            self.act = nn.ReLU()
        
        self.skipcon_idx = skipcon_idx   # int, layer indx  
        self.skipcon_ed_idx = skipcon_ed_idx
        self.fc_skipcon = fc_skipcon  # bool, 
        self.idx = idx

    def forward(self, x, edge_index, edge_attr, skip_connec=None, skip_connec_ed=None, fc_skipconnec=None, skip_info = None):


        if skip_connec is not None:
            x = torch.add(x, skip_connec)
        
        if skip_connec_ed is not None:
            edge_attr = torch.add(edge_attr, skip_connec_ed)

        if skip_info is not None:
            x = torch.cat([x, skip_info],1)

        if self.fc_skipcon:
            x = torch.cat([x, fc_skipconnec],1)

        x, edge_attr = self.conv(x, edge_index, edge_attr)

        if self.act is not None:
            x , edge_attr = self.act(x), self.act(edge_attr)
        
        x , edge_attr = self.smooth(x, edge_index, edge_attr)
        
        if self.node_norm_layer is not None:
            x = self.node_norm_layer(x)
            #edge_attr = self.edge_norm_layer(edge_attr)

        return x, edge_attr

class FlowGNN_BC_FC_block(nn.Module):

    def __init__(self, out_dim, hidden_dim = 32, layers_num = 2):
        super(FlowGNN_BC_FC_block,self).__init__()
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
            x = layer(x)
        return x
        
        

class FlowGNN_deepmod(nn.Module):

    def __init__(self, edge_filters, node_filters, fc_in_dim, fc_out_dim, SC_list=None, SC_ed_list=None,
                 fc_skip_indx=None, batchnorm = None, activations = None, msg = None, selu=None, loss_func = 'mae', hidden_size = 128, geom_in_dim=2,out_dim = 4):
        super(FlowGNN_deepmod,self).__init__()

        self.edge_filters = edge_filters
        self.node_filters = node_filters
        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.SC_ed_list = SC_ed_list
        self.SC_list = SC_list     #[[1,2], None, None, [2,3,4],...]  каждому слою сопоставляется множество слоев из которых приходят SC
        self.fc_skip_indx = fc_skip_indx # [1,3,4,6] слои куда заходит fc слой
        self.batchnorm = batchnorm
        self.activation = activations
        self.selu = selu
        self.msg = msg
        self.hidden_size = hidden_size
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.loss_func = loss_func

        self.layer_list = nn.ModuleList()
        self.FC_list = nn.ModuleList()

        for i, (ef, nf) in enumerate(zip(self.edge_filters, self.node_filters)):

            if i in self.fc_skip_indx:
                self.layer_list.append(FlowGNN_BN_block(ef,nf, skipcon_idx = self.SC_list[i],skipcon_ed_idx=self.SC_ed_list[i], fc_skipcon=True, batchnorm = self.batchnorm[i],
                act = self.activation[i], msg = self.msg[i], idx = i, selu = self.selu[i]))
                self.FC_list.append(FlowGNN_BC_FC_block(self.fc_out_dim,hidden_dim=32, layers_num=2))
            else:  
                self.layer_list.append(FlowGNN_BN_block(ef,nf, skipcon_idx = self.SC_list[i], skipcon_ed_idx=self.SC_ed_list[i], batchnorm = self.batchnorm[i],
                                    act = self.activation[i], msg = self.msg[i],idx = i, selu = self.selu[i]))
            
                
        self.decoder = nn.LazyLinear(self.out_dim)
                    
    
    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:,:3], data.edge_index, data.edge_attr, data.bc, data.batch

        x_outs = {}
        edge_outs = {}
        skipcon=None
        skipcon_ed = None
        skip_info = x[:,:self.geom_in_dim]
        fc_out=None
        if self.fc_skip_indx is not None:
            fc_out = self.FC_list[0](bc)
        fc_count = 1

        for i, layer in enumerate(self.layer_list):
            
            #print(i, 'hui')
            if layer.skipcon_idx is not None:
                #print(skipcon)
                skipcon = x_outs[layer.skipcon_idx[0]]
                for i in layer.skipcon_idx[1:]:
                    skipcon += x_outs[i]

            if layer.skipcon_ed_idx is not None:
                skipcon_ed = edge_outs[layer.skipcon_ed_idx[0]]
                for i in layer.skipcon_ed_idx[1:]:
                    skipcon_ed += edge_outs[i]
                #print(skipcon.shape, 'hui2')

            if layer.idx in self.fc_skip_indx[1:]:

                graph_pool = global_mean_pool(x, batch)
                graph_pool = graph_pool[batch]
                fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out, graph_pool],1))
                #fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out],1))
                fc_count += 1
            #print(skipcon_ed, 'hui')
            x, edge_attr = layer(x, edge_index, edge_attr, skipcon, skipcon_ed, fc_out, skip_info)

            skipcon = None
            skipcon_ed = None
            x_outs[i] = x
            edge_outs[i] = edge_attr
        
        pred = self.decoder(x)

        return pred
                        
    def loss(self,pred, inp):
        true_flow = inp.flow
        if self.loss_func == 'mae':
            error = torch.mean(torch.abs(true_flow - pred), 1)
        elif self.loss_func == 'mse':
            error = torch.mean(torch.square(true_flow - pred), 1)
        return torch.mean(error) 


                
class FlowGNN_with_FC_BN_RC(nn.Module):  #DeepGCN

    def __init__(self, hidden_size, num_layers, fc_in_dim, fc_out_dim, fcskipcon_indx = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super().__init__()
        self.hidden = hidden_size
        self.hidden_nodes = hidden_nodes
        self.num_layers = num_layers
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.fcskipcon_idx = fcskipcon_indx

        self.node_encoder = nn.LazyLinear(16)
        self.edge_encoder = nn.LazyLinear(16)

        self.node_decoder = nn.LazyLinear(self.out_dim)

        self.conv1 = ProcessorLayer(64,64, self.hidden_nodes)
        self.smooth1 = SmoothingLayer()

        self.layers = nn.ModuleList()

        for i in range(1,self.num_layers-1):

            input_dim = self.hidden + self.geom_in_dim
            if i in  self.fcskipcon_idx:
                input_dim += self.fc_out_dim 
            #print(input_dim)
            conv = GENConv(input_dim, self.hidden, aggr='softmax', 
                        t=1.0, learn_t=True, num_layers=2, norm='batch', msg_norm=True, ckpt_grad = False)
            norm = nn.BatchNorm1d(self.hidden, affine = True)
            act = nn.ReLU()
            layer = DeepGCNLayer(conv, norm, act, block='res')
            self.layers.append(layer)
        
        self.convn = ProcessorLayer(8,16,self.hidden)
        self.smoothn = SmoothingLayer()

    def forward(self, data):
        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        skip_info = x[:,:self.geom_in_dim]
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.smooth1(x, edge_index, edge_attr)
        
        prev_x = x
        for i, layer in enumerate(self.layers, start = 1):
            
            x = torch.cat([prev_x, skip_info], 1)
            #print(x.shape, 1)
            if i in self.fcskipcon_idx:
                x = torch.cat([x, bc], 1) 
                #print(x.shape, 2)
            x = layer(x, prev_x, edge_index, edge_attr)
            #print(x.shape, 3)
            prev_x = x
            # x = torch.cat([x, skip_info], 1)
            #           
        x, edge_attr = self.convn(x, edge_index, edge_attr)
        x, edge_attr = self.smoothn(x, edge_index, edge_attr)

        out = self.node_decoder(x)
        return out

    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)         





class FlowGNN_3Decoder(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, decoder_edge_feat_dims,decoder_nodes_feat_dims, geom_in_dim = 2, hidden_nodes=128):
        super(FlowGNN_3Decoder,self).__init__()

        self.encoder_edge_feat_dims = edge_feat_dims
        self.encoder_num_filters = num_filters
        self.decoder_edges = decoder_edge_feat_dims
        self.decoder_nodes = decoder_nodes_feat_dims
        self.geom_in_dim = geom_in_dim
        #self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        
        self.encoder = nn.ModuleList()
        self.FC_list = nn.ModuleList()
        for i, (ef, nf) in enumerate(zip(self.encoder_edge_feat_dims, self.encoder_num_filters)):
            self.encoder.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.encoder.append(SmoothingLayer(idx = i))

            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

        self.decoder_1 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_2 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_3 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_4 = Decoder(self.decoder_edges, self.decoder_nodes)
        

    def forward(self, data):
        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.encoder:
            x, edge_attr = layer(x, edge_index, edge_attr)

            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])

                if layer.idx == 1:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)

                elif layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    #print(self.FC_list[0], fc_id)
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
        
        x1 = self.decoder_1(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x2 = self.decoder_2(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x3 = self.decoder_3(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x4 = self.decoder_4(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        
        out = torch.cat([x1,x2,x3, x4], 1)
        return out
    
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    

class Decoder(nn.Module):



    def __init__(self, edge_feat_dims, num_filters, hidden_nodes = 128, fc_out_dim = 4, out_dim = 1):
        super(Decoder,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.hidden_nodes = hidden_nodes
        self.fc_out_dim = fc_out_dim

        self.processor= nn.ModuleList()
        self.FC_list = nn.ModuleList()
        self.FC_decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, self.hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, x, edge_attr, edge_index, bc, x_out, skip_info, fc_out = None):

        #x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        

        for layer in self.processor:

            x, edge_attr = layer(x, edge_index, edge_attr)

            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])

                if layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
                
        x = self.FC_decoder(x)
        return x





class ProcessorLayer(MessagePassing):

    def __init__(self, edge_feats, node_feats, hidden_state, msg = False, idx = 0, selu = False):
        super(ProcessorLayer, self).__init__()

        self.name = 'processor'
        self.idx = idx
        activation = nn.ReLU()

        self.msg = msg
        if msg:
            self.messagenorm = MessageNorm(learn_scale = True)

        if selu:
            activation = nn.SELU()

        self.edge_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      #nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(edge_feats)
                                      #activation,
                                      )
                                      
        self.node_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      #nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(node_feats),
                                      activation,
                                      )
        
        
    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr)

        if self.msg:
            out = self.messagenorm(x, out)
            
        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges


    def message(self, x_i, x_j, edge_attr):
        #print(x_i.shape, x_j.shape,edge_attr.shape)

        updated_edges = torch.cat([torch.div(x_i + x_j,2), torch.abs(x_i - x_j)/2, edge_attr], 1)
        #print(updated_edges.shape, 'hui message')
        updated_edges = self.edge_mlp(updated_edges)
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        #print(updated_edges.shape, torch.max(edge_index[0, :]), edge_index[0, :].shape[0], torch.min(edge_index[0, :]), 'hui')
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'sum')
        return out, updated_edges
    import torch
from torch import nn

import pandas
import torch_scatter
from torch_geometric.data import Dataset
from torch_geometric.nn.pool.glob import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import EdgePooling, DeepGCNLayer, GENConv, MessageNorm
import os


class FlowGNN(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN,self).__init__()


        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        
        self.processor = nn.ModuleList()
        
        self.decoder = nn.LazyLinear(out_dim)

        for ef, nf in zip(self.edge_feat_dims, self.num_filters):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes))
            self.processor.append(SmoothingLayer()) 

    
    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,:7], data.edge_index, data.edge_attr

        skip_info = x[:,:self.geom_in_dim]

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



    

class FlowGNN_with_FC_skipcon(nn.Module):   # Модель из отчета росатома с двумя skip_connection
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_skipcon,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc

        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)
            
            if layer.name == 'smoothing' and layer.idx == 6:
                x = x + x_out[1][0]

                x = torch.cat([x, skip_info], 1)
            if layer.name == 'smoothing' and layer.idx == 7:
                x = x + x_out[0][0]
                x = torch.cat([x, skip_info], 1)


            if layer.name == 'smoothing' and layer.idx == 2:
                fc_out1 = self.FC1(bc)
                #print(x.shape, fc_out1.shape)
                x = torch.cat([x, fc_out1], 1)
                #(print(x.shape, layer.idx, 'after FC1'))

            if layer.name == 'smoothing' and layer.idx == 5:
                x = x + x_out[2][0]
                fc2_inp = torch.cat([bc, fc_out1], 1)
                fc_out2 = self.FC2(fc2_inp)
                x = torch.cat([x, fc_out2], 1)
                #(print(x.shape, layer.idx, 'after FC2'))


        # x1, edge_attr1 = layer(x, edge_index, edge_attr)
        
        # x2, edge_attr2 = layer(x1, edge_index, edge_attr1)
        # x3, edge_attr3 = layer(x2, edge_index, edge_attr2)
        # fc1_out = self.FC1(bc)



        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    


    
class FlowGNN_with_FC_skipcon_deep(nn.Module):  # глубокая версия с полносвязными слоями на каждый сверточный слой, а также двумя skip connection
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_skipcon_deep,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC_list = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            

            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

                if layer.idx == 1:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)

                elif layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    #print(self.FC_list[0], fc_id)
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)

            if layer.name == 'smoothing' and layer.idx == 8:
                x = x + x_out[4][0]
                x = torch.cat([x, skip_info], 1)
            if layer.name == 'smoothing' and layer.idx == 10:
                x = x + x_out[2][0]
                x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    

class FlowGNN_with_FC_deep(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, fcskipcon_indx = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_deep,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.fcskipcon_indx = fcskipcon_indx
        self.processor = nn.ModuleList()

        self.FC_list = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i in fcskipcon_indx):      # i%2 == 1
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        #x_out = []
        fc_counter = 0
        for layer in self.processor:
            

            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                #x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

                if layer.idx == self.fcskipcon_indx[0]:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)
                    fc_counter += 1
                    
                elif layer.idx in self.fcskipcon_indx:
                    fc_out = self.FC_list[fc_counter](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
                    fc_counter += 1

            # if layer.name == 'smoothing' and layer.idx == 8:
            #     x = x + x_out[4][0]
            #     x = torch.cat([x, skip_info], 1)
            # if layer.name == 'smoothing' and layer.idx == 10:
            #     x = x + x_out[2][0]
            #     x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
       
class FlowGNN_withFC(nn.Module):  # модель из отчета росатома. Два полносвязный слоя предают выходы в вершины в сдвух местах
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_withFC,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))

        self.FC_list = nn.Modulelist()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc

        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

            if layer.name == 'smoothing' and layer.idx == 2:
                fc_out1 = self.FC1(bc)
                x = torch.cat([x, fc_out1], 1)
                #(print(x.shape, layer.idx, 'after FC1'))

            if layer.name == 'smoothing' and layer.idx == 5:
                fc_out2 = self.FC2(bc)
                x = torch.cat([x, fc_out2], 1)
                #(print(x.shape, layer.idx, 'after FC2'))


        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)    
    
class FlowGNN_skipBC(nn.Module):    # напрямую прокидываютс параметры гран условий в вершины
    
    def __init__(self, edge_feat_dims, num_filters, skipcon_indx  = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_skipBC,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.skipcon_indx = skipcon_indx
        
        self.processor = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,:3], data.edge_index, data.edge_attr,  # здесь были data.bc, убрал для прямых гран.усл.
        
        bc = data.x[:,3:]
        x = torch.cat([x, bc], 1)

        skip_info = x[:,:self.geom_in_dim]

        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                
            # if layer.name == 'smoothing' and layer.idx == 2:
            #     x = torch.cat([x, bc], 1)

            # if layer.name == 'smoothing' and layer.idx == 5:
            #     x = torch.cat([x, bc], 1)
                #if layer.idx%3 == 0:
                if layer.idx in self.skipcon_indx:   #if layer.idx%3 ==0
                    x = torch.cat([x, bc], 1)
            

        pred = self.decoder(x)
        return pred
    
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)

class FlowGNN_with_FC_pool(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_pool,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.Pool1 = EdgePooling(in_channels=64, edge_score_method=EdgePooling.compute_edge_score_softmax)
        #self.Pool2 = EdgePooling(in_channels=128, edge_score_method=EdgePooling.compute_edge_score_softmax)

            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:,:3], data.edge_index, data.edge_attr, data.bc, data.batch
        skip_info = x[:,:self.geom_in_dim]

        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)

                if layer.idx == 2:
                    fc_out1 = self.FC1(bc)
                    x = torch.cat([x, fc_out1], 1)
                    x, edge_index, batch, unpool = self.Pool1(x, edge_index, batch)

                if layer.idx == 4:
                    x, edge_index, batch = self.Pool1.unpool(x, unpool)
                if layer.idx == 5:
                    fc_out2 = self.FC2(bc)
                    x = torch.cat([x, fc_out2], 1)

        pred = self.decoder(x)
        return pred

class FlowGNN_BN_block(nn.Module):
    
    def __init__(self, edge_dims, node_dims, hidden_size = 128,  skipcon_idx = None, skipcon_ed_idx=None,
                 fc_skipcon = False, batchnorm=False, act=False, msg=False, geom_in_dim=2, idx = 0, selu=False):
        super(FlowGNN_BN_block,self).__init__()
        self.conv = ProcessorLayer(edge_dims, node_dims, hidden_size, msg, selu)
        self.smooth = SmoothingLayer()

        self.node_norm_layer = None
        self.act = None

        if batchnorm:
            #self.node_norm_layer = nn.LayerNorm(node_dims)
            self.node_norm_layer = nn.BatchNorm1d(node_dims)
           # self.edge_norm_layer = nn.LayerNorm(edge_dims)
        if act:
            self.act = nn.ReLU()
        
        self.skipcon_idx = skipcon_idx   # int, layer indx  
        self.skipcon_ed_idx = skipcon_ed_idx
        self.fc_skipcon = fc_skipcon  # bool, 
        self.idx = idx

    def forward(self, x, edge_index, edge_attr, skip_connec=None, skip_connec_ed=None, fc_skipconnec=None, skip_info = None):


        if skip_connec is not None:
            x = torch.add(x, skip_connec)
        
        if skip_connec_ed is not None:
            edge_attr = torch.add(edge_attr, skip_connec_ed)

        if skip_info is not None:
            x = torch.cat([x, skip_info],1)

        if self.fc_skipcon:
            x = torch.cat([x, fc_skipconnec],1)

        x, edge_attr = self.conv(x, edge_index, edge_attr)

        if self.act is not None:
            x , edge_attr = self.act(x), self.act(edge_attr)
        
        x , edge_attr = self.smooth(x, edge_index, edge_attr)
        
        if self.node_norm_layer is not None:
            x = self.node_norm_layer(x)
            #edge_attr = self.edge_norm_layer(edge_attr)

        return x, edge_attr

class FlowGNN_BC_FC_block(nn.Module):

    def __init__(self, out_dim, hidden_dim = 32, layers_num = 2):
        super(FlowGNN_BC_FC_block,self).__init__()
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
            x = layer(x)
        return x
        
        

class FlowGNN_deepmod(nn.Module):

    def __init__(self, edge_filters, node_filters, fc_in_dim, fc_out_dim, SC_list=None, SC_ed_list=None,
                 fc_skip_indx=None, batchnorm = None, activations = None, msg = None, selu=None, loss_func = 'mae', hidden_size = 128, geom_in_dim=2,out_dim = 4):
        super(FlowGNN_deepmod,self).__init__()

        self.edge_filters = edge_filters
        self.node_filters = node_filters
        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.SC_ed_list = SC_ed_list
        self.SC_list = SC_list     #[[1,2], None, None, [2,3,4],...]  каждому слою сопоставляется множество слоев из которых приходят SC
        self.fc_skip_indx = fc_skip_indx # [1,3,4,6] слои куда заходит fc слой
        self.batchnorm = batchnorm
        self.activation = activations
        self.selu = selu
        self.msg = msg
        self.hidden_size = hidden_size
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.loss_func = loss_func

        self.layer_list = nn.ModuleList()
        self.FC_list = nn.ModuleList()

        for i, (ef, nf) in enumerate(zip(self.edge_filters, self.node_filters)):

            if i in self.fc_skip_indx:
                self.layer_list.append(FlowGNN_BN_block(ef,nf, skipcon_idx = self.SC_list[i],skipcon_ed_idx=self.SC_ed_list[i], fc_skipcon=True, batchnorm = self.batchnorm[i],
                act = self.activation[i], msg = self.msg[i], idx = i, selu = self.selu[i]))
                self.FC_list.append(FlowGNN_BC_FC_block(self.fc_out_dim,hidden_dim=32, layers_num=2))
            else:  
                self.layer_list.append(FlowGNN_BN_block(ef,nf, skipcon_idx = self.SC_list[i], skipcon_ed_idx=self.SC_ed_list[i], batchnorm = self.batchnorm[i],
                                    act = self.activation[i], msg = self.msg[i],idx = i, selu = self.selu[i]))
            
                
        self.decoder = nn.LazyLinear(self.out_dim)
                    
    
    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:,:3], data.edge_index, data.edge_attr, data.bc, data.batch

        x_outs = {}
        edge_outs = {}
        skipcon=None
        skipcon_ed = None
        skip_info = x[:,:self.geom_in_dim]
        fc_out=None
        if self.fc_skip_indx is not None:
            fc_out = self.FC_list[0](bc)
        fc_count = 1

        for i, layer in enumerate(self.layer_list):
            
            #print(i, 'hui')
            if layer.skipcon_idx is not None:
                #print(skipcon)
                skipcon = x_outs[layer.skipcon_idx[0]]
                for i in layer.skipcon_idx[1:]:
                    skipcon += x_outs[i]

            if layer.skipcon_ed_idx is not None:
                skipcon_ed = edge_outs[layer.skipcon_ed_idx[0]]
                for i in layer.skipcon_ed_idx[1:]:
                    skipcon_ed += edge_outs[i]
                #print(skipcon.shape, 'hui2')

            if layer.idx in self.fc_skip_indx[1:]:

                #graph_pool = global_mean_pool(x, batch)
                #graph_pool = graph_pool[batch]
                #fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out, graph_pool],1))
                fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out],1))
                fc_count += 1
            #print(skipcon_ed, 'hui')
            x, edge_attr = layer(x, edge_index, edge_attr, skipcon, skipcon_ed, fc_out, skip_info)

            skipcon = None
            skipcon_ed = None
            x_outs[i] = x
            edge_outs[i] = edge_attr
        
        pred = self.decoder(x)

        return pred
                        
    def loss(self,pred, inp):
        true_flow = inp.flow
        if self.loss_func == 'mae':
            error = torch.mean(torch.abs(true_flow - pred), 1)
        elif self.loss_func == 'mse':
            error = torch.mean(torch.square(true_flow - pred), 1)
        return torch.mean(error) 


                
class FlowGNN_with_FC_BN_RC(nn.Module):  #DeepGCN

    def __init__(self, hidden_size, num_layers, fc_in_dim, fc_out_dim, fcskipcon_indx = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super().__init__()
        self.hidden = hidden_size
        self.hidden_nodes = hidden_nodes
        self.num_layers = num_layers
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.fcskipcon_idx = fcskipcon_indx

        self.node_encoder = nn.LazyLinear(16)
        self.edge_encoder = nn.LazyLinear(16)

        self.node_decoder = nn.LazyLinear(self.out_dim)

        self.conv1 = ProcessorLayer(64,64, self.hidden_nodes)
        self.smooth1 = SmoothingLayer()

        self.layers = nn.ModuleList()

        for i in range(1,self.num_layers-1):

            input_dim = self.hidden + self.geom_in_dim
            if i in  self.fcskipcon_idx:
                input_dim += self.fc_out_dim 
            #print(input_dim)
            conv = GENConv(input_dim, self.hidden, aggr='softmax', 
                        t=1.0, learn_t=True, num_layers=2, norm='batch', msg_norm=True, ckpt_grad = False)
            norm = nn.BatchNorm1d(self.hidden, affine = True)
            act = nn.ReLU()
            layer = DeepGCNLayer(conv, norm, act, block='res')
            self.layers.append(layer)
        
        self.convn = ProcessorLayer(8,16,self.hidden)
        self.smoothn = SmoothingLayer()

    def forward(self, data):
        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        skip_info = x[:,:self.geom_in_dim]
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.smooth1(x, edge_index, edge_attr)
        
        prev_x = x
        for i, layer in enumerate(self.layers, start = 1):
            
            x = torch.cat([prev_x, skip_info], 1)
            #print(x.shape, 1)
            if i in self.fcskipcon_idx:
                x = torch.cat([x, bc], 1) 
                #print(x.shape, 2)
            x = layer(x, prev_x, edge_index, edge_attr)
            #print(x.shape, 3)
            prev_x = x
            # x = torch.cat([x, skip_info], 1)
            #           
        x, edge_attr = self.convn(x, edge_index, edge_attr)
        x, edge_attr = self.smoothn(x, edge_index, edge_attr)

        out = self.node_decoder(x)
        return out

    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)         





class FlowGNN_3Decoder(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, decoder_edge_feat_dims,decoder_nodes_feat_dims, geom_in_dim = 2, hidden_nodes=128):
        super(FlowGNN_3Decoder,self).__init__()

        self.encoder_edge_feat_dims = edge_feat_dims
        self.encoder_num_filters = num_filters
        self.decoder_edges = decoder_edge_feat_dims
        self.decoder_nodes = decoder_nodes_feat_dims
        self.geom_in_dim = geom_in_dim
        #self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        
        self.encoder = nn.ModuleList()
        self.FC_list = nn.ModuleList()
        for i, (ef, nf) in enumerate(zip(self.encoder_edge_feat_dims, self.encoder_num_filters)):
            self.encoder.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.encoder.append(SmoothingLayer(idx = i))

            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

        self.decoder_1 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_2 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_3 = Decoder(self.decoder_edges, self.decoder_nodes)
        self.decoder_4 = Decoder(self.decoder_edges, self.decoder_nodes)
        

    def forward(self, data):
        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.encoder:
            x, edge_attr = layer(x, edge_index, edge_attr)

            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])

                if layer.idx == 1:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)

                elif layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    #print(self.FC_list[0], fc_id)
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
        
        x1 = self.decoder_1(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x2 = self.decoder_2(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x3 = self.decoder_3(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        x4 = self.decoder_4(x, edge_attr, edge_index, bc, x_out, skip_info, fc_out)
        
        out = torch.cat([x1,x2,x3, x4], 1)
        return out
    
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    

class Decoder(nn.Module):



    def __init__(self, edge_feat_dims, num_filters, hidden_nodes = 128, fc_out_dim = 4, out_dim = 1):
        super(Decoder,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.hidden_nodes = hidden_nodes
        self.fc_out_dim = fc_out_dim

        self.processor= nn.ModuleList()
        self.FC_list = nn.ModuleList()
        self.FC_decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, self.hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, x, edge_attr, edge_index, bc, x_out, skip_info, fc_out = None):

        #x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        

        for layer in self.processor:

            x, edge_attr = layer(x, edge_index, edge_attr)

            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])

                if layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
                
        x = self.FC_decoder(x)
        return x





class ProcessorLayer(MessagePassing):

    def __init__(self, edge_feats, node_feats, hidden_state, msg = False, idx = 0, selu = False):
        super(ProcessorLayer, self).__init__()

        self.name = 'processor'
        self.idx = idx
        activation = nn.ReLU()

        self.msg = msg
        if msg:
            self.messagenorm = MessageNorm(learn_scale = True)

        if selu:
            activation = nn.SELU()

        self.edge_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      #nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(edge_feats)
                                      #activation,
                                      )
                                      
        self.node_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      #nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(node_feats),
                                      activation,
                                      )
        
        
    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr)

        if self.msg:
            out = self.messagenorm(x, out)
            
        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges


    def message(self, x_i, x_j, edge_attr):
        #print(x_i.shape, x_j.shape,edge_attr.shape)

        updated_edges = torch.cat([torch.div(x_i + x_j,2), torch.abs(x_i - x_j)/2, edge_attr], 1)
        #print(updated_edges.shape, 'hui message')
        updated_edges = self.edge_mlp(updated_edges)
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        #print(updated_edges.shape, torch.max(edge_index[0, :]), edge_index[0, :].shape[0], torch.min(edge_index[0, :]), 'hui')
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'sum')
        return out, updated_edges
    

class SmoothingLayer(MessagePassing):

    def __init__(self, idx=0):
        super(SmoothingLayer, self).__init__()
        
        self.name = 'smoothing'
        self.idx = idx

    def forward(self, x, edge_index, edge_attr):

        out_nodes, out_edges = self.propagate(edge_index, x=x, edge_attr = edge_attr)
        return out_nodes, out_edges
  
    def message(self, x_i, x_j):
        updated_edges = (x_i + x_j)/2
        return updated_edges

    def aggregate(self, updated_edges, edge_index):

        node_dim = 0
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'mean')
        return out, updated_edges
import torch
from torch import nn

import pandas
import torch_scatter
from torch_geometric.data import Dataset
from torch_geometric.nn.pool.glob import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import EdgePooling, DeepGCNLayer, GENConv, MessageNorm
import os


class FlowGNN(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN,self).__init__()


        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        
        self.processor = nn.ModuleList()
        
        self.decoder = nn.LazyLinear(out_dim)

        for ef, nf in zip(self.edge_feat_dims, self.num_filters):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes))
            self.processor.append(SmoothingLayer()) 

    
    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,:7], data.edge_index, data.edge_attr

        skip_info = x[:,:self.geom_in_dim]

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



    

class FlowGNN_with_FC_skipcon(nn.Module):   # Модель из отчета росатома с двумя skip_connection
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_skipcon,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc

        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)
            
            if layer.name == 'smoothing' and layer.idx == 6:
                x = x + x_out[1][0]

                x = torch.cat([x, skip_info], 1)
            if layer.name == 'smoothing' and layer.idx == 7:
                x = x + x_out[0][0]
                x = torch.cat([x, skip_info], 1)


            if layer.name == 'smoothing' and layer.idx == 2:
                fc_out1 = self.FC1(bc)
                #print(x.shape, fc_out1.shape)
                x = torch.cat([x, fc_out1], 1)
                #(print(x.shape, layer.idx, 'after FC1'))

            if layer.name == 'smoothing' and layer.idx == 5:
                x = x + x_out[2][0]
                fc2_inp = torch.cat([bc, fc_out1], 1)
                fc_out2 = self.FC2(fc2_inp)
                x = torch.cat([x, fc_out2], 1)
                #(print(x.shape, layer.idx, 'after FC2'))


        # x1, edge_attr1 = layer(x, edge_index, edge_attr)
        
        # x2, edge_attr2 = layer(x1, edge_index, edge_attr1)
        # x3, edge_attr3 = layer(x2, edge_index, edge_attr2)
        # fc1_out = self.FC1(bc)



        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    


    
class FlowGNN_with_FC_skipcon_deep(nn.Module):  # глубокая версия с полносвязными слоями на каждый сверточный слой, а также двумя skip connection
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_skipcon_deep,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC_list = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i%2 == 1):
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            

            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

                if layer.idx == 1:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)

                elif layer.idx%2 == 1:
                    fc_id = layer.idx//2
                    #print(self.FC_list[0], fc_id)
                    fc_out = self.FC_list[fc_id](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)

            if layer.name == 'smoothing' and layer.idx == 8:
                x = x + x_out[4][0]
                x = torch.cat([x, skip_info], 1)
            if layer.name == 'smoothing' and layer.idx == 10:
                x = x + x_out[2][0]
                x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
    

class FlowGNN_with_FC_deep(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, fcskipcon_indx = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_deep,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.fcskipcon_indx = fcskipcon_indx
        self.processor = nn.ModuleList()

        self.FC_list = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 
            if (i in fcskipcon_indx):      # i%2 == 1
                self.FC_list.append(nn.Sequential(nn.LazyLinear( 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim)))

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        fc_out = None
        skip_info = x[:,:self.geom_in_dim]
        #x_out = []
        fc_counter = 0
        for layer in self.processor:
            

            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                #x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

                if layer.idx == self.fcskipcon_indx[0]:
                    fc_out = self.FC_list[0](bc)
                    x = torch.cat([x, fc_out],1)
                    fc_counter += 1
                    
                elif layer.idx in self.fcskipcon_indx:
                    fc_out = self.FC_list[fc_counter](torch.cat([bc,fc_out], 1))
                    x = torch.cat([x, fc_out],1)
                    fc_counter += 1

            # if layer.name == 'smoothing' and layer.idx == 8:
            #     x = x + x_out[4][0]
            #     x = torch.cat([x, skip_info], 1)
            # if layer.name == 'smoothing' and layer.idx == 10:
            #     x = x + x_out[2][0]
            #     x = torch.cat([x, skip_info], 1)

        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)
       
class FlowGNN_withFC(nn.Module):  # модель из отчета росатома. Два полносвязный слоя предают выходы в вершины в сдвух местах
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_withFC,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))

        self.FC_list = nn.Modulelist()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc

        skip_info = x[:,:self.geom_in_dim]
        x_out = []
        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                x_out.append([x, layer.idx])
                #print(x.shape, layer.idx)

            if layer.name == 'smoothing' and layer.idx == 2:
                fc_out1 = self.FC1(bc)
                x = torch.cat([x, fc_out1], 1)
                #(print(x.shape, layer.idx, 'after FC1'))

            if layer.name == 'smoothing' and layer.idx == 5:
                fc_out2 = self.FC2(bc)
                x = torch.cat([x, fc_out2], 1)
                #(print(x.shape, layer.idx, 'after FC2'))


        pred = self.decoder(x)
        return pred
     
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)    
    
class FlowGNN_skipBC(nn.Module):    # напрямую прокидываютс параметры гран условий в вершины
    
    def __init__(self, edge_feat_dims, num_filters, skipcon_indx  = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_skipBC,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.skipcon_indx = skipcon_indx
        
        self.processor = nn.ModuleList()
            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):

        x, edge_index, edge_attr = data.x[:,:3], data.edge_index, data.edge_attr,  # здесь были data.bc, убрал для прямых гран.усл.
        
        bc = data.x[:,3:]
        x = torch.cat([x, bc], 1)

        skip_info = x[:,:self.geom_in_dim]

        for layer in self.processor:
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)
                
            # if layer.name == 'smoothing' and layer.idx == 2:
            #     x = torch.cat([x, bc], 1)

            # if layer.name == 'smoothing' and layer.idx == 5:
            #     x = torch.cat([x, bc], 1)
                #if layer.idx%3 == 0:
                if layer.idx in self.skipcon_indx:   #if layer.idx%3 ==0
                    x = torch.cat([x, bc], 1)
            

        pred = self.decoder(x)
        return pred
    
    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)

class FlowGNN_with_FC_pool(nn.Module):
    def __init__(self, edge_feat_dims, num_filters, fc_in_dim, fc_out_dim, geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super(FlowGNN_with_FC_pool,self).__init__()

        self.edge_feat_dims = edge_feat_dims
        self.num_filters = num_filters
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
            
        self.processor = nn.ModuleList()

        self.FC1 = nn.Sequential(nn.Linear(self.fc_in_dim, 24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.FC2 = nn.Sequential(nn.LazyLinear(24),
                                 nn.LeakyReLU(),
                                 nn.Linear(24, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, self.fc_out_dim))
        
        self.Pool1 = EdgePooling(in_channels=64, edge_score_method=EdgePooling.compute_edge_score_softmax)
        #self.Pool2 = EdgePooling(in_channels=128, edge_score_method=EdgePooling.compute_edge_score_softmax)

            
        self.decoder = nn.LazyLinear(out_dim)

        for i, (ef, nf) in enumerate(zip(self.edge_feat_dims, self.num_filters)):
            self.processor.append(ProcessorLayer(ef,nf, hidden_nodes, idx = i))
            self.processor.append(SmoothingLayer(idx = i)) 

    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:,:3], data.edge_index, data.edge_attr, data.bc, data.batch
        skip_info = x[:,:self.geom_in_dim]

        for layer in self.processor:
            
            x, edge_attr = layer(x, edge_index, edge_attr)
            if layer.name == 'smoothing':
                x = torch.cat([x, skip_info], 1)

                if layer.idx == 2:
                    fc_out1 = self.FC1(bc)
                    x = torch.cat([x, fc_out1], 1)
                    x, edge_index, batch, unpool = self.Pool1(x, edge_index, batch)

                if layer.idx == 4:
                    x, edge_index, batch = self.Pool1.unpool(x, unpool)
                if layer.idx == 5:
                    fc_out2 = self.FC2(bc)
                    x = torch.cat([x, fc_out2], 1)

        pred = self.decoder(x)
        return pred

class FlowGNN_BN_block(nn.Module):
    
    def __init__(self, edge_dims, node_dims, hidden_size = 128,  skipcon_idx = None, skipcon_ed_idx=None,
                 fc_skipcon = False, batchnorm=False, act=False, msg=False, geom_in_dim=2, idx = 0, selu=False):
        super(FlowGNN_BN_block,self).__init__()
        self.conv = ProcessorLayer(edge_dims, node_dims, hidden_size, msg, selu)
        self.smooth = SmoothingLayer()

        self.node_norm_layer = None
        self.act = None

        if batchnorm:
            #self.node_norm_layer = nn.LayerNorm(node_dims)
            self.node_norm_layer = nn.BatchNorm1d(node_dims)
           # self.edge_norm_layer = nn.LayerNorm(edge_dims)
        if act:
            self.act = nn.ReLU()
        
        self.skipcon_idx = skipcon_idx   # int, layer indx  
        self.skipcon_ed_idx = skipcon_ed_idx
        self.fc_skipcon = fc_skipcon  # bool, 
        self.idx = idx

    def forward(self, x, edge_index, edge_attr, skip_connec=None, skip_connec_ed=None, fc_skipconnec=None, skip_info = None):

        if skip_connec is not None:
            x = torch.add(x, skip_connec)
        
        if skip_connec_ed is not None:
            edge_attr = torch.add(edge_attr, skip_connec_ed)

        if skip_info is not None:
            x = torch.cat([x, skip_info],1)

        if self.fc_skipcon:
            x = torch.cat([x, fc_skipconnec],1)
        #print(self.idx, 'hui')
        x, edge_attr = self.conv(x, edge_index, edge_attr)
        #print(self.idx)  
        if self.act is not None:
            x , edge_attr = self.act(x), self.act(edge_attr)
        
        x , edge_attr = self.smooth(x, edge_index, edge_attr)
        
        if self.node_norm_layer is not None:
            x = self.node_norm_layer(x)
            #edge_attr = self.edge_norm_layer(edge_attr)

        return x, edge_attr

class FlowGNN_BC_FC_block(nn.Module):

    def __init__(self, out_dim, hidden_dim = 32, layers_num = 2):
        super(FlowGNN_BC_FC_block,self).__init__()
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
            #print('FC hui')
            #print('FC layer num', i)
            x = layer(x)
           # print('FC', )
        return x
        
        

class FlowGNN_deepmod(nn.Module):

    def __init__(self, edge_filters, node_filters, fc_in_dim, fc_out_dim, SC_list=None, SC_ed_list=None,
                 fc_skip_indx=None, batchnorm = None, activations = None, msg = None, selu=None, loss_func = 'mae', hidden_size = 128, geom_in_dim=2,out_dim = 4):
        super(FlowGNN_deepmod,self).__init__()

        self.edge_filters = edge_filters
        self.node_filters = node_filters
        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.SC_ed_list = SC_ed_list
        self.SC_list = SC_list     #[[1,2], None, None, [2,3,4],...]  каждому слою сопоставляется множество слоев из которых приходят SC
        self.fc_skip_indx = fc_skip_indx # [1,3,4,6] слои куда заходит fc слой
        self.batchnorm = batchnorm
        self.activation = activations
        self.selu = selu
        self.msg = msg
        self.hidden_size = hidden_size
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim
        self.loss_func = loss_func

        self.layer_list = nn.ModuleList()
        self.FC_list = nn.ModuleList()

        for i, (ef, nf) in enumerate(zip(self.edge_filters, self.node_filters)):

            if i in self.fc_skip_indx:
                self.layer_list.append(FlowGNN_BN_block(ef,nf, skipcon_idx = self.SC_list[i],skipcon_ed_idx=self.SC_ed_list[i], fc_skipcon=True, batchnorm = self.batchnorm[i],
                act = self.activation[i], msg = self.msg[i], idx = i, selu = self.selu[i]))
                self.FC_list.append(FlowGNN_BC_FC_block(self.fc_out_dim,hidden_dim=32, layers_num=2))
            else:  
                self.layer_list.append(FlowGNN_BN_block(ef,nf, skipcon_idx = self.SC_list[i], skipcon_ed_idx=self.SC_ed_list[i], batchnorm = self.batchnorm[i],
                                    act = self.activation[i], msg = self.msg[i],idx = i, selu = self.selu[i]))
            
                
        self.decoder = nn.LazyLinear(self.out_dim)
                    
    
    def forward(self, data):
        x, edge_index, edge_attr, bc, batch = data.x[:,:3], data.edge_index, data.edge_attr, data.bc, data.batch

        x_outs = {}
        edge_outs = {}
        skipcon=None
        skipcon_ed = None
        skip_info = x[:,:self.geom_in_dim]
        fc_out=None
        if self.fc_skip_indx is not None:
            fc_out = self.FC_list[0](bc)
        fc_count = 1

        for i, layer in enumerate(self.layer_list):
            
            #print(i, 'hui')
            if layer.skipcon_idx is not None:
                #print(skipcon)
                skipcon = x_outs[layer.skipcon_idx[0]]
                for i in layer.skipcon_idx[1:]:
                    skipcon += x_outs[i]

            if layer.skipcon_ed_idx is not None:
                skipcon_ed = edge_outs[layer.skipcon_ed_idx[0]]
                for i in layer.skipcon_ed_idx[1:]:
                    skipcon_ed += edge_outs[i]
                #print(skipcon.shape, 'hui2')

            if layer.idx in self.fc_skip_indx[1:]:

                graph_pool = global_mean_pool(x, batch)
                graph_pool = graph_pool[batch]
                fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out, graph_pool],1))
                #fc_out = self.FC_list[fc_count](torch.cat([bc, fc_out],1))
                fc_count += 1
            #print(skipcon_ed, 'hui')
            x, edge_attr = layer(x, edge_index, edge_attr, skipcon, skipcon_ed, fc_out, skip_info)

            skipcon = None
            skipcon_ed = None
            x_outs[i] = x
            edge_outs[i] = edge_attr
        
        pred = self.decoder(x)

        return pred
                        
    def loss(self,pred, inp):
        true_flow = inp.flow
        if self.loss_func == 'mae':
            error = torch.mean(torch.abs(true_flow - pred), 1)
        elif self.loss_func == 'mse':
            error = torch.mean(torch.square(true_flow - pred), 1)
        return torch.mean(error) 


                
class FlowGNN_with_FC_BN_RC(nn.Module):  #DeepGCN

    def __init__(self, hidden_size, num_layers, fc_in_dim, fc_out_dim, fcskipcon_indx = [], geom_in_dim = 2, out_dim=3, hidden_nodes=128):
        super().__init__()
        self.hidden = hidden_size
        self.hidden_nodes = hidden_nodes
        self.num_layers = num_layers
        self.geom_in_dim = geom_in_dim
        self.out_dim = out_dim

        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim
        self.fcskipcon_idx = fcskipcon_indx

        self.node_encoder = nn.LazyLinear(16)
        self.edge_encoder = nn.LazyLinear(16)

        self.node_decoder = nn.LazyLinear(self.out_dim)

        self.conv1 = ProcessorLayer(64,64, self.hidden_nodes)
        self.smooth1 = SmoothingLayer()

        self.layers = nn.ModuleList()

        for i in range(1,self.num_layers-1):

            input_dim = self.hidden + self.geom_in_dim
            if i in  self.fcskipcon_idx:
                input_dim += self.fc_out_dim 
            #print(input_dim)
            conv = GENConv(input_dim, self.hidden, aggr='softmax', 
                        t=1.0, learn_t=True, num_layers=2, norm='batch', msg_norm=True, ckpt_grad = False)
            norm = nn.BatchNorm1d(self.hidden, affine = True)
            act = nn.ReLU()
            layer = DeepGCNLayer(conv, norm, act, block='res')
            self.layers.append(layer)
        
        self.convn = ProcessorLayer(8,16,self.hidden)
        self.smoothn = SmoothingLayer()

    def forward(self, data):
        x, edge_index, edge_attr, bc = data.x[:,:3], data.edge_index, data.edge_attr, data.bc
        skip_info = x[:,:self.geom_in_dim]
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.smooth1(x, edge_index, edge_attr)
        
        prev_x = x
        for i, layer in enumerate(self.layers, start = 1):
            
            x = torch.cat([prev_x, skip_info], 1)
            #print(x.shape, 1)
            if i in self.fcskipcon_idx:
                x = torch.cat([x, bc], 1) 
                #print(x.shape, 2)
            x = layer(x, prev_x, edge_index, edge_attr)
            #print(x.shape, 3)
            prev_x = x
            # x = torch.cat([x, skip_info], 1)
            #           
        x, edge_attr = self.convn(x, edge_index, edge_attr)
        x, edge_attr = self.smoothn(x, edge_index, edge_attr)

        out = self.node_decoder(x)
        return out

    def loss(self, pred, inp):
        true_flow = inp.flow
        error = torch.mean(torch.abs(true_flow - pred), 1)
        return torch.mean(error)         





class ProcessorLayer(MessagePassing):

    def __init__(self, edge_feats, node_feats, hidden_state, msg = False, idx = 0, selu = False):
        super(ProcessorLayer, self).__init__()

        self.name = 'processor'
        self.idx = idx
        activation = nn.ReLU()

        self.msg = msg
        if msg:
            self.messagenorm = MessageNorm(learn_scale = True)

        if selu:
            activation = nn.SELU()

        self.edge_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      #nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(edge_feats)
                                      #activation,
                                      )
                                      
        self.node_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      #nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(node_feats),
                                      activation,
                                      )
        
        
    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr)

        if self.msg:
            out = self.messagenorm(x, out)
            
        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges


    def message(self, x_i, x_j, edge_attr):
        #print(x_i.shape, x_j.shape,edge_attr.shape)

        updated_edges = torch.cat([torch.div(x_i + x_j,2), torch.abs(x_i - x_j)/2, edge_attr], 1)
        #print(updated_edges.shape, 'hui message')
        updated_edges = self.edge_mlp(updated_edges)
        return updated_edges
    
    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        #print(updated_edges.shape, torch.max(edge_index[0, :]), edge_index[0, :].shape[0], torch.min(edge_index[0, :]), 'hui')
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'sum')
        return out, updated_edges
    

class SmoothingLayer(MessagePassing):

    def __init__(self, idx=0):
        super(SmoothingLayer, self).__init__()
        
        self.name = 'smoothing'
        self.idx = idx

    def forward(self, x, edge_index, edge_attr):

        out_nodes, out_edges = self.propagate(edge_index, x=x, edge_attr = edge_attr)
        return out_nodes, out_edges
  
    def message(self, x_i, x_j):
        updated_edges = (x_i + x_j)/2
        return updated_edges

    def aggregate(self, updated_edges, edge_index):

        node_dim = 0
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'mean')
        return out, updated_edges


class SmoothingLayer(MessagePassing):

    def __init__(self, idx=0):
        super(SmoothingLayer, self).__init__()
        
        self.name = 'smoothing'
        self.idx = idx

    def forward(self, x, edge_index, edge_attr):

        out_nodes, out_edges = self.propagate(edge_index, x=x, edge_attr = edge_attr)
        return out_nodes, out_edges
  
    def message(self, x_i, x_j):
        updated_edges = (x_i + x_j)/2
        return updated_edges

    def aggregate(self, updated_edges, edge_index):

        node_dim = 0
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'mean')
        return out, updated_edges
