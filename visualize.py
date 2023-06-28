import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt


class ResultsProfiler():

    def __init__(self, model, model_dir, dataset,test_res_file, device, out_dim = 4):

        self.model = model
        self.model_dir = model_dir
        self.dataset = dataset
        self.device = device
        self.test_results_pth = None
        self.out_dim = out_dim
        self.test_results_pth = test_res_file
        self.metrics = None

        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))
        self.model.to(device)
        


    def plot_loss(self,):
        
        self.train_loss = pd.read_csv(os.path.join(self.model_dir, 'train_loss.csv'), header = None)
        self.val_loss = pd.read_csv(os.path.join(self.model_dir, 'val_loss.csv'), header = None)

        plt.plot(range(len(self.train_loss)), self.train_loss, label = 'train loss')
        plt.plot(range(len(self.val_loss)), self.val_loss, label = 'val loss')
        plt.legend()

    def get_sample_by_id(self, idx):
        data = self.dataset[idx]
        true_flow = self.dataset[idx].flow.numpy()[:,:self.out_dim]
        pred = torch.load(self.test_results_pth)[idx]
        #error = np.abs(true_flow - pred)
        return data, true_flow, pred


    def model_save_history(self,):
        self.val_loss = pd.read_csv(os.path.join(self.model_dir, 'val_loss.csv'), header = None).to_numpy().squeeze()
        val_loss = self.val_loss[0]
        save_losses = []
        for i, value in enumerate(self.val_loss):
            if value < val_loss:
                save_losses.append([i, value])
                val_loss = value
        save_losses = np.array(save_losses)
        plt.plot(save_losses[:,0], save_losses[:,1])
        return save_losses
                

    def plot_hist(self, norm_params: list = None, unnorm_ind: bool = False, norm_ind:bool = False):

        errors = []
        errors_rel = []
        av_speed = []
        obj_maxlen = []
        obj_maxwidth = []

        all_pred = torch.load(self.test_results_pth)
        for i, pred in enumerate(all_pred):
            true_flow = self.dataset[i].flow.numpy()
            
            if norm_params and unnorm_ind:
                true_flow = unnorm(true_flow, norm_params[0], norm_params[1])
                pred = unnorm(pred, norm_params[0], norm_params[1]) 

            if norm_params and norm_ind:

                true_flow = norm(true_flow,norm_params[0], norm_params[1] )
                pred = norm(pred, norm_params[0], norm_params[1]) 



            error = np.mean(np.abs(true_flow - pred), axis = 0)
            error_rel = np.nanmedian(2 * ((pred-true_flow) / ((pred) + (true_flow))), axis=0)
            #print(error.shape)
            obj = self.dataset[i].x[:,2]
            data_x_obj = self.dataset[i].x[obj == 1][:,:2]
            max_obj_xy= torch.max(data_x_obj,0)[0]
            min_obj_xy= torch.min(data_x_obj,0)[0]
            max_obj_len_wid = max_obj_xy - min_obj_xy
            av_speed_val = np.mean(true_flow[:,0])
            av_speed.append(av_speed_val)
            obj_maxlen.append(max_obj_len_wid[0])
            obj_maxwidth.append(max_obj_len_wid[1])
            errors.append(error)
            errors_rel.append(error_rel)
        
        errors = np.array(errors)
        errors_rel = np.array(errors_rel)
        #print(errors)
        av_speed = np.array(av_speed)
        obj_maxlen = np.array(obj_maxlen)
        obj_maxwidth = np.array(obj_maxwidth)

        u_err, v_err = errors[:,0], errors[:,1]
        p_err, t_err = errors[:,2], errors[:,3]

        u_mean, u_std = np.mean(u_err), np.std(u_err)
        v_mean, v_std = np.mean(v_err), np.std(v_err)
        p_mean, p_std = np.mean(p_err), np.std(p_err)
        t_mean, t_std = np.mean(t_err), np.std(t_err)


        figsize = (10,10)
        fig, axs = plt.subplots(2,2, figsize = figsize)
        axs[0][0].hist(u_err, bins = 200)
        axs[0][0].set_title(f'U  Mean:{u_mean:.6f} std:{u_std:.6f}')
        axs[0][1].hist(v_err, bins = 200)
        axs[0][1].set_title(f'V  Mean:{v_mean:.6f} std:{v_std:.6f}')
        axs[1][0].hist(p_err, bins = 200)
        axs[1][0].set_title(f'P  Mean:{p_mean:.6f} std:{p_std:.6f}')
        axs[1][1].hist(t_err, bins = 200)
        axs[1][1].set_title(f'T  Mean:{t_mean:.6f} std:{t_std:.6f}')
        plt.show()

        return errors, errors_rel,  av_speed, obj_maxlen,obj_maxwidth



    def predict_and_save(self, test_results_pth = None, unnorm_prms = None, load = False):

        data_list = []
        num_samples = len(self.dataset)
        mae_losses = []
        mse_losses = []
        rmae_losses = []
        test_loader = DataLoader(self.dataset, batch_size=1, shuffle = False)
        self.model.eval()

        
        for i, sample in enumerate(test_loader):
            #print(sample)
            sample.to(self.device)
            with torch.no_grad():
                pred = self.model(sample)
                pred = pred.detach().cpu().numpy()
                sample.cpu()
                data_list.append(pred)
                true_flow = sample.flow.numpy()[:,:self.out_dim]
                if unnorm_prms is not None:
                    pred = unnorm(pred, unnorm_prms[0], unnorm_prms[1])  #0 - min_uvp, 1 - range_uvp
                    true_flow = unnorm(true_flow, unnorm_prms[0], unnorm_prms[1]) #0 - min_uvp, 1 - range_uvp
            
                mae_losses.append(np.mean(np.abs(pred - true_flow), axis = 0))
                mse_losses.append(np.mean(np.square(pred - true_flow), axis = 0))
                #diff = np.abs(pred - true_flow)/np.abs(true_flow + 0.001)
                diff = np.abs(2 * ((pred-true_flow) / ((pred) + (true_flow))))
                rmae_losses.append(np.nanmedian(diff, axis = 0))
        
                
        self.max_error_idx_val = [np.argmax(np.mean(mae_losses, axis = 1)), np.max(np.sum(mae_losses, axis=1))]
        self.min_error_idx_val = [np.argmin(np.sum(mae_losses, axis = 1)), np.min(np.sum(mae_losses, axis=1))]

        mae = np.mean(mae_losses, axis=0)
        mse = np.mean(mse_losses, axis=0)
        rmae = np.mean(rmae_losses, axis=0)
        std = np.std(mae_losses, axis=0)

        metrics = {'MAE': mae, 'MSE': mse, 'RMAE': rmae, 'STD': std}
        self.test_results_pth = test_results_pth
        self.metrics = metrics

        if test_results_pth is not None:
            torch.save(data_list, test_results_pth)

        return data_list, metrics
    

    
    def plot(self, idx, norm_coord, norm_flow, norm_params = None):
        
        data = self.dataset[idx].x.numpy()
        true_flow = self.dataset[idx].flow.numpy()[:,:self.out_dim]
        pred = torch.load(self.test_results_pth)[idx]
        #error = 2 * ((pred-true_flow) / (np.abs(pred) + np.abs(true_flow)))
        error = np.abs(true_flow - pred)

        s_y, s_x = self.out_dim, 3
        figsize = (20, 6)
        fig, ax = plt.subplots(s_y,s_x, sharex=True, sharey=True, figsize=figsize)
        
        axs1 = [ax[0][0], ax[1][0], ax[2][0], ax[3][0]]
        axs2 = [ax[0][1], ax[1][1], ax[2][1],ax[3][1]]
        axs3 = [ax[0][2], ax[1][2], ax[2][2],ax[3][2]]

        # axs1 = [ax[0], ax[1], ax[2], ax[3]]
        # axs2 = [ax[4], ax[5], ax[6],ax[7]]
        # axs3 = [ax[8], ax[9], ax[10],ax[1]]        


        
        plot_data(data, pred, fig, axs1[:self.out_dim], norm_coord, norm_flow, norm_params)    
        plot_data(data, true_flow, fig, axs2[:self.out_dim], norm_coord, norm_flow, norm_params)
        plot_data(data, error, fig, axs3[:self.out_dim], norm_coord, norm_flow, norm_params)




def plot_data(data, flow, fig, axs, norm_coord, norm_flow, norm_params = None):

    xy, obj = data[:,:2], data[:,2]
    
    if norm_params is not None:
        
        dom_bound_min, dom_bound_max = norm_params[0], norm_params[1] 
        min_uvpt, range_flow = norm_params[2], norm_params[3] 
        range_xy = dom_bound_max-dom_bound_min
        print(range_xy)
        if norm_coord:
            xy = unnorm(xy, dom_bound_min, range_xy)
        if norm_flow:
            flow = unnorm(flow, min_uvpt, range_flow)
    
    x_obj = xy[:,0][obj == 1]
    y_obj = xy[:,1][obj == 1]

    x_obj = np.append(x_obj, x_obj[0])
    y_obj = np.append(y_obj, y_obj[0])
    # xy_obj = xy[obj == 1]
    # xy_obj = np.concatenate([xy, xy[0].reshape(1,-1)], axis = 0)
    #print(xy.shape, xy_obj.shape, obj.shape)

    for i in range(flow.shape[1]):
        
        pcm = axs[i].tripcolor(xy[:,0],xy[:,1], flow[:,i], cmap='viridis', shading='gouraud')
        #cp = axs[i].contourf(xy[:,0],xy[:,1], flow[:,i])
        fig.colorbar(pcm, ax=axs[i]).ax.tick_params(labelsize=16)
        fig.tight_layout()
        axs[i].fill(x_obj,y_obj, 'w')
        axs[i].plot(x_obj,y_obj, 'k')
        axs[i].tick_params(labelsize = 16)
        axs[i].margins(0,0)
        

def cals_dataset_stats(dataset):
    av_speed = []
    obj_maxlen = []
    obj_maxwidth = []
    for i, smpl in enumerate(dataset):
        data_x = smpl.x
        data_flow = smpl.flow
        obj = data_x[:,2]
        data_x_obj = data_x[obj == 1][:,:2]
        max_obj_xy= torch.max(data_x_obj,0)[0]
        min_obj_xy= torch.min(data_x_obj,0)[0]
        max_obj_len_wid = max_obj_xy - min_obj_xy
        av_speed_val = torch.mean(data_flow[:,0])
        av_speed.append(av_speed_val)
        obj_maxlen.append(max_obj_len_wid[0])
        obj_maxwidth.append(max_obj_len_wid[1])
    
    av_speed = np.array(av_speed)
    obj_maxlen = np.array(obj_maxlen)
    obj_maxwidth = np.array(obj_maxwidth)

    return av_speed, obj_maxlen,obj_maxwidth 
    
    
def unnorm(data, minval, range):

    res = (data * range) + minval
    return res

def norm(data, minval, range):
    res = (data/range) - minval
    return res
