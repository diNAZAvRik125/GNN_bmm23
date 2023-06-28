import os

import numpy as np
import torch


def train_model(model, train_dataloader, val_dataloader, device, num_epochs, optimizer, scheduler, best_model_dir,
                collect_grads=False, comp_weigts=False, wait_epochs=2):
    train_loss_hist = []
    val_loss_hist = []
    best_model_loss = np.inf

    # train_cfg = cfg['train_cfg']
    grad_norms = {'node_mlp': [], 'edge_mlp': [], 'FC': []}
    # architecture = dataset_cfg['model_name']
    # dataset_dir = os.path.split(dataset_cfg['dataset_dir'])[0]
    # dataset_name = dataset_cfg['dataset_name']

    #     run = wandb.init(
    #     project = 'gnn_diplom',
    #     name = 'gnn_FC_pool',
    #     config = {
    #     'architecture': architecture,
    #     'dataset_dir': dataset_dir,
    #     'dataset_name' : dataset_name,
    #     'epochs' : num_epochs,
    #     'decay_step': train_cfg['decay_step'],
    #     'decay_factor': train_cfg['decay_factor'],
    #     'batch_size': train_cfg['batch_size']
    #     }
    # )
    # artifact = wandb.Artifact('model', type = 'model')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    model.to(device)

    for epoch in range(num_epochs):

        train_loss, grad_norm = train_step(model, train_dataloader, device, optimizer, collect_grads=collect_grads)

        if epoch == 0 and comp_weigts:
            torch.save(model.state_dict(), os.path.join(best_model_dir, 'init_model.pt'))

        if collect_grads and wait_epochs:
            grad_norms['node_mlp'].append(grad_norm['node_mlp'])
            grad_norms['edge_mlp'].append(grad_norm['edge_mlp'])
            grad_norms['FC'].append(grad_norm['FC'])

        scheduler.step()

        train_loss_hist.append(train_loss.detach().cpu().numpy())
        val_loss = validation_step(model, val_dataloader, device)
        val_loss_hist.append(val_loss.detach().cpu().numpy())

        print(f'Epoch: {epoch}')
        print("train loss", train_loss.item(),
              "val loss", val_loss.item())
        if val_loss < best_model_loss:
            best_model_loss = val_loss
            torch.save(model.state_dict(), os.path.join(best_model_dir, 'best_model.pt'))
            print('Saved best model')
        # run.log({'train_loss': train_loss, 'val_loss': val_loss})
        np.savetxt(os.path.join(best_model_dir, 'train_loss.csv'), train_loss_hist, delimiter=',')
        np.savetxt(os.path.join(best_model_dir, 'val_loss.csv'), val_loss_hist, delimiter=',')
    # torch.save(wandb.run.dir, architecture + '_model.pt')
    # artifact.add_file(os.path.join(best_model_dir, 'best_model.pt'))
    # run.log_artifact(artifact)
    end.record()
    with open(os.path.join(best_model_dir, 'train_time.txt'), 'w') as f:
        f.write(str(start.elapsed_time(end)))
    grad_norms['node_mlp'] = np.mean(np.array(grad_norms['node_mlp']), axis=0)
    grad_norms['edge_mlp'] = np.mean(np.array(grad_norms['edge_mlp']), axis=0)
    grad_norms['FC'] = np.mean(np.array(grad_norms['FC']), axis=0)

    # run.finish()
    return train_loss_hist, val_loss_hist, grad_norms


def train_step(model, dataloader, device, optimizer, collect_grads):
    total_loss = 0
    num_loops = 0
    model.train()
    grads_av_epoch = {'node_mlp': [], 'edge_mlp': [], 'FC': []}
    for batch in dataloader:
        batch_gpu = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch_gpu)
        loss = model.loss(pred, batch_gpu)
        loss.backward()
        if collect_grads:
            grad_norm = collect_gradients(model)

            grads_av_epoch['node_mlp'].append(grad_norm['node_mlp'])
            grads_av_epoch['edge_mlp'].append(grad_norm['edge_mlp'])
            grads_av_epoch['FC'].append(grad_norm['FC'])

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)  # добавил клип градиентов
        optimizer.step()
        total_loss += loss
        num_loops += 1
    total_loss /= num_loops

    # print(grad_norm.shape)

    grads_av_epoch['node_mlp'] = np.mean(np.array(grads_av_epoch['node_mlp']), axis=0)
    grads_av_epoch['edge_mlp'] = np.mean(np.array(grads_av_epoch['edge_mlp']), axis=0)
    grads_av_epoch['FC'] = np.mean(np.array(grads_av_epoch['FC']), axis=0)

    return total_loss, grads_av_epoch


def validation_step(model, dataloader, device):
    total_loss = 0
    num_loops = 0
    # data_list = []
    model.eval()
    for batch in dataloader:
        batch_gpu = batch.to(device)
        with torch.no_grad():
            pred = model(batch_gpu)
            loss = model.loss(pred, batch_gpu)
            total_loss += loss
            num_loops += 1
    total_loss /= num_loops
    return total_loss


def collect_gradients(model):
    grad_norm_conv_ed = []
    grad_norm_conv_nd = []
    grad_norm_fc = []
    grads_dict = {}
    for name, param in model.named_parameters():

        norm = param.grad.norm().cpu()
        if 'conv' in name:
            if 'node' in name and 'weight' in name:
                grad_norm_conv_nd.append(norm)
            if 'edge' in name and 'weight' in name:
                grad_norm_conv_ed.append(norm)
        if 'FC' in name and 'weight' in name:
            grad_norm_fc.append(norm)

    grads_dict['node_mlp'] = grad_norm_conv_nd
    grads_dict['edge_mlp'] = grad_norm_conv_ed
    grads_dict['FC'] = grad_norm_fc

    return grads_dict


def compare_weights(model_dir):
    start_model = torch.load(os.path.join(model_dir, 'init_model.pt'))
    final_model = torch.load(os.path.join(model_dir, 'best_model.pt'))

    weight_diff_conv_ed = []
    weight_diff_conv_nd = []
    weight_diff_fc = []
    weight_diff = {}

    for (name1, params1), (name2, params2) in zip(start_model.items(), final_model.items()):
        diff_norm = torch.norm(torch.subtract(params1, params2), p=1)

        if 'conv' in name1:
            if 'node' in name1 and 'weight' in name1:
                weight_diff_conv_nd.append(diff_norm.item() / torch.numel(params1))
            if 'edge' in name1 and 'weight' in name1:
                weight_diff_conv_ed.append(diff_norm.item() / torch.numel(params1))
        if 'FC' in name1 and 'weight' in name1:
            weight_diff_fc.append(diff_norm.item() / torch.numel(params1))

    weight_diff['node_mlp'] = weight_diff_conv_nd
    weight_diff['edge_mlp'] = weight_diff_conv_ed
    weight_diff['FC'] = weight_diff_fc

    return weight_diff
