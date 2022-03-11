'''
Training and generating condensed dataset strategies
are based on the code provided in
https://github.com/VICO-UoE/DatasetCondensation
by Zhao Bo and Bilen Hakan
'''

import os
import sys
import time
import copy
from attr import define
import numpy as np
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision.models import resnet18

from utils import get_arguments, get_network, get_path, match_loss, at_loss, get_time
from config import get_loops, get_eval_pool
from data import get_dataset, get_daparam, TensorDataset
from diffaugment import DiffAugment, ParamDiffAug
from process import epoch, evaluate_synset


def main(args):
    
    #### Training Configuration ####
    ## Directory Init ##
    args.save_path = get_path(args)
    
    ## Logging Init ##
    logging.basicConfig(filename=os.path.join(args.save_path, 'logging.log'),
                        filemode='a',
                        format='[%(asctime)s] %(levelname)s:%(message)s',
                        datefmt='%m/%d %H:%M:%S',
                        level=logging.INFO)
    
    ## CUDA Init ##
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using {args.device}...')
    
    ## DataLoader Config ##
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    
    if args.dsa:
        logging.info('Augmentation : True')
        logging.info('Augmentation Method : DSA')
        logging.info('DSA Param : \n'+args.dsa_param.print_param())
    

    ## Training Config ##
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    eval_it_pool = np.arange(0, args.Iteration+1, 100).tolist() if args.eval_mode == 'S' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    
    
    #### Start of training ####
    f = open(os.path.join(args.save_path, 'log.txt'), 'w')
    sys.stdout = f # Change the standard output to the file we created.
    
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] # save images in (1,3,32,32) size
        masks_all = torch.load('attention_mask.pt')
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):    # Classify in classes
            indices_class[lab].append(i)
        ## All to CUDA
        # images_all = torch.cat(images_all, dim=0).to(args.device)   # 50000x3x32x32
        # masks_all = torch.cat(masks_all, dim=0).to(args.device)     # 50000x1x32x32
        # labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device) # 50000x1
        ## Batch to CUDA
        images_all = torch.cat(images_all, dim=0)
        masks_all = torch.cat(masks_all, dim=0)
        labels_all = torch.tensor(labels_all, dtype=torch.long)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images(n = ipc) from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle], masks_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc)[0].detach().to(args.device).data
        else:
            print('initialize synthetic data from random noise')

        ''' training '''
        # Visualization logging
        img_loss_vis = []
        train_loss_vis = []
        train_acc_vis = []
        
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())

        best_model_state_dict = None
        best_test_acc = 0
        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(args, model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        net_eval, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        if best_test_acc < acc_test:
                            best_model_state_dict = net_eval.state_dict()
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    torch.save(best_model_state_dict, os.path.join(args.save_path, 'best_eval_model.pt'))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

            ''' Train synthetic data '''

            # Gradient Matching
            net = get_network(args, args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            train_loss_avg = 0
            train_acc_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when training synthetic data.

            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 32  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC)[0] for c in range(num_classes)], dim=0).to(args.device)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                losses = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    
                    losses += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                losses.backward()
                optimizer_img.step()
                loss_avg += losses.item()
                
                ''' update network '''
                train_losses = 0
                train_accs = 0
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    train_loss, train_acc = epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)
                    train_losses += train_loss
                    train_accs += train_acc
                train_loss_avg += train_losses / args.inner_loop
                train_acc_avg += train_accs / args.inner_loop

            train_loss_avg /= (num_classes*args.outer_loop)
            train_acc_avg /= (num_classes*args.outer_loop)
            loss_avg /= (num_classes*args.outer_loop)
            
            img_loss_vis.append(loss_avg)
            train_loss_vis.append(train_loss_avg)
            train_acc_vis.append(train_acc_avg)
            
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
            
            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc_exp%d.pt'%(args.method, args.dataset, args.model, args.ipc, exp)))
        visualize(img_loss_vis, train_loss_vis, train_acc_vis, args.save_path, 'vis_%s_%s_%s_%dipc_exp%d.png'%(args.method, args.dataset, args.model, args.ipc, exp))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
    
    f.close()

def visualize(matching_loss, loss, acc, save_path, exp_name):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,3,figsize=(15,4))
    outer_loop = list(range(1,len(matching_loss)+1))
    axs[0].plot(outer_loop, matching_loss, '-r')
    axs[0].set_title("Matching Loss")
    axs[1].plot(outer_loop, acc,'-g')
    axs[1].set_title("Accuracy")
    axs[2].plot(outer_loop, loss, '-r')
    axs[2].set_title("Training loss")
    plt.suptitle(exp_name.upper())
    plt.savefig(os.path.join(save_path, exp_name+'.png'))
    plt.close('all')

if __name__ == '__main__':
    parser = get_arguments()
    config = parser.parse_args()
    main(config)
