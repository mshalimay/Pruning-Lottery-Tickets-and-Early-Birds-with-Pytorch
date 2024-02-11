from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import utils.utils as u
import models
from prune import reinitialize_weights, unstruct_pruning, structured_pruning, reinit_network


class Timer:
    def __init__(self):
        self.start_time = self.t0 = time.time()
        self.elapsed = 0
    def accumulate(self):
        t = time.time()
        self.elapsed += t - self.t0
        self.t0 = t


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='Train resnet, resmobile or simplenet on mnist or cifar10')

    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')

    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')

    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 160)')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')

    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # parser.add_argument('--no-mps', action='store_true', default=False,
    #                     help='disables macOS GPU training')

    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')

    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')

    parser.add_argument('--print-freq', type=int, default=20, metavar='N',
                        help='print training status every `log-interval` batches. Use -1 to disable (default: 20)')

    parser.add_argument('--save', action='store_true', default=False,
                        help='For Saving checkpoints')

    parser.add_argument('--model', type=str, default='resnetb', choices=['resnet', 'resnetb'],
                        help='neural network to train (default: resnet)')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='dataset to train on (default: cifar10)')
    
    parser.add_argument('--log-train', action='store_true', default=True,
                        help='save training logs to csv (default: True)')

    parser.add_argument('--o', type=str, default='sgd', choices=['adadelta', 'sgd', 'adam', 'adamw'],
                        help='optimizer to use (default: sgd)')

    parser.add_argument('--sched', type=str, default='none', choices=['step', 'plateau', 'cosine', 'cosine_r', 'cyclic', 'manual', 'none'],
                        help='learning rate scheduler to use (default: none)')

    parser.add_argument('--lr_update_epochs', type=int, nargs='+', default=[80, 120],
                        help='Epochs to update learning rate (default: [80, 120]). Only used if --sched is set to "manual"')
    
    parser.add_argument('--w-decay', type=float, default=1e-4,
                        help='weight decay for optimizer (default: 1e-4)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD optimizer (default: 0.9)')

    parser.add_argument('--nest', type = int, default = 1,  choices=[0, 1],
                        help='Use Nesterov momentum in SGD optimizer (default: 1)')

    parser.add_argument('--nw', type = int, default = 2, help = 'Number of workers for data loader (default: 2)')

    parser.add_argument('--pr', type = float, default = 0.5, help = 'Overall pruning rate')

    parser.add_argument('--depth', type = int, default = 20, help = 'Depth of preresnet (default: 20)')

    parser.add_argument('--sr', action='store_true', default=False,
                        help='use subgradient descent on the sparsity-induced penalty term')

    parser.add_argument('--reinit', type=str, default=None, choices=['random', 'init'],
                    help='reinitialize weights after pruning (default: None)')

    parser.add_argument('--priter', type = int, default = 0, 
                        help = ("""Number of pruning iterations (default: 0). 
                                0: trains without pruning. 1: one-shot pruning. >1: iterative pruning."""))

    parser.add_argument('--start-pr-iter', type = int, default = 0,
                        help = 'Start pruning iteration (default: 0)')

    parser.add_argument('--epochs-save', type = int, nargs='+', default = [],
                        help = 'Epochs to save checkpoints (default: [])')

    parser.add_argument('--load', type=str, default=None,
                        help='path to load checkpoint with initial model weights (default: None)')

    parser.add_argument('--prunety', type=str, default='u', choices=['u', 's'],
                        help='u: unstructured s: structured pruning (default: unstructured)')

    # filter
    parser.add_argument('--filter', default='none', type=str, choices=['none', 'lowpass', 'highpass'])
    parser.add_argument('--sigma', default=1.0, type=float, help='gaussian filter hyper-parameter')

    # sparsity
    parser.add_argument('--sparsity_gt', default=0, type=float, help='sparsity controller')

    return parser.parse_args()

def main():
    args = parse_arguments()
    # args.load = 'auto'
    # args.epochs = 1
    # args.priter = 1
    # args.pr = 0.9
    # args.sched = 'manual'
    # args.nw = 5
    # args.save = True
    # # args.reinit = 'init'
    # args.prunety = 's'

    #===========================================================================
    # parse arguments and set up
    #===========================================================================

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()

    # set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # CUDA settings
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}

    if use_cuda:
        cuda_kwargs = {'num_workers': args.nw,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    model_name = args.model.lower()
    # create directory to save models, if specified
    base_ckpt_path = f"./checkpoints/{model_name}_{args.dataset}_depth={args.depth}_batch={args.batch_size}_epoch={args.epochs}"
    checkpoint_path = f"{base_ckpt_path}/priters={args.priter}_pr={args.pr}/"
    if args.save:
        os.makedirs(checkpoint_path, exist_ok=True)
    # create training log directory, if specified
    base_log_path = f"./training_log/{model_name}_{args.dataset}_depth={args.depth}_batch={args.batch_size}_epoch={args.epochs}"
    train_log_path = (f"{base_log_path}/priters={args.priter}_pr={args.pr}/")
    if args.log_train:
        os.makedirs(train_log_path, exist_ok=True)
       
    #===========================================================================
    # load and prepare datasets
    #===========================================================================
    # load and preprocess data
    data_train, data_test = u.load_preprocess_data(args.dataset, args)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(data_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, **test_kwargs)
   
    # warm up GPU before training
    # if use_cuda: u.GPU_warmup(device, warmup_iterations=40)

    #===========================================================================
    # instantiate and prepare model
    #===========================================================================
    # instantiate the desired model 
    net = models.__dict__[args.model](dataset=args.dataset, depth=args.depth)

    # transfer model to GPU, CPU 
    model = net.to(device)
    
    # define the loss function
    loss = torch.nn.CrossEntropyLoss()

    #===========================================================================
    # set prunning configs
    #===========================================================================
    # initialize weights
    best_state_full_network = None

    # random initialization
    if args.load is None:
        u.weight_init(model)
        initial_state_dict = model.state_dict()
    # load checkpoint if specified
    else:
        # try to load the best checkpoint with passed argument or default path
        best_check_file = args.load if os.path.isfile(args.load) else os.path.join(base_ckpt_path, 'priter=0_best.pth.tar')
        if os.path.isfile(best_check_file):
            checkpoint = torch.load(best_check_file)
            best_state_full_network = checkpoint['model_state']      
            initial_state_dict = checkpoint['initial_state']
            model.load_state_dict(best_state_full_network)
            args.start_pr_iter = 1
            print(f'Loaded checkpoint from {best_check_file}.')
        # if no checkpoint is found, train from scratch with random initialization
        else:
            print(f'No best state for full network found. Training a model from scratch.')
            u.weight_init(model)
            initial_state_dict = model.state_dict()
            args.start_pr_iter = 0      
    
    # set pruning rate per iteration
    pr_rate = args.pr
    if args.priter > 1:
        pr_rate = 1-(1-args.pr)**(1/args.priter)

    # TODO: iterative pruning for `structured` case (not implemented yet)
    if args.prunety == 's':
        if args.priter > 1:
            raise NotImplementedError('Structured pruning not implemented for iterative pruning yet.')
        elif args.reinit is not None:
            raise NotImplementedError('Reinitialization not implemented for structured pruning yet.')

    #===========================================================================
    # prune-train-test loop
    #===========================================================================
    masks = None
    best_state = None # keep track of best weights for current pruning iteration

    for priter in range(args.start_pr_iter, args.priter + 1):
        print(f'Pruning iteration {priter}')
        # log file for current pruning iteration
        train_file = os.path.join(base_log_path if priter==0 else train_log_path, f'train_log_{priter}.csv')

        # delete old log file if it exists
        if os.path.isfile(train_file): os.remove(train_file)
        
        # priter != 0 => prune and re-init weights before training
        if priter != 0:
            # prune current model
            if args.prunety == 'u':
                masks, zero_flag = unstruct_pruning(model, pr_rate, masks)
            else:
                cfg, cfg_masks = structured_pruning(model, pr_rate, masks)

            # print total weights and non-zero weights
            nonzeros, total = count_nonzero(model)
            print(f'Pruning iteration {priter} - Non-zero weights: {nonzeros}, Total weights: {total}, % {nonzeros/total*100:.2f} non-zero weights.')

            # record accuracy after pruning, but before reinitialization/retraining
            if args.log_train:
                state_temp = model.state_dict()
                # use the best state of full network to compute test accuracy
                reinitialize_weights(model, masks, best_state_full_network)
                test_loss, test_acc =  test(model, device, loss, test_loader)
                u.log_training(train_file, {'epoch': 0, 'test_loss': test_loss, 'test_acc': test_acc, 
                                            'train_loss': np.nan, 'train_acc': np.nan, 'time_elapsed': np.nan})
                # set model back to the state before computing test accuracy
                model.load_state_dict(state_temp)

            # reinitialize weights/networks if specified
            if args.reinit is not None:      
                # reinit weights if specified
                if args.reinit == 'random': # random reinit
                    reinit_state_dict = u.weight_init(model)
                elif args.reinit == 'init': # reinit to initial state dict (lottery ticket)
                    reinit_state_dict = initial_state_dict
                reinitialize_weights(model, masks, reinit_state_dict)

            if args.prunety == 's':
                new_model = models.__dict__[args.model](dataset=args.dataset, depth=args.depth, cfg=cfg)
                new_model = new_model.to(device)
                reinit_network(new_model=new_model, old_model=model, cfg=cfg, cfg_mask=cfg_masks)
                model = new_model
            
        # set optimizer and scheduler
        optimizer = set_optimizer(model, args)
        scheduler = Scheduler(optimizer, args)
        
        # keep track of best accuracy
        best_accuracy = -float('inf') ; best_loss = float('inf')
  
        # train the network
        for epoch in range(1, args.epochs + 1):
            timer = Timer()
            # train and test the model
            train_loss, train_acc = train(model, loss, device, train_loader, optimizer, epoch, args.sr, masks, args.print_freq)
            test_loss, test_acc = test(model, device, loss, test_loader)

            # accumulate time elapsed
            timer.accumulate()
            
            # step learning rate scheduler
            scheduler.step(metric=best_loss, epoch=epoch)
            scheduler.print_update()
            
            # update best model state
            is_best = test_acc > best_accuracy
            if is_best:
                best_accuracy = test_acc
                best_loss = test_loss
                best_state = model.state_dict()
                if priter == 0:
                    best_state_full_network = model.state_dict()
                    

            # save checkpoints if specified
            if args.save:
                path = base_ckpt_path if priter==0 else checkpoint_path

                # save in specific epochs logic
                if args.epochs_save:
                    # save best model state achieved for each epoch <= args.epochs_save and at the specific epochs
                    if is_best or args.epochs_save:
                        # eg: args.epochs_save = [10,30,50]. 
                        # If is_best, epoch = 5 saves in <...>epoch=10_best.pth.tar; if epoch = 15, saves in <...>epoch=30_best.pth.tar
                        # if not is_best, but epoch = 30, saves in <...>epoch=30.pth.tar
                        annotate_epoch = np.min([e for e in args.epochs_save if e >= epoch])       
                        u.save_checkpoint(
                            {'epoch': epoch, 'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(), 'initial_state': initial_state_dict,
                            'test_loss': test_loss, 'test_acc': test_acc, 'args': args},
                            is_best, annotate_epoch, path, priter)

                # no specific epochs to save => save best model state
                elif is_best:
                    u.save_checkpoint(
                        {'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(), 'initial_state': initial_state_dict,
                        'test_loss': test_loss, 'test_acc': test_acc, 'args': args},
                        is_best, None, path, priter)

            # save training logs to csv if specified
            if args.log_train:         
                data = {'epoch': epoch, 'test_loss': test_loss, 'test_acc': test_acc, 
                        'train_loss': train_loss, 'train_acc': train_acc, 'time_elapsed': timer.elapsed}
                u.log_training(train_file, data)

        # set weights to best state of current network
        # obs: uses the BEST state of previous pruning iteration for pruning (not the last state)
        model.load_state_dict(best_state)

        # reset random seed if full network training
        if priter == 0 and args.seed: 
            torch.manual_seed(args.seed) 

        # best accuracy achieved after all pruning iterations
        print(f'Best accuracy for priter={args.priter}: {best_accuracy:.4f}')
   
#===========================================================================
# Train and test auxiliary functions
#===========================================================================

# Training

def train(model, loss_fun, device, train_loader, optimizer, epoch, sr, masks, print_interval):
    # switch to training mode
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # move data and target to device
        data, target = data.to(device), target.to(device)

        # zero out gradients and forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fun(output, target)

        # backward pass
        loss.backward()
        if sr: updateBN(model)
        # freeze pruned weights by zeroing out their gradients, if any
        # NOTE: in current version, no need to freeze during struct pruning and masks is None accordingly
        if masks is not None:
            freeze_grads(model, masks, device)

        # update weights
        optimizer.step()

        # compute average loss and accuracy
        avg_loss = loss.item() / len(train_loader)        
        accuracy = 100 * (output.argmax(1) == target).float().mean().item()

        # print training status
        if print_interval > 0 and batch_idx % print_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} {100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    return avg_loss, np.round(accuracy, 4)

# additional subgradient descent on the sparsity-induced penalty term
def updateBN(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.grad.data.add_(0.01 * torch.sign(m.weight.data))

# freeze gradients for pruned weights
def freeze_grads(model, masks, device):
    for name, m in model.named_modules():
        if name in masks:
            # ensure mask and model in same device
            mask = masks[name].to(device)  
            # zero out gradients for pruned weights
            m.weight.grad.data.mul_(mask)


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()   # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)       # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.4f}%)\n')
    return test_loss, np.round(accuracy,4)

# functions to set optimizer and scheduler
def set_optimizer(model, args):
    # define the optimizer
    if args.o.lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    elif args.o.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay, dampening=0, nesterov=args.nest)
    elif args.o.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    else:
        raise NotImplementedError
    return optimizer

class Scheduler:
    def __init__(self, optimizer, args):
        self.set_scheduler(optimizer, args)

    def step(self, metric=None, epoch=None):
        if self.scheduler_type is None:
            return

        if self.scheduler is not None:
            if self.use_metric:
                self.scheduler.step(metric)
            else:
                self.scheduler.step()
        elif self.scheduler_type == 'manual' and epoch in self.lr_update_epochs:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma

        # update learning rate
        self.old_lr = self.cur_lr
        self.cur_lr = self.optimizer.param_groups[0]['lr']

    def print_update(self):
        if self.old_lr != self.cur_lr:
            print(f'Learning rate updated to {self.cur_lr:.6f}')
        
    def set_scheduler(self, optimizer, args):
        self.use_metric = False
        self.gamma = args.gamma
        self.scheduler_type = None if args.sched == 'none' else args.sched
        self.cur_lr = args.lr
        self.old_lr = args.lr
        self.optimizer = optimizer

        if self.scheduler_type is None:
            self.scheduler = None
        elif self.scheduler_type == 'step':
            self.scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        elif self.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=self.gamma, patience=10, min_lr=0.0005)
            self.use_metric = True
        elif self.scheduler_type  == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif self.scheduler_type  == 'cosine_r':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0)
        elif self.scheduler_type  == 'cyclic':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-5, max_lr = args.lr, 
                                                        step_size_up = 5, step_size_down=10, mode = "triangular")
        elif self.scheduler_type  == 'manual':
            self.scheduler = None
            self.lr_update_epochs = args.lr_update_epochs
        else:
            raise NotImplementedError




# for debugging
def count_nonzero(model, conv_only=True):
    nonzeros = 0
    total = 0
    for m in model.modules():
        if conv_only and not isinstance(m, torch.nn.Conv2d):
            continue
        # Check if the module has the 'weight' attribute
        if hasattr(m, 'weight') and hasattr(m.weight, 'data'):
            nonzeros += torch.sum(m.weight.data != 0).item()
            total += m.weight.data.numel()
    return nonzeros, total

if __name__ == '__main__':
    main()
