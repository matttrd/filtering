import torch as ch
torch = ch
import robustness
from robustness.tools import helpers
from robustness.tools.helpers import AverageMeter, has_attr
from robustness.model_utils import make_and_restore_model
from datasets import DATASETS
from torchvision.utils import make_grid
import torchvision

from graph_image_python import compute_graph
from argparse import ArgumentParser
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import cox
import foolbox
import git  
# from robustness.tools import constants as consts

parser = ArgumentParser()

### MAIN ARGS
parser.add_argument("--data", help="Path to the dataset", type=str, default='/datasets/imagenet')
parser.add_argument("--batch-size", help="Batch size for data loading", type=int, default=128)
parser.add_argument("--workers", help="# data loading workers", type=int, default=10)
parser.add_argument("--data-aug", help="Whether to use data augmentation", type=int, default=0)
parser.add_argument("--resume", help="Resume (where the model is stored)", type=str, required='/models/resnet50_l2_eps0.ckpt')
parser.add_argument("--out-dir", help="Dir where results are stored", type=str, default='/home/matteo/result_filter/defense/')
parser.add_argument("--exp-name", help="Experiment name", type=str, default='test')
parser.add_argument("--dataset", help="Dataset", type=str, default='imagenet')
parser.add_argument("--arch", help="Architecture", type=str, default='resnet50')
parser.add_argument("--attack", help="Type of attack", type=str, default='bpda')
parser.add_argument("--frac", help="Fraction of dataset", type=float, default=1.)
parser.add_argument("--rec-first", help="Fraction of dataset", type=bool, default=False)


### PGD ARGS
parser.add_argument("--outer-attack-steps", help="Number of attack cycles", type=int, default=20)
parser.add_argument("--attack-steps", help="Number of steps for PGD attack", type=int, default=20)
parser.add_argument("--constraint", help="Adv constraint", type=str, default='2') # '2'|'inf'|'unconstrained'|'fourier'|'random_smooth'
parser.add_argument("--eps", help="Adversarial perturbation budget", type=float, default=3)
parser.add_argument("--attack-lr", help="Step size for PGD", type=float, default=0.5)
parser.add_argument("--use-best", help="If 1 (0) use best (final) PGD step as example", type=int, default=1)
parser.add_argument("--random-restarts", help="Number of random PGD restarts for eval", type=int, default=0)
parser.add_argument("--random-start", help="Start with random noise instead of pgd step", type=int, default=0)


#### FILTER ARGS
parser.add_argument("--res", help="Resolution of filtering", type=float, default=0.5)
parser.add_argument("--dim", help="Dimension of resized images", type=int, default=None)



LOGS_SCHEMA = {
    'iteration': int,
    'adv_prec1':float,
    'adv_loss': float}

LOGS_TABLE = 'logs'

def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    # try:
    #     repo = git.Repo(path=osp.dirname(osp.realpath(__file__)),
    #                         search_parent_directories=True)
    #     version = repo.head.object.hexsha
    # except git.exc.InvalidGitRepositoryError:
    #     version = __version__
    # args.version = version

    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    return store


def get_model(arch, resume, dataset, data, model_type):
    data_path = osp.expandvars(data)
    ds = DATASETS[dataset](data_path)
    if model_type == 'original':
        model, checkpoint = make_and_restore_model(arch=arch, dataset=ds, resume_path=resume)
    elif model_type == 'prerelu':
        model, checkpoint = make_and_restore_prerelu_model(arch=arch, dataset_name=dataset, 
                                                           dataset=ds, resume_path=resume)
    if 'module' in dir(model): model = model.module
    return model



def get_loaders(dataset, data, workers, batch_size, data_aug):
    data_path = osp.expandvars(data)
    ds = DATASETS[dataset](data_path)
    train_loader, val_loader = ds.make_loaders(workers,
                    batch_size, data_aug=bool(data_aug), 
                    shuffle_train=False, shuffle_val=True)
    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    return train_loader, val_loader, ds



def get_attack_kwargs(args):
    attack_kwargs = {
            'constraint': args.constraint,
            'eps': args.eps,
            'step_size': args.attack_lr,
            'iterations': args.attack_steps,
            'random_start': args.random_start,
            'custom_loss': None,
            'random_restarts': args.random_restarts,
            'use_best': bool(args.use_best)
            }
    return attack_kwargs



def main():
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)
    store = setup_store_with_metadata(args)
    writer = store.tensorboard

    if store is not None: 
        store.add_table(LOGS_TABLE, LOGS_SCHEMA)

    train_loader, test_loader, ds = get_loaders(args.dataset, args.data, args.workers, args.batch_size, args.data_aug)
    
    model = get_model(args.arch, args.resume, args.dataset, args.data, 'original')
    model.eval()
    
    ##### ------------------------ MODEL EVAL --------------------------
    criterion = ch.nn.CrossEntropyLoss()
    losses    = AverageMeter()
    top1      = AverageMeter()
    top5      = AverageMeter()
    iterator = tqdm(enumerate(test_loader), total=len(test_loader))
        
    if args.attack == 'bpda':
        from attacks import BPDA
        attack_kwargs = get_attack_kwargs(args)
        do_denoise = False
        if args.dataset == 'imagenet':
            do_denoise = True

        attack = partial(BPDA, model=model, args=args, attack_kwargs=attack_kwargs, do_denoise=do_denoise)

    for i, (im, label) in iterator:

        if i > int(len(test_loader) * args.frac):
            break
        
        try:
            adv_im = attack(im, label)

            with ch.no_grad():
                adv_out, _ = model(adv_im, label, 0)
                loss = criterion(adv_out, label)
            if len(loss.shape) > 0: loss = loss.mean()
            losses.update(loss.item(), im.size(0))

            model_logits = adv_out[0] if (type(adv_out) is tuple) else adv_out
            # measure accuracy and record loss
            top1_acc = float('nan')
            top5_acc = float('nan')
            # try:
            maxk = min(5, model_logits.shape[-1])
            if has_attr(args, "custom_accuracy"):
                prec1, prec5 = args.custom_accuracy(model_logits, label)
            else:
                prec1, prec5 = helpers.accuracy(model_logits, label, topk=(1, maxk))
                prec1, prec5 = prec1[0], prec5[0]

            losses.update(loss.item(), im.size(0))
            top1.update(prec1, im.size(0))
            top5.update(prec5, im.size(0))

            top1_acc = top1.avg
            top5_acc = top5.avg
            # except:
            #     warnings.warn('Failed to calculate the accuracy.')
        
            # ITERATOR
            desc = ('Loss {loss.avg:.4f} | '
                    '1 {top1_acc:.3f} | 5 {top5_acc:.3f} ||'.format(
                        loss=losses, top1_acc=top1_acc, top5_acc=top5_acc))

            iterator.set_description(desc)
            iterator.refresh()

            if i == 0:
                nat_grid = make_grid(im[:15, ...])
                # rec_grid = make_grid(rec_img[:15,...])
                adv_grid = make_grid(adv_im[:15, ...])
                # diff = (adv_im - rec_img).abs()[:15, ...]
                # diff_grid = make_grid(diff / diff.reshape(diff.shape[0],-1).max(1)[0][...,None,None,None])
                # writer.add_image('Diff rec-adv', diff_grid, 0)
                writer.add_image('Nat input', nat_grid, 0)
                # writer.add_image('Rec input', rec_grid, 0)
                writer.add_image('Adv input', adv_grid, 0)

            log_info = {
                'iteration': i,
                'adv_prec1':top1_acc,
                'adv_loss':losses.avg,
            }
            # Log info into the logs table
            if store: store[LOGS_TABLE].append_row(log_info)
        except:
            continue
      
    return top1_acc, losses.avg





if __name__ == '__main__':
    main()