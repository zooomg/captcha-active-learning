import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
import IPython.display as ipd
from sklearn.model_selection import train_test_split
import os
import argparse
import wandb
import json


from model.resnet import resnet152, resnet34, resnet18
from utils.dataloader import CaptchaDataset
from utils.criteria import *
from src.train import *
from src.eval import *


CUDA = torch.cuda.is_available()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-verbose', default=0, type=int,
                        help="Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. default: 0")
    parser.add_argument('-epochs', default=5, type=int, help="Number of epoch to train. default: 5")
    parser.add_argument('-batch_size', default=1, type=int, help="Number of samples per gradient update. default: 1")
    parser.add_argument('-chkt_filename', default='./weights/ResNet18-CAPTCHA_ceal', help="Model Checkpoint filename to save. default: \'./weights/ResNet18-CAPTCHA_ceal\'")
    parser.add_argument('-t', '--fine_tunning_interval', default=1, type=int, help="Fine-tuning interval. default: 1")
    parser.add_argument('-T', '--maximum_iterations', default=29, type=int,
                        help="Maximum iteration number. default: 29")
    parser.add_argument('-I', '--initial_annotated_perc', default=0.1, type=float,
                        help="Initial Annotated Samples Percentage. default: 0.1")
    parser.add_argument('-K', '--uncertain_samples_size', default=2000, type=int,
                        help="Uncertain samples selection size. default: 2000")
    parser.add_argument('-uc', '--uncertain_criteria', default='lc',
                        help="Uncertain selection Criteria: \'rs\'(Random Sampling), \'lc\'(Least Confidence), \'ms\'(Margin Sampling), \'en\'(Entropy), \'bvsb\'(Best versus Second Best), \'pmes\'(Pick the Most Even Samples). default: lc")
    parser.add_argument('-gn','--gpu_number', default=0, type=int,
                        help='Number of GPU. default: 0')
    parser.add_argument('-dp', '--data_path', default='./Large_Captcha_Dataset',
                        help="Location of dataset. default: \'./Large_Captcha_Dataset\'")
    parser.add_argument('-wandb', default=False, action="store_true",
                        help="Do you wanna use wandb? just give it True! default:False")
    parser.add_argument('-dc', '--digit_compression', default='mean',
                        help="Compress five digits to one.(mean, median) default: \'mean\'")
    args = parser.parse_args()
    
    return args

def initialize_model(init_dataloader, test_dataloader, args, experiment):
    path = f'{args.chkt_filename}_init.pt'
    device = f'cuda:{args.gpu_number}'
    model = resnet18()
    if CUDA:
        model = model.to(device)
    if os.path.exists(path):
        print('Load initial model')
        model.load_state_dict(torch.load(path))
    else:
        print('Train initial model')
        train(model, init_dataloader, experiment, desc="init train", args=args)
        torch.save(model.state_dict(), path)
        print(path)
    print('Eval initial model')
    evaluate(model, test_dataloader, experiment, args)
    return model

def get_y_pred_probs(model, du, args):
    device = f'cuda:{args.gpu_number}'
    y_pred_probs = []
    for (x, y) in tqdm(du, desc='dataset(unlabeled)'):
        x = x.to(device)
        y_pred_prob = model(x)
        y_pred_prob = list(map(lambda _y: (nn.functional.softmax(_y, dim=-1)).detach().cpu().numpy(), y_pred_prob))
        y_pred_probs.append(y_pred_prob)

    y_pred_probs = np.array(y_pred_probs)
    y_pred_probs = np.transpose(y_pred_probs, (0, 2, 1, 3)).reshape(-1, 5, 62)
    return y_pred_probs

def get_uncertain_samples(y_pred_prob, n_samples, criteria, option='mean'):
    if criteria == 'lc':
        return least_confidence(y_pred_prob, n_samples, option)
    elif criteria == 'ms':
        return margin_sampling(y_pred_prob, n_samples, option)
    elif criteria == 'en':
        return entropy(y_pred_prob, n_samples, option)
    elif criteria == 'rs':
        return None, random_sampling(y_pred_prob, n_samples)
    elif criteria == 'bvsb':
        return bvsb(y_pred_prob, n_samples, option)
    elif criteria == 'pmes':
        return pmes(y_pred_prob, n_samples, option)
    else:
        raise ValueError(
            'Unknown criteria value \'%s\', use one of [\'rs\',\'lc\',\'ms\',\'en\',\'bvsb\',\'pmes\']' % criteria)
                        

def run_active_learning(args, experiment, dl, du, dtest):
    model = initialize_model(dl, dtest, args, experiment)
    
    for i in range(args.maximum_iterations):
        y_pred_probs = get_y_pred_probs(model, du, args)
        
        print("model predictions obtained")

        _, un_idx = get_uncertain_samples(y_pred_probs, args.uncertain_samples_size, criteria=args.uncertain_criteria, option=args.digit_compression)
        
        print("uncertain samples obtained")
        
        un_idx = [du.sampler.indices[idx] for idx in un_idx]
        
        dl.sampler.indices.extend(un_idx)
        
        print("dl upated")
        
        for val in un_idx:
            if val in du.sampler.indices:
                du.sampler.indices.remove(val)
        
        print("du updated")
                        
        print(
            f'Update size of `dl` and `du` by adding uncertain {len(un_idx)} samples in `dl`\nlen(dl): {len(dl.sampler.indices)}, len(du): {len(du.sampler.indices)}'
        )
        
        if i % args.fine_tunning_interval == 0 :
            train(model, dl, experiment, desc="fine-tune", args=args)
        
        
        evaluate(model, dtest, experiment, args)

        print(
            "Iteration: {}, len(dl): {}, len(du): {}".format(
                i, len(dl.sampler.indices),len(du.sampler.indices))
        )



if __name__ == '__main__':
    args = get_args()
    experiment = None
    if args.wandb:
        experiment = wandb.init(project="real-project", entity="captcha-active-learning-jinro", config={
            "learning_rate": 1e-5,
            "epochs": args.epochs,
            "sample_size": args.uncertain_samples_size,
            "batch_size": args.batch_size,
            "uncertain_criteria": args.uncertain_criteria,
            "maximum_iterations": args.maximum_iterations,
            "fine_tunning_interval": args.fine_tunning_interval,
            "initial_annotated_perc": args.initial_annotated_perc,
            "digit_compression": args.digit_compression
        })

    print(json.dumps({
            "learning_rate": 1e-5,
            "epochs": args.epochs,
            "sample_size": args.uncertain_samples_size,
            "batch_size": args.batch_size,
            "uncertain_criteria": args.uncertain_criteria,
            "maximum_iterations": args.maximum_iterations,
            "fine_tunning_interval": args.fine_tunning_interval,
            "initial_annotated_perc": args.initial_annotated_perc,
            "digit_compression": args.digit_compression
    }, indent=4))
        
    dataset = CaptchaDataset(args.data_path, isFilter=True, isCrop=True, isResize=True)
    print('dataset loaded')
    
    random_seed = 123
    test_split = 0.2
    validation_split = args.initial_annotated_perc
    shuffling_dataset = True
    dataset_size = len(dataset) # 82328
    
    indices = list(range(dataset_size))
    test_split_size = int(np.floor(test_split * dataset_size)) # 16465
    validation_split_size = int(np.floor((dataset_size-test_split) * validation_split)) # 8232
    
    if shuffling_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[test_split_size:], indices[:test_split_size]
    train_indices, val_indices = train_indices[validation_split_size:], train_indices[:validation_split_size]
    
    if shuffling_dataset:
        torch.manual_seed(random_seed)
        train_sampler = SubsetRandomSampler(train_indices) # 57631
        valid_sampler = SubsetRandomSampler(val_indices) # 8232
        test_sampler = SubsetRandomSampler(test_indices) # 16465
    else:
        train_sampler = SequentialSampler(train_indices) # 57631
        valid_sampler = SequentialSampler(val_indices) # 8232
        test_sampler = SequentialSampler(test_indices) # 16465
    
    du = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    dl = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)
    dtest = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)
    print(f'dataloader loaded, {len(du)} {len(dl)} {len(dtest)}')
    
    run_active_learning(args, experiment, du=du, dl=dl, dtest=dtest)