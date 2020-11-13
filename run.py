import time
import numpy as np
from train_eval_titanic import train
from data_cleaning import load_data
from importlib import import_module
import argparse
import torch


parser = argparse.ArgumentParser(description='EE4483 mini project')
parser.add_argument('--model', type=str, required=True, help='choose a model: ')
args = parser.parse_args()

classifier_list = ['GNB', 'KNN', 'SVM', 'DT', 'ET', 'GB', 'RF', 'BC', 'LR', 'RC', 'LNN']



if __name__ == '__main__':
    dataset = 'titanic'
    train_vectors, dev_vectors, train_labels, dev_labels = load_data('train')


    x = import_module('models.' + args.model)
    config = x.Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    model = x.Model(config)
    train(config, model, train_vectors, train_labels, dev_vectors, dev_labels)