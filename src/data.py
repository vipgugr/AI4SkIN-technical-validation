import pickle
import numpy as np
import random
import tensorflow as tf
import os
import pandas as pd
from sklearn.decomposition import PCA

def unpack_data(data):
    return data['features'], data['label_list'], data['names']

def unpack_data_grx_train(data):
    return data['features'], np.array(data['label_list']), data['names'], data['MV'], data['EM']

def unpack_data_grx_test(data):
    return data['features'], np.array(data['label_list']), data['names'], data['MV'], data['EM'], data['GT']

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed()
  tf.set_random_seed(seed)

def get_emb_av(values, folder):
    embd = [np.load(os.path.join(folder, f_name + ".npy"), allow_pickle=True) for f_name in values]
    return np.array([np.mean(patch,axis=0) for patch in embd])

def get_features(df, emb_mode):
    raise("Update the data path for the embeddings")
    emb_path = 'your/route/to/the/data' + emb_mode
    names = list(df['WSI'].values)
    X = get_emb_av(names, emb_path)
    return X, names

def process_labels_cr_v1(labels):
    labels_cr = []
    labels_mask = np.zeros((labels.shape[0], labels.shape[1]))
    for ix, x in enumerate(labels):
        labels_im = []
        for iy, y in enumerate(x):
            if y != -1:
                labels_im.append([iy, y])
            else:
                labels_mask[ix, iy] = 0
        labels_cr.append(np.array(labels_im))
    return labels_cr, labels_mask

class AI4SKIN_data:
    def __init__(self, label, emb_mode):

        
        raise("Update the data path for the labels")
        # load data
        label_path = '/route/to/labels'

        # Splits
        train_df = pd.read_csv(label_path + 'train.csv')
        val_df = pd.read_csv(label_path + 'val.csv')
        test_df = pd.read_csv(label_path + 'test.csv')

        if label == 'svgpcr':
            self.y_train, _ = process_labels_cr_v1(train_df.iloc[:,1:11].values)
        else:
            self.y_train = train_df[label].values

        self.y_train_ev = train_df['GT']

        X_train, train_names = get_features(train_df, emb_mode)
        X_val, val_names = get_features(val_df, emb_mode)
        X_test, test_names = get_features(test_df, emb_mode)

        self.y_val = val_df['GT'].values
        self.y_test = test_df['GT'].values

        # Normalize

        self.m, self.std = X_train.mean(0), X_train.std(0)

        self.X_train = self._norm(X_train)
        self.X_val = self._norm(X_val)
        self.X_test = self._norm(X_test)

    def _norm(self, data):
        return (data-self.m)/self.std
