import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import data
import train
import tensorflow as tf
import gpflow
import shutil
shutil.rmtree('experiments/')



#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np
import pandas as pd
from utils import set_seeds

set_seeds(123)

# Hyperparameters
n_epochs = 500
n_inducing = 200
batch_size = 64
K = 6
n_runs = 5

# Experiment
train_names = ['GT', 'DS', 'MACE', 'GLAD', 'MV', 'svgpcr']
embs = ['PLIP', 'CONCH', 'VGG16IN', 'UNI']
# embs = ['VGG16IN', 'UNI']

metrics = {'acc':np.zeros((len(train_names),n_runs)), 'f1': np.zeros((len(train_names),n_runs))}

for emb in embs:
    for ix, train_name in enumerate(train_names):
        class_list = []
        for i in range(n_runs):
            # Save_path
            save_path = 'experiments/{}/{}/{}/'.format(emb,train_name,i)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Load and select data
            load_data = data.AI4SKIN_data(train_name, emb)
            X_tr, X_vl, X_ts = load_data.X_train, load_data.X_val, load_data.X_test
            y_tr, y_tr_ev, y_vl, y_ts = load_data.y_train, load_data.y_train_ev, load_data.y_val, load_data.y_test

            # Model definition and train
            iters_per_epoch = len(X_tr)//batch_size
            model = train.create_setup(X_tr, y_tr, 1.0, 1.0, batch_size, n_inducing, K, train_name)

            best_model, best_val \
                = train.run_adam(model, n_epochs, iters_per_epoch, X_tr,
                                    y_tr_ev, X_vl, y_vl, save_path)


            print('The best model in val obtained\n ' + best_val[0] + ': ' + str(best_val[1]))

            #############################################
            ##############       TEST       #############
            #############################################
            results, classwise = train.evaluate(best_model,X_ts,y_ts)

            print(classwise, pd.DataFrame(classwise))

            metrics['acc'][ix, i] = results["acc"]
            metrics['f1'][ix, i] = results["f1"]
            print("Test:\n", results)

            #Classwise results
            class_list.append(pd.DataFrame(classwise))

        # Compute and save class results
        class_df = pd.concat(class_list)
        by_row_index = class_df.groupby(class_df.index)
        class_mean = by_row_index.mean()
        class_std = by_row_index.std()

        print(class_df)

        class_mean.to_csv('experiments/{}/{}/mean_results.csv'.format(emb, train_name))
        class_std.to_csv('experiments/{}/{}/std_results.csv'.format(emb, train_name))


    acc_mean, acc_std =  metrics['acc'].mean(1), metrics['acc'].std(1)
    f1_mean, f1_std =  metrics['f1'].mean(1), metrics['f1'].std(1)

    results_df = pd.DataFrame({'model': train_names, 'acc_mean': acc_mean, 'acc_std': acc_std, 'f1_mean': f1_mean, 'f1_std': f1_std})
    save_results = "experiments/{}/{}_results.csv".format(emb,emb)
    results_df.to_csv(save_results, index=False)
