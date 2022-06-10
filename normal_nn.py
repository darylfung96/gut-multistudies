from combat.pycombat import pycombat
import numpy as np
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import metrics
import matplotlib.pyplot as plt
import wandb
from bayes_opt import BayesianOptimization

from network import LightningNetwork

wandb.login()

#TODO remove this
import warnings
warnings.filterwarnings("ignore")

# this is for not end to end training #

current_dataset = 'joined'  # joined or plaque
type_data = 'species'  # genus or species
is_batch_corrected = True

if is_batch_corrected:
    batch_corrected_text = 'batch_corrected'
else:
    batch_corrected_text = ''

if current_dataset == 'joined':
    if is_batch_corrected:
        data = pd.read_csv(f'raw_combined data/Joined_mbImputed_Species_clr_splsda_batch_220525.txt', sep='\t')
        labels = pd.read_csv(f'raw_combined data/Joined_5_Plaque_species_level_OTUs_RA_with_disease_n_study.txt', sep='\t')
        data = pd.merge(labels.iloc[:, :3], data, how='left')
        print('a')
    else:
        data = pd.read_csv(f'raw_combined data/Joined_5_Plaque_{type_data}_level_OTUs_RA_with_disease_n_study.txt', sep='\t')
elif current_dataset == 'plaque':
    data = pd.read_csv(f'raw_combined data/Plaque_union_Joined5_{type_data}_raw_relative_abundacne.txt', sep='\t')
else:
    raise Exception('Use either "joined" or "plaque" for current_dataset.')

autoencoder_latent_shape = 16
hidden_size = 36
num_layers = 2
unique_labels = np.unique(data.values[:, 2])
is_impute = False
imputer = IterativeImputer(max_iter=10, random_state=0, missing_values=0)

features = data.values[:, 3:].astype(np.float32)
if is_impute:
    imputer.fit(features)
    features = imputer.transform(features)
labels = data.values[:, 1:2]
label_encoder = OneHotEncoder()
encoded_labels = label_encoder.fit_transform(labels).toarray().astype(np.float32)
# create different decoders for different study name
wandb.init(name=f'wasif_data_nn', project='wasif_data', group=f'nn'
                                                                                 f'{current_dataset} {type_data} '
                                                                         f'{"impute" if is_impute else ""} '
                                                                         f'num layers: {num_layers}'
                                                                         f'latent:{autoencoder_latent_shape} '
                                                                         f'hidden: {hidden_size}', reinit=True)

unique_labels = np.unique(data.values[:, 2])
data_study_index = data['Study_name'].apply(lambda x: np.where(unique_labels == x)[0][0]).values
one_hot_study_index = np.eye(np.max(data_study_index)+1)[data_study_index].astype(np.float32)

features = data.values[:, 3:].astype(np.float32)

labels = data.values[:, 1:2]
label_encoder = OneHotEncoder()
encoded_labels = label_encoder.fit_transform(labels).toarray().astype(np.float32)
current_data = features  # only get the values for batch correction

params = {
        'num_layers': (1, 10),
        'hidden_size': (32, 1024)
}

# def eval_test(num_layers, hidden_size):
#     all_avg_auc = []
for unique_label in unique_labels:
    y_true_list = None
    pred_list = None

    wandb.init(name=f'wasif_data_nn_{unique_label}', project='wasif_data',
               group=f'nn'
                     f'{current_dataset} {type_data} '
                     f'{"impute" if is_impute else ""}'
                     f'num layers: {num_layers} '
                     f'latent:{autoencoder_latent_shape} '
                     f'hidden: {hidden_size}', reinit=True)

    test_index = data.index[data['Study_name'] == unique_label].values
    train_index = data.index[data['Study_name'] != unique_label].values

    train_features = current_data[train_index]
    val_features = current_data[test_index]

    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    scaled_val_features = scaler.transform(val_features)

    train_labels = encoded_labels[train_index]
    train_tensor_encoded_labels = torch.from_numpy(train_labels)
    val_labels = encoded_labels[test_index]
    val_tensor_encoded_labels = torch.from_numpy(val_labels)



    y_true_list = None
    pred_list = None

    # num_layers = int(num_layers)
    # hidden_size = int(hidden_size)
    # if hidden_size % 2 != 0:
    #     hidden_size += 1

    # train network
    np.random.seed(100)
    random.seed(100)
    torch.random.manual_seed(100)
    network = LightningNetwork(features.shape[1], encoded_labels.shape[1], num_layers, hidden_size)
    wandb.watch(network, log_freq=5)
    trainer = pl.Trainer(max_epochs=200, callbacks=[EarlyStopping(monitor="val_loss")], progress_bar_refresh_rate=0,
                         enable_progress_bar=False, enable_model_summary=False)

    train_tensor_scaled_features = torch.from_numpy(scaled_train_features)
    train_dataset = TensorDataset(train_tensor_scaled_features, train_tensor_encoded_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_tensor_scaled_features = torch.from_numpy(scaled_val_features)
    val_dataset = TensorDataset(val_tensor_scaled_features, val_tensor_encoded_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    trainer.fit(network, train_dataloader, val_dataloader)

    network.eval()
    network.load_best_weights()
    outputs = network(val_tensor_scaled_features)
    predicted = torch.sigmoid(outputs).detach().numpy()

    if y_true_list is None:
        y_true_list = val_labels
        pred_list = predicted
    else:
        y_true_list = np.concatenate([y_true_list, val_labels], 0)
        pred_list = np.concatenate([pred_list, predicted], 0)

    # wandb.init(name=f'wasif_data_multi_decoder', project='wasif_data', group=f'{current_dataset} {type_data} '
    #                                                                          f'latent: {autoencoder_latent_shape}'
    #                                                                          f'hidden: {hidden_size}', reinit=True)
    # plot roc curve
    # encoded_y_true_list = label_encoder.transform(y_true_list).toarray()
    # fpr, tpr, _ = metrics.roc_curve(y_true_list.ravel(), pred_list.ravel())
    # auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, label=f'ROC curve (area = {auc})')
    # plt.legend()
    # wandb.log({'roc_curve': wandb.Image(plt)})
    # plt.clf()

    # get optimal threshold
    f1_score_list = []
    auc_list = []
    all_fpr = []
    all_tpr = []
    for i in range(encoded_labels.shape[1]):
        y_true = y_true_list[:, i]
        preds = pred_list[:, i].copy()

        fpr, tpr, thresholds = metrics.roc_curve(y_true, preds, pos_label=1)
        all_tpr = all_tpr + tpr.tolist()
        all_fpr = all_fpr + fpr.tolist()

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        preds[preds > optimal_threshold] = 1
        preds[preds <= optimal_threshold] = 0
        wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(preds=preds,
                                    y_true=y_true,
                                    title="confusion matrix", class_names=label_encoder.get_feature_names())})

        # f1 score
        f1_score = metrics.f1_score(y_true, preds)
        f1_score_list.append(f1_score)

        # area under curve
        auc = metrics.roc_auc_score(y_true, preds)
        auc_list.append(auc)

        table = wandb.Table(data=[[f"wasif_data",
                                   optimal_threshold]], columns=["runs",
                                                                 f"optimal threshold {label_encoder.get_feature_names()[i]}"])
        wandb.log({f"optimal_threshold_{label_encoder.get_feature_names()[i]}": wandb.plot.bar(table, "runs",
                                                                                               "optimal threshold",
                                                                                      title=f"optimal threshold {label_encoder.get_feature_names()[i]}")})

    # f1 score
    f1_score_average = sum(f1_score_list) / len(f1_score_list)
    table = wandb.Table(data=[[f"wasif_data", f1_score_average]], columns=["runs", "f1 score"])
    wandb.log({f"f1_score": wandb.plot.bar(table, "runs", "f1 score", title="f1 score")})
    #
    # # area under curve
    auc_average = sum(auc_list) / len(auc_list)
    # all_avg_auc.append(auc_average)
    ### for bayesian optimization
    # return auc_average
    ### not for bayesian optimization
    table = wandb.Table(data=[[f"wasif_data", auc_average]], columns=["runs", "auc"])
    wandb.log({f"auc": wandb.plot.bar(table, "runs", "auc", title="auc")})
    plt.plot(fpr, tpr, label=f'ROC curve (area = {sum(auc_list)/len(auc_list)})')
    plt.legend()
    wandb.log({'roc_curve': wandb.Image(plt)})
    plt.clf()
# return sum(all_avg_auc)/len(all_avg_auc)

### for bayesian
# bo = BayesianOptimization(eval_test, params, random_state=100)
# bo.maximize(5, 40)
# eval_test(num_layers, hidden_size)
