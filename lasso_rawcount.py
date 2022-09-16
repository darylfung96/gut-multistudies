from combat.pycombat import pycombat
import numpy as np
import random
import torch
import itertools
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.decomposition import PCA
from sklearn import metrics
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import wandb
import matplotlib
from sklearn.metrics import roc_auc_score

from training import leave_one_dataset_out, train_one_test_all
from lassonet.interfaces import LassoNetClassifier

wandb.login()
torch.cuda.is_available()

current_dataset = 'plaque'  # joined or plaque or siamcat
type_data = 'species'  # genus or species
is_impute = False
training_type = 'LODO'  # LODO / TOTA
loss = 'ce'

if current_dataset == 'joined':
    data = pd.read_csv(f'raw_combined data/Joined_5_Plaque_{type_data}_level_OTUs_RA_with_disease_n_study.txt', sep='\t')
elif current_dataset == 'plaque':
    data = pd.read_csv(f'raw_combined data/Plaque_species_raw_count_OTU_table_with_meta_data.txt', sep='\t')
    # data = pd.read_csv(f'raw_combined data/Plaque_union_Joined5_{type_data}_raw_relative_abundacne.txt', sep='\t')
elif current_dataset == 'siamcat':
    data = pd.read_csv(f'raw_combined data/siamcat_meta_feat.txt', sep='\t')
else:
    raise Exception('Use either "joined" or "plaque" for current_dataset.')


autoencoder_latent_shape = 40
hidden_size = 256
num_layers = 3

colors = ['r', 'g', 'm', 'k', 'y']
labels = data.values[:, 1:2]
label_encoder = OneHotEncoder()
encoded_labels = label_encoder.fit_transform(labels).toarray().astype(np.float32)

#TOOD make sure to change 6 when using different data
unique_labels = np.unique(data.values[:, 6])  # 6 for plaque, 2 for siamcat
if current_dataset == 'siamcat':
    unique_labels = np.array(['metaHIT', 'Lewis_2015', 'He_2017', 'HMP2', 'Franzosa_2019'])
decoder_indexes = data['Study_name'].apply(lambda x:  np.where(unique_labels == x)[0][0]).values
one_hot_decoder_indexes = np.eye(np.max(decoder_indexes)+1)[decoder_indexes].astype(np.float32)

# normalize within each study group
# TODO make sure to change 7 when using different data
for unique_label in unique_labels:
    current_study_features = data[data['Study_name'] == unique_label]
    scale = StandardScaler()
    scaled_current_study_features = scale.fit_transform(current_study_features.values[:, 7:])  # 3 for siamcat, 7 for plaque
    current_study_features.iloc[:, 7:] = scaled_current_study_features  # 3 for siamcat, 7 for plaque
    data[data['Study_name'] == unique_label] = current_study_features
features = data.values[:, 7:].astype(np.float32)  # 3 for siamcat, 7 for plaque
feature_names = data.columns[7:]

###
pca = PCA(3)
latent_pca = pca.fit_transform(features)
study_group_labels = one_hot_decoder_indexes.argmax(1)
plt.clf()
ax = plt.axes(projection ="3d")
for label in np.unique(study_group_labels):
    indexes = np.where(study_group_labels == label)[0]
    plt.xlabel('pca 1')
    plt.ylabel('pca 2')
    ax.scatter3D(latent_pca[indexes, 0], latent_pca[indexes, 1], latent_pca[indexes, 2],
                 label=unique_labels[label], color=colors[label])
plt.legend()
plt.show()
###

all_data = [features.astype(np.float32)]  # only get the values for batch correction

for idx, current_data in enumerate(all_data):
    # correction_str = 'before correction' if idx == 0 else 'after correction'
    correction_str = 'after correction'
    folds = KFold(5, shuffle=True)

    autoencoder_scaler = StandardScaler()
    scaled_data = autoencoder_scaler.fit_transform(current_data)
    autoencoder_trainer = pl.Trainer(max_epochs=100)
    tensor_scaled_features = torch.from_numpy(scaled_data)
    tensor_onehot_decoder_indexes = torch.from_numpy(one_hot_decoder_indexes)
    tensor_encoded_labels = torch.from_numpy(encoded_labels)
    dataset = TensorDataset(tensor_scaled_features, tensor_onehot_decoder_indexes, tensor_encoded_labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    autoencoder_dataset = itertools.cycle(iter(dataloader))

    # autoencoder_trainer.fit(autoencoder_network, dataloader)
    result_matrices = []
    for unique_label in unique_labels:
        wandb.init(name=f'WD_separate_{unique_label}_{training_type}_lassonet_raw_{loss}', project='wasif_data',
                   group=f'{current_dataset} {type_data} '
                         f'{"impute" if is_impute else ""}'
                         f'num layers: {num_layers} '
                         f'latent:{autoencoder_latent_shape} '
                         f'hidden: {hidden_size}', reinit=True)

        # train autoencoder
        np.random.seed(100)
        random.seed(100)
        torch.random.manual_seed(100)
        lassonet = LassoNetClassifier(hidden_dims=tuple([hidden_size] * num_layers), loss=loss)


        test_index = data.index[data['Study_name'] == unique_label].values
        train_index = data.index[data['Study_name'] != unique_label].values

        # scaler = StandardScaler()
        # train_features = scaler.fit_transform(current_data[train_index])
        # val_features = scaler.transform(current_data[test_index])
        train_features = torch.from_numpy(current_data[train_index])
        val_features = torch.from_numpy(current_data[test_index])

        train_one_hot_decoder_indexes = one_hot_decoder_indexes[train_index]
        val_one_hot_decoder_indexes = one_hot_decoder_indexes[test_index]

        train_labels = encoded_labels[train_index]
        train_tensor_encoded_labels = torch.from_numpy(train_labels)
        val_labels = encoded_labels[test_index]
        val_tensor_encoded_labels = torch.from_numpy(val_labels)

        path = lassonet.path(train_features, train_labels.argmax(1))

        n_selected = []
        auc = []
        lambda_ = []
        for save in path:
            lassonet.load(save.state_dict)
            y_pred = lassonet.predict(val_features)
            n_selected.append(save.selected.sum())
            auc.append(roc_auc_score(val_labels.argmax(1), y_pred.cpu().numpy()))
            lambda_.append(save.lambda_)

        fig = plt.figure(figsize=(12, 12))
        plt.subplot(311)
        plt.grid(True)
        plt.plot(n_selected, auc, ".-")
        plt.xlabel("number of selected features")
        plt.ylabel("classification auc")

        wandb.log({f'selected features vs auc': wandb.Image(plt)})
        plt.clf()
        plt.cla()

        # plot feature importances
        n_features = train_features.shape[1]
        importances = lassonet.feature_importances_.numpy()
        order = np.argsort(importances)[::-1]
        importances = importances[order]
        ordered_feature_names = [feature_names[i] for i in order]
        color = np.array(["g"] * n_features)[order]
        top_num = 100
        plt.bar(
            np.arange(n_features)[:top_num],
            importances[:top_num],
            color=color,
        )
        plt.xticks(np.arange(n_features)[:top_num], ordered_feature_names[:top_num], rotation=90)
        colors = {"real features": "g", "fake features": "r"}
        labels = list(colors.keys())
        # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        # plt.legend(handles, labels)
        plt.ylabel("Feature importance")
        plt.rcParams.update({'font.size': 8})
        wandb.log({f'feature importances': wandb.Image(plt)})

        # if training_type == 'LODO':
        #     result_matrix = leave_one_dataset_out(network, data, current_data, label_encoder, encoded_labels, one_hot_decoder_indexes,
        #                        unique_label, unique_labels, colors)
        #     result_matrices.append(result_matrix)
        # else:
        #     result_matrix = train_one_test_all(network, data, current_data, label_encoder, encoded_labels, one_hot_decoder_indexes,
        #                        unique_label, unique_labels, colors)
        #     result_matrices.append(result_matrix)


    # if training_type == 'TOTA':
    #     result_matrices = np.array(result_matrices)
    #     fig, ax = plt.subplots()
    #     im = ax.imshow(result_matrices)
    #     # Show all ticks and label them with the respective list entries
    #     ax.set_xticklabels([''] + unique_labels.tolist())
    #     ax.set_yticklabels([''] + unique_labels.tolist())
    #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #              rotation_mode="anchor")
    #     # Loop over data dimensions and create text annotations.
    #     for i in range(len(unique_labels)):
    #         for j in range(len(unique_labels)):
    #             text = ax.text(j, i, result_matrices[i, j],
    #                            ha="center", va="center", color="w")
    #     fig.tight_layout()
    #     wandb.log({'result_matrix': wandb.Image(plt)})
    #     plt.show()
    #
    # print(result_matrices)

