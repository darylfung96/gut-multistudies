# from combat.pycombat import pycombat
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
from sklearn.manifold import TSNE
from sklearn import metrics
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import wandb
import matplotlib
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import Lasso, LogisticRegression
from ranger21 import Ranger21

from training import leave_one_dataset_out, train_one_test_all
from lassonet.interfaces import LassoNetClassifier
from visualize_tsne_tensorboard import save_numpy_data

wandb.login()
torch.cuda.is_available()

current_dataset = 'plaque'  # joined or plaque or siamcat
type_data = 'species'  # genus or species
is_impute = False
training_type = 'LODO'  # LODO / TOTA
loss = 'robust'  # ce / robust
optim_name = 'adam'  # adam / ranger
is_batch_loss = True  # None, batch, mmd
autoencoder_sizes = None  # (128, 128, 3)  # or None
M=10

batch_loss_text = 'batch' if is_batch_loss else None

if current_dataset == 'joined':
    data = pd.read_csv(f'raw_combined data/Joined_5_Plaque_{type_data}_level_OTUs_RA_with_disease_n_study.txt', sep='\t')
elif current_dataset == 'plaque':
    data = pd.read_csv(f'raw_combined data/Joined_Plaque_Species_meta_RA_clr.txt', sep='\t')
    # data = pd.read_csv(f'raw_combined data/Plaque_union_Joined5_{type_data}_raw_relative_abundacne.txt', sep='\t')
elif current_dataset == 'siamcat':
    data = pd.read_csv(f'raw_combined data/siamcat_meta_feat.txt', sep='\t')
else:
    raise Exception('Use either "joined" or "plaque" for current_dataset.')


autoencoder_latent_shape = 128
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
# NO NEED TO NORMALIZE FOR CLR
# TODO make sure to change 7 when using different data
# for unique_label in unique_labels:
#     current_study_features = data[data['Study_name'] == unique_label]
#     scale = StandardScaler()
#     scaled_current_study_features = scale.fit_transform(current_study_features.values[:, 7:])  # 3 for siamcat, 7 for plaque
#     current_study_features.iloc[:, 7:] = scaled_current_study_features  # 3 for siamcat, 7 for plaque
#     data[data['Study_name'] == unique_label] = current_study_features
features = data.values[:, 7:].astype(np.float32)  # 3 for siamcat, 7 for plaque
feature_names = data.columns[7:]


###
pca = PCA(2)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
latent_pca = pca.fit_transform(features)
latent_tsne = tsne.fit_transform(features)
study_group_labels = one_hot_decoder_indexes.argmax(1)
plt.clf()

unique_disease_status = np.unique(data['Disease_status'].values)

disease_status_marker = {'CF': 'o', 'ECC': 'x'}
# lassonet with batch loss and robust loss
for disease_status in unique_disease_status:
    status_indexes = np.where(data['Disease_status'] == disease_status)[0]
    current_features = features[status_indexes]
    current_study_group_labels = study_group_labels[status_indexes]
    latent_pca = pca.fit_transform(current_features)

    for label in np.unique(study_group_labels):
        # current_data = data[['Prevotella_salivae', 'Streptococcus_mutans', 'Rothia_aeria', 'Lautropia_mirabilis', 'Corynebacterium_durum',
        #       'Corynebacterium_matruchotii']]
        # current_data = current_data.values
        # latent_pca = pca.fit_transform(current_data)

        indexes = np.where(current_study_group_labels == label)[0]
        plt.xlabel('pca 1')
        plt.ylabel('pca 2')
        # ax.scatter3D(latent_pca[indexes, 0], latent_pca[indexes, 1], latent_pca[indexes, 2],
        #              label=unique_labels[label], color=colors[label])
        plt.scatter(latent_pca[indexes, 0], latent_pca[indexes, 1], s=latent_pca[indexes, 0].shape[0],
                    label=unique_labels[label], color=colors[label], marker=disease_status_marker[disease_status])

# plt.legend()
plt.show()
plt.clf()
# lassonet
for label in np.unique(study_group_labels):
    current_data = data[['Streptococcus_mutans', 'Prevotella_salivae', 'Alloprevotella_sp._HMT_912',
                         'Streptococcus_salivarius', 'Leptotrichia_goodfellowii', 'Rothia_aeria', 'Alloprevotella_sp._HMT_473']]
    current_data = current_data.values
    latent_pca = pca.fit_transform(current_data)

    indexes = np.where(study_group_labels == label)[0]
    plt.xlabel('pca 1')
    plt.ylabel('pca 2')
    # ax.scatter3D(latent_pca[indexes, 0], latent_pca[indexes, 1], latent_pca[indexes, 2],
    #              label=unique_labels[label], color=colors[label])
    plt.scatter(latent_pca[indexes, 0], latent_pca[indexes, 1], s=latent_pca[indexes, 0].shape[0],
                label=unique_labels[label], color=colors[label])
plt.legend()
plt.show()
plt.clf()


# ax = plt.axes(projection ="3d")
for label in np.unique(study_group_labels):
    indexes = np.where(study_group_labels == label)[0]
    plt.xlabel('pca 1')
    plt.ylabel('pca 2')
    # ax.scatter3D(latent_pca[indexes, 0], latent_pca[indexes, 1], latent_pca[indexes, 2],
    #              label=unique_labels[label], color=colors[label])
    plt.scatter(latent_tsne[indexes, 0], latent_tsne[indexes, 1], s=latent_tsne[indexes, 0].shape[0], label=unique_labels[label], color=colors[label])
plt.legend()
plt.show()
###
save_numpy_data(features, np.array(data['Study_name']), 'general')

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
    best_table = []
    f1_best_table = []
    precision_best_table = []
    recall_best_table = []

    for unique_label in unique_labels:
        wandb.init(name=f'WD_separate_{unique_label}_{training_type}_lassonet_clr_{batch_loss_text}_{loss}_{optim_name}', project='wasif_data',
                   group=f'{current_dataset} {type_data} '
                         f'{"impute" if is_impute else ""}'
                         f'num layers: {num_layers} '
                         f'latent:{autoencoder_latent_shape} '
                         f'M: {M} '
                         f'hidden: {hidden_size}', reinit=True)

        # train autoencoder
        np.random.seed(100)
        random.seed(100)
        torch.random.manual_seed(100)
        torch.manual_seed(100)
        if optim_name == 'adam':
            optim = None
        elif optim_name == 'ranger':
            ranger21_func = lambda *args, **kw: Ranger21(*args, **kw, lr=0.001, num_epochs=200, num_batches_per_epoch=5)
            optim = (ranger21_func, ranger21_func)

        if M != 0:
            lassonet = LassoNetClassifier(hidden_dims=tuple([hidden_size] * num_layers), loss=loss, optim=optim,
                                          autoencoder_sizes=autoencoder_sizes, is_batch_loss=is_batch_loss, M=M,
                                          condition_latent_size=one_hot_decoder_indexes.shape[1])
        else:
            lassonet = LogisticRegression(penalty="l1", solver='saga')


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

        # lasso regression
        if M == 0:
            lassonet.fit(train_features, train_labels.argmax(1))
            # lassonet.predict(val_features)

            y_pred = lassonet.predict(val_features)
            current_auc = roc_auc_score(val_labels.argmax(1), y_pred)
            current_f1 = f1_score(val_labels.argmax(1), y_pred)
            current_precision = precision_score(val_labels.argmax(1), y_pred)
            current_recall = recall_score(val_labels.argmax(1), y_pred)

            best_table.append([unique_label, current_auc, 0])
            f1_best_table.append([unique_label, current_f1, 0])
            precision_best_table.append([unique_label, current_precision, 0])
            recall_best_table.append([unique_label, current_recall, 0])
            continue


        path = lassonet.path(train_features, train_labels.argmax(1),
                                 train_one_hot_decoder_indexes=torch.from_numpy(train_one_hot_decoder_indexes))


        # get batch features
        encoded_features = lassonet.get_layer_features(torch.from_numpy(current_data))
        if autoencoder_sizes:
            encoded_features = lassonet.get_encoded_features(torch.from_numpy(current_data))

        # get encoded/batch features plot
        # pca = PCA(2)
        # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        # latent_pca = pca.fit_transform(encoded_features.cpu())
        # latent_tsne = tsne.fit_transform(encoded_features.cpu())
        # study_group_labels = one_hot_decoder_indexes.argmax(1)
        # plt.clf()
        # for label in np.unique(study_group_labels):
        #     indexes = np.where(study_group_labels == label)[0]
        #     plt.xlabel('pca 1')
        #     plt.ylabel('pca 2')
        #     plt.scatter(latent_tsne[indexes, 0], latent_tsne[indexes, 1], s=latent_tsne[indexes, 0].shape[0],
        #                 label=unique_labels[label], color=colors[label])
        # plt.legend()
        # wandb.log({f'{unique_label} feature space': wandb.Image(plt)})
        # plt.clf()
        # plt.cla()
        save_numpy_data(encoded_features.cpu(), np.array(data['Study_name']), f'{unique_label}_{loss}_{optim_name}_{is_batch_loss}')


        n_selected = []
        auc = []
        lambda_ = []
        best_num_features = 0
        best_auc = 0
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        for save in path:
            lassonet.load(save.state_dict)
            y_pred = lassonet.predict(val_features)

            num_selected_features = save.selected.sum()
            current_auc = roc_auc_score(val_labels.argmax(1), y_pred.cpu().numpy())
            if current_auc > best_auc:
                best_auc = current_auc
                best_num_features = num_selected_features
                best_f1 = f1_score(val_labels.argmax(1), y_pred.cpu().numpy())
                best_precision = precision_score(val_labels.argmax(1), y_pred.cpu().numpy())
                best_recall = recall_score(val_labels.argmax(1), y_pred.cpu().numpy())

            n_selected.append(num_selected_features)
            auc.append(current_auc)
            lambda_.append(save.lambda_)
        best_table.append([unique_label, best_auc, best_num_features])
        f1_best_table.append([unique_label, best_f1, best_num_features])
        precision_best_table.append([unique_label, best_precision, best_num_features])
        recall_best_table.append([unique_label, best_recall, best_num_features])

        # plot number of selected features to auc
        fig = plt.figure(figsize=(12, 12))
        plt.subplot(311)
        plt.grid(True)
        plt.plot(n_selected, auc, ".-")
        plt.xlabel("number of selected features")
        plt.ylabel("classification auc")

        wandb.log({f'selected features vs auc': wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()

        # plot feature importances
        n_features = train_features.shape[1]
        importances = lassonet.feature_importances_.numpy()
        order = np.argsort(importances)[::-1]
        importances = importances[order]
        ordered_feature_names = [feature_names[i] for i in order]
        color = np.array(["g"] * n_features)[order]
        top_num = 10

        plt.bar(
            np.arange(n_features)[:top_num],
            importances[:top_num],
            color=color,
        )
        plt.xticks(np.arange(n_features)[:top_num], ordered_feature_names[:top_num], rotation=45, ha='right',
                   fontsize=15)
        # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        # plt.legend(handles, labels)
        plt.ylabel("Feature importance", fontsize=15)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.title(unique_label, fontsize=15)
        plt.savefig(f"feature_importance_{unique_label}.png", dpi=600, bbox_inches='tight')

        wandb.log({f'feature importances {top_num}': wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()
        # plot top 50
        top_num = 50
        plt.rcParams.update({'font.size': 12})
        plt.bar(
            np.arange(n_features)[:top_num],
            importances[:top_num],
            color=color,
        )
        plt.xticks(np.arange(n_features)[:top_num], ordered_feature_names[:top_num], rotation=90)
        # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        # plt.legend(handles, labels)
        plt.ylabel("Feature importance")
        wandb.log({f'feature importances {top_num}': wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()

    wandb_best_table = wandb.Table(data=best_table, columns=['study', 'best auc', 'num of features'])
    wandb.log({f"best_table": wandb.plot.bar(wandb_best_table, "runs", "best", title="best")})
    wandb_best_f1_table = wandb.Table(data=f1_best_table, columns=['study f1', 'best f1', 'num of features f1'])
    wandb.log({f"f1_best_table": wandb.plot.bar(wandb_best_f1_table, "runs", "best", title="best")})
    wandb_best_precision_table = wandb.Table(data=precision_best_table, columns=['study precision', 'best precision', 'num of features precision'])
    wandb.log({f"precision_best_table": wandb.plot.bar(wandb_best_precision_table, "runs", "best", title="best")})
    wandb_best_recall_table = wandb.Table(data=recall_best_table, columns=['study recall', 'best recall', 'num of features recall'])
    wandb.log({f"recall_best_table": wandb.plot.bar(wandb_best_recall_table, "runs", "best", title="best")})

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

