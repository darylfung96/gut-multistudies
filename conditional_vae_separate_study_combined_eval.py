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
import matplotlib.pyplot as plt
import wandb

from network import LightningNetwork, LightningAutoencoderCombineNetwork

wandb.login()

current_dataset = 'plaque'  # joined or plaque
type_data = 'species'  # genus or species
is_impute = False

if current_dataset == 'joined':
    data = pd.read_csv(f'raw_combined data/Joined_5_Plaque_{type_data}_level_OTUs_RA_with_disease_n_study.txt', sep='\t')
elif current_dataset == 'plaque':
    data = pd.read_csv(f'raw_combined data/Plaque_species_raw_count_OTU_table_with_meta_data.txt', sep='\t')
    # data = pd.read_csv(f'raw_combined data/Plaque_union_Joined5_{type_data}_raw_relative_abundacne.txt', sep='\t')
else:
    raise Exception('Use either "joined" or "plaque" for current_dataset.')

autoencoder_latent_shape = 32
hidden_size = 512
num_layers = 3

labels = data.values[:, 1:2]
label_encoder = OneHotEncoder()
encoded_labels = label_encoder.fit_transform(labels).toarray().astype(np.float32)

#TOOD make sure to change 6 when using different data
unique_labels = np.unique(data.values[:, 6])
decoder_indexes = data['Study_name'].apply(lambda x:  np.where(unique_labels == x)[0][0]).values
one_hot_decoder_indexes = np.eye(np.max(decoder_indexes)+1)[decoder_indexes].astype(np.float32)

# normalize within each study group
#TODO make sure to change 7 when using different data
for unique_label in unique_labels:
    current_study_features = data[data['Study_name'] == unique_label]
    scale = StandardScaler()
    scaled_current_study_features = scale.fit_transform(current_study_features.values[:, 7:])
    current_study_features.iloc[:, 7:] = scaled_current_study_features
    data[data['Study_name'] == unique_label] = current_study_features
features = data.values[:, 7:].astype(np.float32)

all_data = [features.astype(np.float32)]  # only get the values for batch correction

for idx, current_data in enumerate(all_data):
    # correction_str = 'before correction' if idx == 0 else 'after correction'
    correction_str = 'after correction'
    folds = KFold(5, shuffle=True)

    wandb.init(name=f'wasif_data_conditional_separate_endend', project='wasif_data', group=f'conditional ae separate endend'
                                                                                    f'{current_dataset} {type_data} '
                                                                         f'{"impute" if is_impute else ""}'
                                                                         f'num layers: {num_layers} '            
                                                                         f'latent:{autoencoder_latent_shape} '
                                                                         f'hidden: {hidden_size}', reinit=True)


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

    for unique_label in unique_labels:
        wandb.init(name=f'wasif_data_conditional_separate_{unique_label} endend', project='wasif_data',
                   group=f'conditional ae separate endend'
                         f'{current_dataset} {type_data} '
                         f'{"impute" if is_impute else ""}'
                         f'num layers: {num_layers} '
                         f'latent:{autoencoder_latent_shape} '
                         f'hidden: {hidden_size}', reinit=True)

        # train autoencoder
        np.random.seed(100)
        random.seed(100)
        torch.random.manual_seed(100)
        network = LightningAutoencoderCombineNetwork(features.shape[1], encoded_labels.shape[1],
                                                     one_hot_decoder_indexes.shape[1], num_layers,
                                                     autoencoder_latent_shape, hidden_size,
                                                     'conditional')
        wandb.watch(network)
        network.set_autoencoder_dataset(autoencoder_dataset)

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

        # train network
        np.random.seed(100)
        random.seed(100)
        torch.random.manual_seed(100)
        wandb.watch(network, log_freq=5)
        trainer = pl.Trainer(max_epochs=200, callbacks=[EarlyStopping(monitor="val_loss", patience=20)])

        train_dataset = TensorDataset(train_features, torch.from_numpy(train_one_hot_decoder_indexes),
                                      train_tensor_encoded_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_dataset = TensorDataset(val_features, torch.from_numpy(val_one_hot_decoder_indexes),
                                    val_tensor_encoded_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        trainer.fit(network, train_dataloader, val_dataloader)

        network.eval()
        network.load_best_weights()

        # onehot_and_val_features = torch.cat([torch.from_numpy(val_one_hot_decoder_indexes),
        #                                           val_features], 1)
        outputs = network([torch.from_numpy(val_one_hot_decoder_indexes), val_features])
        predicted = torch.sigmoid(outputs).detach().numpy()


        # plot roc curve
        fpr, tpr, _ = metrics.roc_curve(val_labels.ravel(), predicted.ravel())
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc})')
        plt.legend()
        wandb.log({'roc_curve': wandb.Image(plt)})
        plt.clf()

        # get optimal threshold
        f1_score_list = []
        auc_list = []
        for i in range(encoded_labels.shape[1]):
            y_true = val_labels[:, i]
            preds = predicted[:, i].copy()

            fpr, tpr, thresholds = metrics.roc_curve(y_true, preds, pos_label=1)

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
            auc = metrics.auc(fpr, tpr)
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

        # area under curve
        auc_average = sum(auc_list) / len(auc_list)
        table = wandb.Table(data=[[f"wasif_data", auc_average]], columns=["runs", "auc"])
        wandb.log({f"auc": wandb.plot.bar(table, "runs", "auc", title="auc")})
