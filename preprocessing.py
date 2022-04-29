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
from sklearn import metrics
import matplotlib.pyplot as plt
import wandb

from network import LightningNetwork, LightningAutoencoderNetwork

wandb.login()

current_dataset = 'joined'  # joined or plaque
type_data = 'species'  # genus or species

if current_dataset == 'joined':
    data = pd.read_csv(f'raw_combined data/Joined_5_Plaque_{type_data}_level_OTUs_RA_with_disease_n_study.txt', sep='\t')
elif current_dataset == 'plaque':
    data = pd.read_csv(f'raw_combined data/Plaque_union_Joined5_{type_data}_raw_relative_abundacne.txt', sep='\t')
else:
    raise Exception('Use either "joined" or "plaque" for current_dataset.')

autoencoder_latent_shape = 128
features = data.values[:, 3:].astype(np.float32)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

labels = data.values[:, 1:2]
label_encoder = OneHotEncoder()
encoded_labels = label_encoder.fit_transform(labels).toarray().astype(np.float32)


unique_labels = np.unique(data.values[:, 2])
# for unique_label in unique_labels:
#     wandb.init(name=unique_label, project='wasif_data', group=f'before correction: {current_dataset} {type_data}', reinit=True)
#     pca = PCA(2)
#     pca_features = pca.fit_transform(scaled_features[data.values[:, 2] == unique_label])
#     table = wandb.Table(data=pca_features, columns=['pca1', 'pca2'])
#     wandb.log({"scatter_plot": wandb.plot.scatter(table=table, x="pca1", y="pca2", title="data")})
#
#
batch = [np.where(unique_labels == item)[0][0] for item in data.values[:, 2]]
pd_scaled_features = pd.DataFrame(scaled_features, columns=data.columns[3:])
scaled_features_corrected = pycombat(pd_scaled_features.transpose(), batch)
scaled_features_corrected = scaled_features_corrected.transpose()
# for unique_label in unique_labels:
#     wandb.init(name=unique_label, project='wasif_data', group=f'after correction: {current_dataset} {type_data}', reinit=True)
#     pca = PCA(2)
#     pca_features = pca.fit_transform(scaled_features_corrected[data.values[:, 2] == unique_label])
#     table = wandb.Table(data=pca_features, columns=['pca1', 'pca2'])
#     wandb.log({"scatter_plot": wandb.plot.scatter(table=table, x="pca1", y="pca2", title="data")})


# all_data = [scaled_features, scaled_features_corrected.values.astype(np.float32)]
all_data = [scaled_features_corrected.values.astype(np.float32)]  # only get the values for batch correction

for idx, current_data in enumerate(all_data):
    # correction_str = 'before correction' if idx == 0 else 'after correction'
    correction_str = 'after correction'
    folds = KFold(5, shuffle=True)

    y_true_list = None
    pred_list = None

    wandb.init(name=f'wasif_data_{idx}', project='wasif_data', group=f'{correction_str}: {current_dataset} {type_data}', reinit=True)
    for idx, (train_index, val_index) in enumerate(folds.split(current_data, encoded_labels)):
        # train autoencoder
        np.random.seed(100)
        random.seed(100)
        torch.random.manual_seed(100)
        autoencoder_network = LightningAutoencoderNetwork(features.shape[1], encoded_labels.shape[1], 2, autoencoder_latent_shape)
        wandb.watch(autoencoder_network)
        autoencoder_trainer = pl.Trainer(max_epochs=200)
        tensor_scaled_features = torch.from_numpy(current_data)
        tensor_encoded_labels = torch.from_numpy(encoded_labels)
        dataset = TensorDataset(tensor_scaled_features, tensor_encoded_labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        autoencoder_trainer.fit(autoencoder_network, dataloader)

        # train network
        np.random.seed(100)
        random.seed(100)
        torch.random.manual_seed(100)
        # network = LightningNetwork(features.shape[1], encoded_labels.shape[1], 2, 128)
        network = LightningNetwork(autoencoder_latent_shape, encoded_labels.shape[1], 2, 128)
        wandb.watch(network, log_freq=5)
        trainer = pl.Trainer(max_epochs=200, callbacks=[EarlyStopping(monitor="val_loss")])

        train_tensor_scaled_features = torch.from_numpy(current_data[train_index])
        with torch.no_grad():
            train_tensor_scaled_features, _ = autoencoder_network(train_tensor_scaled_features)
        train_tensor_encoded_labels = torch.from_numpy(encoded_labels[train_index])
        train_dataset = TensorDataset(train_tensor_scaled_features, train_tensor_encoded_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_tensor_scaled_features = torch.from_numpy(current_data[val_index])
        with torch.no_grad():
            val_tensor_scaled_features, _ = autoencoder_network(val_tensor_scaled_features)
        val_tensor_encoded_labels = torch.from_numpy(encoded_labels[val_index])
        val_dataset = TensorDataset(val_tensor_scaled_features, val_tensor_encoded_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        trainer.fit(network, train_dataloader, val_dataloader)

        outputs = network(val_tensor_scaled_features)
        predicted = torch.sigmoid(outputs).detach().numpy()

        if y_true_list is None:
            y_true_list = labels[val_index]
            pred_list = predicted
        else:
            y_true_list = np.concatenate([y_true_list, labels[val_index]], 0)
            pred_list = np.concatenate([pred_list, predicted], 0)

    wandb.init(name=f'wasif_data_folds', project='wasif_data', group=f'{correction_str}: {current_dataset} {type_data}', reinit=True)
    # plot roc curve
    encoded_y_true_list = label_encoder.transform(y_true_list).toarray()
    fpr, tpr, _ = metrics.roc_curve(encoded_y_true_list.ravel(), pred_list.ravel())
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc})')
    plt.legend()
    wandb.log({'roc_curve': wandb.Image(plt)})
    plt.clf()

    # get optimal threshold
    f1_score_list = []
    auc_list = []
    for i in range(encoded_labels.shape[1]):
        y_true = label_encoder.transform(y_true_list)[:, i]
        preds = pred_list[:, i].copy()

        fpr, tpr, thresholds = metrics.roc_curve(y_true.toarray(), preds, pos_label=1)

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        preds[preds > optimal_threshold] = 1
        preds[preds <= optimal_threshold] = 0
        wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(preds=preds,
                                    y_true=y_true.toarray()[:, 0],
                                    title="confusion matrix", class_names=label_encoder.get_feature_names())})

        # f1 score
        f1_score = metrics.f1_score(y_true.toarray(), preds)
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

    # plot confusion matrix for main
    # wandb.init(name='wasif_data_main', project='wasif_data', group=f'{correction_str}: {current_dataset} {type_data}', reinit=True)
    # train_scaled_features, val_scaled_features, train_labels, val_labels, train_encoded_labels, val_encoded_labels = train_test_split(current_data, labels, encoded_labels, test_size=0.2, random_state=100)
    # np.random.seed(100)
    # random.seed(100)
    # torch.random.manual_seed(100)
    # network = LightningNetwork(features.shape[1], encoded_labels.shape[1], 2, 128)
    # wandb.watch(network, log_freq=5)
    # trainer = pl.Trainer(max_epochs=200, callbacks=[EarlyStopping(monitor="val_loss")])
    #
    # train_tensor_scaled_features = torch.from_numpy(train_scaled_features)
    # train_tensor_encoded_labels = torch.from_numpy(train_encoded_labels)
    # train_dataset = TensorDataset(train_tensor_scaled_features, train_tensor_encoded_labels)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #
    # val_tensor_scaled_features = torch.from_numpy(val_scaled_features)
    # val_tensor_encoded_labels = torch.from_numpy(val_encoded_labels)
    # val_dataset = TensorDataset(val_tensor_scaled_features, val_tensor_encoded_labels)
    # val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    #
    # trainer.fit(network, train_dataloader, val_dataloader)
    #
    # outputs = network(val_scaled_features)
    # predicted = torch.sigmoid(outputs).detach().numpy()
    #
    # # get optimal threshold
    # wandb.log({'roc_curve': wandb.plot.roc_curve(y_true=val_labels, y_probas=predicted,
    #                                                  labels=label_encoder.get_feature_names())})
    # for i in range(encoded_labels.shape[1]):
    #     y_true = val_encoded_labels[:, i]
    #     preds = predicted[:, i].copy()
    #
    #     fpr, tpr, thresholds = metrics.roc_curve(y_true, preds, pos_label=1)
    #
    #     optimal_idx = np.argmax(tpr - fpr)
    #     optimal_threshold = thresholds[optimal_idx]
    #
    #     preds[preds > optimal_threshold] = 1
    #     preds[preds <= optimal_threshold] = 0
    #     wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(preds=preds, y_true=y_true, title="confusion matrix",
    #                                                                class_names=label_encoder.get_feature_names())})
    #
    #     # f1 score
    #     f1_score = metrics.f1_score(y_true, preds)
    #     table = wandb.Table(data=[["wasif_data_main", f1_score]], columns=["runs", "f1 score"])
    #     wandb.log({f"f1_score_{label_encoder.get_feature_names()[i]}": wandb.plot.bar(table, "runs", "f1 score", "f1 score")})
    #
    #     # area under curve
    #     auc = metrics.auc(fpr, tpr)
    #     table = wandb.Table(data=[[f"wasif_data_main", auc]], columns=["runs", "auc"])
    #     wandb.log({f"auc_{label_encoder.get_feature_names()[i]}": wandb.plot.bar(table, "runs", "auc",
    #                                                                              title="auc")})
    #
    #     table = wandb.Table(data=[[f"wasif_data_main", optimal_threshold]], columns=["runs", "optimal threshold"])
    #     wandb.log(
    #         {f"optimal_threshold_{label_encoder.get_feature_names()[i]}": wandb.plot.bar(table, "runs", "optimal threshold",
    #                                                                                      title="optimal threshold")})
