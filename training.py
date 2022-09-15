import wandb
import numpy as np
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.decomposition import PCA
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from network import LightningAutoencoderCombineBENetwork

def leave_one_dataset_out(network, data, current_data, label_encoder, encoded_labels, one_hot_decoder_indexes,
                          test_label, unique_labels, colors):
	test_index = data.index[data['Study_name'] == test_label].values
	train_index = data.index[data['Study_name'] != test_label].values

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
	outputs = network(val_features)
	# outputs = network([torch.from_numpy(val_one_hot_decoder_indexes), val_features])
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
	result_matrix = []

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
		                                                           title="confusion matrix",
		                                                           class_names=label_encoder.get_feature_names())})

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
	result_matrix.append(round(auc_average, 2))
	table = wandb.Table(data=[[f"wasif_data", auc_average]], columns=["runs", "auc"])
	wandb.log({f"auc": wandb.plot.bar(table, "runs", "auc", title="auc")})

	latent = network.get_encoded_features(torch.from_numpy(current_data)).detach().numpy()
	pca = PCA(2)
	latent_pca = pca.fit_transform(latent)
	study_group_labels = one_hot_decoder_indexes.argmax(1)
	plt.clf()
	for label in np.unique(study_group_labels):
		indexes = np.where(study_group_labels == label)[0]
		plt.xlabel('pca 1')
		plt.ylabel('pca 2')
		plt.scatter(latent_pca[indexes, 0], latent_pca[indexes, 1], label=unique_labels[label], color=colors[label])
	plt.legend()
	plt.show()
	return result_matrix


def train_one_test_all(network, data, current_data, label_encoder, encoded_labels, one_hot_decoder_indexes,
                          train_label, unique_labels, colors):
	test_index = data.index[data['Study_name'] != train_label].values
	train_index = data.index[data['Study_name'] == train_label].values

	result_matrix = []
	f1_score_table = []
	auc_table = []

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

	for current_label in unique_labels:
		current_index = data.index[data['Study_name'] == current_label].values
		current_val_features = torch.from_numpy(current_data[current_index])
		current_val_labels = encoded_labels[current_index]

		outputs = network(current_val_features)
		predicted = torch.sigmoid(outputs).detach().numpy()

		# plot roc curve
		fpr, tpr, _ = metrics.roc_curve(current_val_labels.ravel(), predicted.ravel())
		auc = metrics.auc(fpr, tpr)
		plt.plot(fpr, tpr, label=f'{train_label} {current_label} ROC curve (area = {auc})')
		plt.legend()
		wandb.log({'roc_curve': wandb.Image(plt)})
		plt.clf()

		# get optimal threshold
		f1_score_list = []
		auc_list = []
		for i in range(encoded_labels.shape[1]):
			y_true = current_val_labels[:, i]
			preds = predicted[:, i].copy()

			fpr, tpr, thresholds = metrics.roc_curve(y_true, preds, pos_label=1)

			optimal_idx = np.argmax(tpr - fpr)
			optimal_threshold = thresholds[optimal_idx]

			preds[preds > optimal_threshold] = 1
			preds[preds <= optimal_threshold] = 0
			wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(preds=preds,
			                                                           y_true=y_true,
			                                                           title="confusion matrix",
			                                                           class_names=label_encoder.get_feature_names())})

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
		f1_score_table.append([f"{current_label}", f1_score_average])

		# area under curve
		auc_average = sum(auc_list) / len(auc_list)
		auc_table.append([f"{current_label}", auc_average])

		result_matrix.append(round(auc_average, 2))

		latent = network.get_encoded_features(torch.from_numpy(current_data)).detach().numpy()
		pca = PCA(2)
		latent_pca = pca.fit_transform(latent)
		study_group_labels = one_hot_decoder_indexes.argmax(1)
		plt.clf()
		for label in np.unique(study_group_labels):
			indexes = np.where(study_group_labels == label)[0]
			plt.xlabel('pca 1')
			plt.ylabel('pca 2')
			plt.scatter(latent_pca[indexes, 0], latent_pca[indexes, 1], label=unique_labels[label], color=colors[label])
		plt.legend()
		plt.show()

	table = wandb.Table(data=f1_score_table, columns = ["runs", "f1 score"])
	wandb.log({f"f1_score_{train_label}": wandb.plot.bar(table, "runs", "f1 score", title="f1 score")})
	table = wandb.Table(data=auc_table, columns=["runs", "auc"])
	wandb.log({f"auc {train_label}": wandb.plot.bar(table, "runs", "auc", title="auc")})

	return result_matrix