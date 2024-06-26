## Importing required Libraries
import os
import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
# import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def save_numpy_data(numpy_data, numpy_labels, label):
	dir = os.path.join('tensorboard_visualization', label)
	os.makedirs(dir, exist_ok=True)

	data_filename = os.path.join(dir, 'numpy_data.npy')
	label_filename = os.path.join(dir, 'numpy_label.npy')
	np.save(data_filename, numpy_data)
	np.save(label_filename, numpy_labels)


def load_numpy_data(label):
	data_filename = os.path.join('tensorboard_visualization', label, 'numpy_data.npy')
	label_filename = os.path.join('tensorboard_visualization', label, 'numpy_label.npy')
	return np.load(data_filename, allow_pickle=True), np.load(label_filename, allow_pickle=True)


def run_visualize_tsne_tensorboard(label):
	"""

	:param numpy_labels:
	:return:
	"""
	numpy_data, numpy_labels = load_numpy_data(label)

	LOG_DIR = os.path.join('tensorboard_visualization', label)
	os.makedirs(LOG_DIR, exist_ok=True)

	with open(os.path.join(LOG_DIR, f'df_labels.tsv'), 'w') as f:
		for label in numpy_labels:
			f.write(label)
			f.write('\n')

	metadata = os.path.join(LOG_DIR, 'df_labels.tsv')

	tf_data = tf.Variable(numpy_data)
	with tf.Session() as sess:
		saver = tf.train.Saver([tf_data])
		sess.run(tf_data.initializer)
		saver.save(sess, os.path.join(LOG_DIR, f'data.ckpt'))
		config = projector.ProjectorConfig()

		embedding = config.embeddings.add()
		embedding.tensor_name = tf_data.name
		embedding.metadata_path = 'df_labels.tsv'
		# Saves a config file that TensorBoard will read during startup.
		projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)


def run_umap_visualization(label):
	if 'batch' in label :
		return
	colors = ['r', 'g', 'm', 'k', 'y']
	numpy_data, numpy_labels = load_numpy_data(label)

	reducer = TSNE(perplexity=80, n_iter=1000, learning_rate='auto')#umap.UMAP(n_neighbors=15, n_epochs=250)
	# reducer = umap.UMAP(n_neighbors=15, n_epochs=500)
	embeddings = reducer.fit_transform(numpy_data)

	unique_labels = np.unique(numpy_labels)

	for label_index, unique_label in enumerate(unique_labels):
		indexes = np.where(numpy_labels == unique_label)[0]
		plt.scatter(embeddings[indexes, 0], embeddings[indexes, 1], label=unique_label, color=colors[label_index])
	plt.legend()
	plt.title(label)
	plt.show()


if __name__ == '__main__':
	labels = os.listdir('tensorboard_visualization')
	for label in labels:
		run_umap_visualization(label)

