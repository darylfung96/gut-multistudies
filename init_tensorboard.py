import os

from visualize_tsne_tensorboard import run_visualize_tsne_tensorboard

labels = os.listdir('tensorboard_visualization')
for label in labels:
	run_visualize_tsne_tensorboard(label)