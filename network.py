import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import robust_loss_pytorch

import pytorch_lightning as pl
import wandb
from robust_loss_pytorch import AdaptiveLossFunction
from ranger21 import Ranger21

from vqvae import VectorQuantizer
from losses import SupConLoss
from mmd_loss import MMD_loss


class LightningNetwork(pl.LightningModule):
    def __init__(self, input_shape, output_shape, num_layers, hidden_size, conditional_latent_size=0):
        super(LightningNetwork, self).__init__()
        self.conditional_latent_size = conditional_latent_size
        self.network = Network(input_shape, output_shape, num_layers, hidden_size, conditional_latent_size)
        self.loss = nn.BCELoss()
        # AdaptiveLossFunction()

        self.best_average_val_loss = 1e-9
        self.val_losses = []
        self.best_weights = None

    def get_last_features(self, inputs):
        if self.conditional_latent_size != 0:
            current_inputs, onehot = inputs
            if type(current_inputs) != torch.Tensor:
                current_inputs = torch.from_numpy(current_inputs)
            return self.network.get_last_features([current_inputs, onehot])

        return self.network.get_last_features(inputs)

    def forward(self, inputs):
        if self.conditional_latent_size != 0:
            current_inputs, onehot = inputs
            if type(current_inputs) != torch.Tensor:
                current_inputs = torch.from_numpy(current_inputs)
            return self.network([current_inputs, onehot])

        return self.network(inputs)

    def load_best_weights(self):
        self.load_state_dict(self.best_weights)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # optimizer = Ranger21(self.parameters(), lr=0.001, num_epochs=200, num_batches_per_epoch=5)
        return optimizer

    def on_train_start(self):
        self.train()

    def training_step(self, train_batch, batch_index):
        if self.conditional_latent_size != 0:
            x, onehot, y = train_batch
            predicted = torch.sigmoid(self.network([x, onehot]))
        else:
            x, y = train_batch
            predicted = torch.sigmoid(self.network(x))
        loss = self.loss(predicted, y)
        self.log('train_loss', loss)

        wandb.log({"train_loss": loss})
        return loss

    def on_validation_start(self):
        self.eval()
        self.train(False)

    def validation_step(self, val_batch, batch_index):
        if self.conditional_latent_size != 0:
            x, onehot, y = val_batch
            predicted = torch.sigmoid(self.network([x, onehot]))
        else:
            x, y = val_batch
            predicted = torch.sigmoid(self.network(x))
        loss = self.loss(predicted, y)
        self.log('val_loss', loss, prog_bar=True)

        self.val_losses.append(loss)

        wandb.log({"val_loss": loss})
        return loss

    def on_validation_end(self) -> None:
        average_val_loss = sum(self.val_losses) / len(self.val_losses)
        if self.best_average_val_loss < average_val_loss:
            self.best_average_val_loss = average_val_loss
            self.best_weights = self.state_dict()

        self.val_losses = []


class LightningContrastNetwork(pl.LightningModule):
    def __init__(self, input_shape, output_shape, num_layers, hidden_size):
        super(LightningContrastNetwork, self).__init__()
        first_layer = nn.Linear(input_shape, hidden_size)

        all_layers = [first_layer, nn.ReLU()]

        for i in range(num_layers):
            all_layers.append(nn.Linear(hidden_size, hidden_size))
            all_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*all_layers)
        self.second_last_output_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_shape)

        # loss
        self.loss = nn.BCELoss()
        self.contrast_loss = SupConLoss()
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 64))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def forward(self, inputs):
        output = self.layers(inputs)
        second_output = self.second_last_output_layer(output)
        last_output = self.output_layer(second_output)
        return last_output

    def on_train_start(self):
        self.train()

    def get_last_features(self, inputs):
        return self.get_features(inputs)

    def get_features(self, inputs):
        output = self.layers(inputs)
        return F.relu(self.second_last_output_layer(output))

    def training_step(self, train_batch, batch_index):
        x, y = train_batch

        output = self.layers(x)
        second_output = self.second_last_output_layer(output)
        contrast_head_output = self.head(second_output)
        last_output = self.output_layer(second_output)

        predicted = torch.sigmoid(last_output)
        contrast_loss = self.contrast_loss(F.normalize(contrast_head_output, dim=1).unsqueeze(1), y.argmax(1))
        loss = self.loss(predicted, y)
        total_loss = loss + 0.01 * contrast_loss

        self.log("train_loss", loss)
        wandb.log({"train_loss": loss})
        return total_loss

    def on_validation_start(self):
        self.eval()
        self.train(False)

    def validation_step(self, val_batch, batch_index):
        x, y = val_batch

        output = self.layers(x)
        second_output = self.second_last_output_layer(output)
        contrast_head_output = F.normalize(self.head(second_output), dim=1)
        last_output = self.output_layer(second_output)

        predicted = torch.sigmoid(last_output)
        contrast_loss = self.contrast_loss(contrast_head_output.unsqueeze(1), y.argmax(1))
        loss = self.loss(predicted, y)

        total_loss = loss + 0.01 * contrast_loss

        self.log("val_loss", loss)
        wandb.log({"val_loss": loss})
        return total_loss



class LightningMultiTaskNetwork(LightningNetwork):
    def __init__(self, input_shape, output_shape, secondary_output_shape, num_layers, hidden_size, conditional_latent_size=0):
        super(LightningMultiTaskNetwork, self).__init__(input_shape, output_shape, num_layers, hidden_size,
                                                        conditional_latent_size)
        self.second_loss = nn.NLLLoss()
        self.best_loss = 1e9
        self.best_weights = None
        self.num_layers = num_layers

        # network definition #
        self.first_layer = nn.Sequential(nn.Linear(input_shape, hidden_size), nn.ReLU())
        self.all_layers = []
        self.secondary_layers = []

        self.train_secondary_only = True

        for i in range(num_layers):
            self.all_layers.append(nn.Linear(hidden_size, hidden_size))
            self.all_layers.append(nn.ReLU())

            self.secondary_layers.append(nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, secondary_output_shape)
            ))
            # all_layers.append(nn.Dropout(0.5))

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_shape)
        )

        self.optim = torch.optim.Adam(self.parameters())

        # loss
        self.loss = nn.BCELoss()
        self.recons_loss = nn.MSELoss()

        self.reconstruction_task_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_shape)
        )

        self.second_task_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, secondary_output_shape)
        )

    def set_train_secondary_only(self, value):
        self.train_secondary_only = value

    def get_intermediate_layers(self, inputs):
        all_outputs = []
        current_outputs = self.first_layer(inputs)
        for i in range(self.num_layers):
            current_output = self.all_layers[i](current_outputs)
            all_outputs.append(current_output)
        return all_outputs

    def forward(self, inputs):
        current_inputs = self.first_layer(inputs)
        for i in range(self.num_layers):
            current_inputs = self.all_layers[i](current_inputs)
        output = self.output_layer(current_inputs)

        second_task_output = self.second_task_layer(current_inputs)
        reconstruction_output = self.reconstruction_task_layer(current_inputs)

        return output, second_task_output, reconstruction_output

    def set_best_weights(self, best_weights):
        self.best_weights = best_weights

    def load_best_weights(self):
        self.load_state_dict(self.best_weights)

    def training_step(self, train_batch, batch_index):
        x, y, secondary_y = train_batch

        output, second_task_output, reconstruction_output = self(x)
        predicted = torch.sigmoid(output)
        secondary_prediction = torch.log_softmax(second_task_output, 1)

        intermediate_outputs = self.get_intermediate_layers(x)
        intermediate_losses = 0
        for i in range(self.num_layers):
            secondary_output = torch.log_softmax(self.secondary_layers[i](intermediate_outputs[i]), 1)
            intermediate_loss = self.second_loss(secondary_output, secondary_y.argmax(1))
            intermediate_losses += intermediate_loss

        if not self.train_secondary_only:
            loss = self.loss(predicted, y)
        else:
            loss = 0

        second_loss = self.second_loss(secondary_prediction, secondary_y.argmax(1))
        recons_loss = self.recons_loss(torch.sigmoid(reconstruction_output), x)
        # total_loss = loss + 0.25 * second_loss + 0.25 * intermediate_losses
        total_loss = loss + 0.8 * second_loss + 0.1 * intermediate_losses

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_secondary_loss', second_loss)

        wandb.log({"train_loss": loss, 'train_secondary_loss': second_loss})
        return total_loss

    def validation_step(self, val_batch, batch_index):
        x, y, secondary_y = val_batch

        output, second_task_output, reconstruction_output = self(x)
        predicted = torch.sigmoid(output)
        secondary_prediction = torch.log_softmax(second_task_output, 1)

        intermediate_outputs = self.get_intermediate_layers(x)
        intermediate_losses = 0
        for i in range(self.num_layers):
            secondary_output = torch.log_softmax(self.secondary_layers[i](intermediate_outputs[i]), 1)
            intermediate_loss = self.second_loss(secondary_output, secondary_y.argmax(1))
            intermediate_losses += intermediate_loss

        if not self.train_secondary_only:
            loss = self.loss(predicted, y)
        else:
            loss = self.second_loss(secondary_prediction, secondary_y.argmax(1))

        second_loss = self.second_loss(secondary_prediction, secondary_y.argmax(1))
        total_loss = loss + second_loss + intermediate_losses
        # recons_loss = self.recons_loss(torch.sigmoid(reconstruction_output), x)
        # total_loss = loss + second_loss + recons_loss

        if loss < self.best_loss:
            self.best_weights = self.state_dict()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_secondary_loss', second_loss)

        wandb.log({"val_loss": loss, 'val_secondary_loss': second_loss})
        return total_loss


class LightningAutoencoderNetwork(pl.LightningModule):

    def __init__(self, input_shape, output_shape, condition_latent_size, num_layers, hidden_size, network_type='normal'):
        # network_type can be 'normal' or 'conditional'
        super(LightningAutoencoderNetwork, self).__init__()

        networks_dict = {
            'conditional': lambda: ConditionalNetwork(input_shape, condition_latent_size, output_shape, hidden_size, network_type),
            'normal': lambda: AutoencoderNetwork(input_shape, output_shape, hidden_size, network_type),
            'vqvae': lambda: ConditionalNetwork(input_shape, condition_latent_size, output_shape, hidden_size, network_type)
        }

        self.network_type = network_type
        self.network = networks_dict[network_type]()
        self.loss = nn.MSELoss()
        self.second_loss = nn.NLLLoss()
        self.contrastive_loss = SupConLoss()

        # for contrastive
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, condition_latent_size))
        self.encoded_head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 64))

    def get_encoded_features(self, inputs):
        if type(inputs) != torch.Tensor:
            inputs = torch.from_numpy(inputs)
        return self.network.get_encoded_features(inputs)

    def decode(self, encoded_features):
        return self.network.decode(encoded_features)

    def forward(self, inputs):
        if type(inputs) != torch.Tensor:
            inputs = torch.from_numpy(inputs)
        if self.type == 'vqvae':
            loss, encoded_features, persplexity, _ = self.network(inputs)
            return encoded_features
        else:
            encoded_features = self.network(inputs)
            return encoded_features

    def load_best_weights(self):
        self.load_state_dict(self.best_weights)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # optimizer = Ranger21(self.parameters(), lr=0.001, num_epochs=200, num_batches_per_epoch=5)
        return optimizer

    def on_train_start(self) -> None:
        self.train(True)

    def training_step(self, train_batch, batch_index):
        x, onehot_decoder_indexes, y = train_batch

        total_loss = 0
        if self.network_type == 'conditional' or self.network_type == 'vqvae':
            # mean, log_var, encoded_output, reconstructed_output = self.network([onehot_decoder_indexes, x])
            encoded_output, reconstructed_output = self.network([onehot_decoder_indexes, x])
            if self.type == 'vqvae':
                loss, encode_features, persplexity, _ = encoded_output
                total_loss = total_loss + loss
        else:
            encoded_output, reconstructed_output = self.network(x)
            if self.type == 'vqvae':
                loss, encode_features, persplexity, _ = encoded_output
                total_loss = total_loss + loss
            # mean, log_var, encoded_output, reconstructed_output = self.network(x)

        reconstruction_loss = self.loss(reconstructed_output, x)
        total_loss = total_loss + reconstruction_loss
        # kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # for contrastive loss
        # contrast_loss = self.contrastive_loss(F.normalize(self.head(x), dim=1).unsqueeze(1), onehot_decoder_indexes.argmax(1))
        # encoded_contrast_loss = self.contrastive_loss(F.normalize(self.encoded_head(encoded_output),
        #                                                           dim=1).unsqueeze(1), onehot_decoder_indexes.argmax(1))
        # second_loss = self.second_loss(torch.log_softmax(self.head(encoded_output), 1), onehot_decoder_indexes.argmax(1))

        # loss = reconstruction_loss #+ encoded_contrast_loss #+ contrast_loss + encoded_contrast_loss  # + kld_loss
        # output_loss = self.classify_loss(output, y)

        self.log('train_autoencoder_loss', total_loss)
        # self.log('train_contrast_loss', contrast_loss)
        wandb.log({"train_autoencoder_loss": total_loss})
        return total_loss

    def on_validation_start(self):
        self.eval()
        self.train(False)

    def validation_step(self, val_batch, batch_index):
        x, onehot_decoder_indexes, y = val_batch
        mean, log_var, encoded_output, reconstructed_output = self.network([onehot_decoder_indexes, x])

        reconstruction_loss = self.loss(reconstructed_output, x)
        # contrast_loss = self.contrastive_loss(F.normalize(self.head(x)).unsqueeze(1), onehot_decoder_indexes.argmax(1))
        # kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = reconstruction_loss# + kld_loss
        self.log('val_autoencoder_loss', loss)
        # self.log('train_contrast_loss', contrast_loss)
        wandb.log({"train_autoencoder_loss": loss})
        return loss


class LightningAutoencoderCombineNetwork(pl.LightningModule):

    def __init__(self, input_shape, output_shape, condition_latent_size, num_layers, latent_size, hidden_size, network_type='normal'):
        # network_type can be 'normal' or 'conditional'
        super(LightningAutoencoderCombineNetwork, self).__init__()

        networks_dict = {
            'conditional': lambda: ConditionalNetwork(input_shape, condition_latent_size, output_shape, latent_size, network_type),
            'normal': lambda: AutoencoderNetwork(input_shape, output_shape, latent_size, network_type),
            'vqvae': lambda: ConditionalNetwork(input_shape, condition_latent_size, output_shape, latent_size, network_type),
            'vae': lambda: VariationalAutoencoderNetwork(input_shape, output_shape, latent_size, network_type)
        }

        self.latent_size = latent_size
        self.condition_latent_size = condition_latent_size
        self.network_type = network_type
        self.autoencoder_network = networks_dict[network_type]()
        self.loss = nn.MSELoss()
        self.second_loss = nn.NLLLoss()
        self.contrastive_loss = SupConLoss()
        self.autoencoder_dataset = None

        # for classification
        self.network = Network(condition_latent_size + latent_size, output_shape, num_layers, hidden_size)
        self.binary_loss = nn.BCELoss()

        # for contrastive
        self.head = nn.Sequential(nn.Linear(latent_size, 64), nn.ReLU(), nn.Linear(64, condition_latent_size))
        self.encoded_head = nn.Sequential(nn.Linear(latent_size, 64), nn.ReLU(), nn.Linear(64, 64))

        # best weights
        self.best_average_val_loss = 1e-9
        self.val_losses = []
        self.best_weights = None

        # generate scheduler for autoencoder and classification weights
        all_epochs = np.arange(1, 100)
        self.autoencoder_alpha = list(reversed([self._exp_decay(epoch) for epoch in all_epochs]))
        self.autoencoder_alpha = [1 - item for item in self.autoencoder_alpha]

    def _get_autoencoder_alpha(self):
        if self.current_epoch >= 99:
            return 0

        return self.autoencoder_alpha[self.current_epoch]

    def _exp_decay(self, epoch):
        if epoch >= 100:
            return 0
        initial_lrate = 1
        k = 0.1
        lrate = initial_lrate * np.exp(-k * epoch)
        return lrate

    def get_encoded_features(self, inputs):
        if type(inputs) != torch.Tensor:
            inputs = torch.from_numpy(inputs)
        return self.autoencoder_network.get_encoded_features(inputs)

    def set_autoencoder_dataset(self, dataset):
        self.autoencoder_dataset = dataset

    def decode(self, encoded_features):
        return self.autoencoder_network.decode(encoded_features)

    def forward(self, inputs):
        if self.type == 'vqvae':
            loss, encoded_features, persplexity, _ = self.autoencoder_network(inputs)
            return encoded_features
        else:
            encoded_features, reconstruction_output = self.autoencoder_network(inputs)
            outputs = self.network(encoded_features)
            return outputs

    def load_best_weights(self):
        self.load_state_dict(self.best_weights)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # optimizer = Ranger21(self.parameters(), lr=0.001, num_epochs=200, num_batches_per_epoch=5)
        return optimizer

    def on_train_start(self) -> None:
        self.train(True)

    def training_step(self, train_batch, batch_index):
        x, onehot_decoder_indexes, y = train_batch

        total_loss = 0
        if self.network_type == 'conditional' or self.network_type == 'vqvae':
            # mean, log_var, encoded_output, reconstructed_output = self.network([onehot_decoder_indexes, x])
            autoencoder_x, autoencoder_onehot_decoder_indexes, autoencoder_y = next(self.autoencoder_dataset)
            _, reconstructed_output = self.autoencoder_network([autoencoder_onehot_decoder_indexes, autoencoder_x])
            encoded_output, _ = self.autoencoder_network([onehot_decoder_indexes, x])
            outputs = torch.sigmoid(self.network(encoded_output))

            classification_loss = (1-self._get_autoencoder_alpha()) * self.binary_loss(outputs, y)
            total_loss = total_loss + classification_loss

            if self.type == 'vqvae':
                loss, encode_features, persplexity, _ = encoded_output
                total_loss = total_loss + loss
        else:
            autoencoder_x, autoencoder_onehot_decoder_indexes, autoencoder_y = next(self.autoencoder_dataset)
            _, reconstructed_output = self.autoencoder_network([autoencoder_onehot_decoder_indexes, autoencoder_x])
            encoded_output, reconstructed_output = self.autoencoder_network(x)
            if self.type == 'vqvae':
                loss, encode_features, persplexity, _ = encoded_output
                total_loss = total_loss + loss
            # mean, log_var, encoded_output, reconstructed_output = self.autoencoder_network(x)

        reconstruction_loss = self.loss(reconstructed_output, autoencoder_x)
        total_loss = total_loss + self._get_autoencoder_alpha() * reconstruction_loss
        # kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        # for contrastive loss
        # contrast_loss = self.contrastive_loss(F.normalize(self.head(x), dim=1).unsqueeze(1), onehot_decoder_indexes.argmax(1))
        # encoded_contrast_loss = self.contrastive_loss(F.normalize(self.encoded_head(encoded_output),
        #                                                           dim=1).unsqueeze(1), onehot_decoder_indexes.argmax(1))
        # second_loss = self.second_loss(torch.log_softmax(self.head(encoded_output), 1), onehot_decoder_indexes.argmax(1))

        # loss = reconstruction_loss #+ encoded_contrast_loss #+ contrast_loss + encoded_contrast_loss  # + kld_loss
        # output_loss = self.classify_loss(output, y)

        self.log('train_loss', total_loss)
        # self.log('train_contrast_loss', contrast_loss)
        wandb.log({"train_loss": total_loss})
        return total_loss

    def on_validation_start(self):
        self.eval()
        self.train(False)

    def validation_step(self, val_batch, batch_index):
        x, onehot_decoder_indexes, y = val_batch
        encoded_output, _ = self.autoencoder_network([onehot_decoder_indexes, x])
        outputs = torch.sigmoid(self.network(encoded_output))

        classification_loss = self.binary_loss(outputs, y)
        # contrast_loss = self.contrastive_loss(F.normalize(self.head(x)).unsqueeze(1), onehot_decoder_indexes.argmax(1))
        # kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = classification_loss # + kld_loss
        self.val_losses.append(loss)

        self.log('val_loss', loss)
        # self.log('train_contrast_loss', contrast_loss)
        wandb.log({"val_loss": loss})
        return loss

    def on_validation_end(self) -> None:
        average_val_loss = sum(self.val_losses) / len(self.val_losses)
        if self.best_average_val_loss < average_val_loss:
            self.best_average_val_loss = average_val_loss
            self.best_weights = self.state_dict()

        self.val_losses = []


class LightningAutoencoderCombineBENetwork(LightningAutoencoderCombineNetwork):

    def __init__(self, input_shape, output_shape, condition_latent_size, num_layers, latent_size, hidden_size,
                 network_type='normal'):

        super(LightningAutoencoderCombineBENetwork, self).__init__(input_shape, output_shape,
                                                                   condition_latent_size, num_layers,
                                                                   latent_size, hidden_size, network_type)

        self.network = Network(latent_size, output_shape, num_layers, hidden_size)
        self.batch_network = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2),
            nn.ReLU(),
            nn.Linear(latent_size * 2, latent_size * 2),
            nn.ReLU(),
            nn.Linear(latent_size * 2, condition_latent_size)
        )
        self.robust_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=2, float_dtype=np.float32,
                                                                             device='cpu')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.parameters()) + list(self.robust_loss.parameters()), lr=0.001)
        return optimizer

    def _classification_loss(self):
        ...

    def _reconstruction_loss(self):
        autoencoder_x, autoencoder_onehot_decoder_indexes, autoencoder_y = next(self.autoencoder_dataset)
        _, reconstructed_output = self.autoencoder_network(autoencoder_x)
        reconstruction_loss = self.loss(reconstructed_output, autoencoder_x)
        return reconstruction_loss

    def _classification_loss(self, x, y):
        encoded_output, _ = self.autoencoder_network(x)
        outputs = torch.sigmoid(self.network(encoded_output))
        classification_loss = torch.mean(self.robust_loss.lossfun(outputs - y))
        return encoded_output, classification_loss

    def training_step(self, train_batch, batch_index):
        x, onehot_decoder_indexes, y = train_batch

        total_loss = 0
        # reconstruction
        reconstruction_loss = self._reconstruction_loss()
        total_loss = total_loss + self._get_autoencoder_alpha() * reconstruction_loss

        # classification
        encoded_output, classification_loss = self._classification_loss(x, y)

        # batch effect correction
        onehot_output = torch.log_softmax(self.batch_network(encoded_output), 1)
        batch_loss = torch.clamp(-self.second_loss(onehot_output, onehot_decoder_indexes.argmax(1)), -1)

        total_loss = total_loss + (1 - self._get_autoencoder_alpha()) * classification_loss + 0.01 * batch_loss

        if self.type == 'vqvae':
            loss, encode_features, persplexity, _ = encoded_output
            total_loss = total_loss + loss

        self.log('train_loss', total_loss)
        self.log('train_reconstruction_loss', reconstruction_loss)
        self.log('train_classification_loss', classification_loss)
        wandb.log({"train_loss": total_loss,
                   'train_reconstruction_loss': reconstruction_loss,
                   'train_classification_loss': classification_loss,
                   'alpha': self.robust_loss.alpha()[0,0].data,
                   'scale': self.robust_loss.scale()[0,0].data})
        return total_loss

    def validation_step(self, val_batch, batch_index):
        x, onehot_decoder_indexes, y = val_batch
        encoded_output, _ = self.autoencoder_network(x)
        outputs = torch.sigmoid(self.network(encoded_output))

        classification_loss = self.binary_loss(outputs, y)
        loss = classification_loss # + kld_loss
        self.val_losses.append(loss)

        self.log('val_loss', loss)
        # self.log('train_contrast_loss', contrast_loss)
        wandb.log({"val_loss": loss})
        return loss


class LightningAutoencoderCombineBEMMDNetwork(LightningAutoencoderCombineBENetwork):
    def __init__(self, input_shape, output_shape, condition_latent_size, num_layers, latent_size, hidden_size,
                 network_type='normal'):
        super(LightningAutoencoderCombineBEMMDNetwork, self).__init__(input_shape,
                                                                      output_shape, condition_latent_size,
                                                                      num_layers, latent_size, hidden_size,
                 network_type=network_type)
        self.mmd_loss = MMD_loss()

    def training_step(self, train_batch, batch_index):
        x, onehot_decoder_indexes, y = train_batch

        total_loss = 0
        # reconstruction
        reconstruction_loss = self._reconstruction_loss()
        total_loss = total_loss + self._get_autoencoder_alpha() * reconstruction_loss

        # classification
        encoded_output, classification_loss = self._classification_loss(x, y)

        # batch effect correction
        onehot_output = torch.log_softmax(self.batch_network(encoded_output), 1)
        batch_loss = self.second_loss(onehot_output, onehot_decoder_indexes.argmax(1))
        total_loss = total_loss + (1 - self._get_autoencoder_alpha()) * classification_loss + 0.01 * batch_loss

        encoded_output, _ = self.autoencoder_network(x)

        mmd_loss = self.mmd_loss(encoded_output, encoded_output)
        total_loss = total_loss + 0.01 * mmd_loss

        self.log('train_loss', total_loss)
        self.log('train_reconstruction_loss', reconstruction_loss)
        self.log('train_classification_loss', classification_loss)
        self.log('train_mmd_loss', mmd_loss)
        wandb.log({"train_loss": total_loss,
                   'train_reconstruction_loss': reconstruction_loss,
                   'train_classification_loss': classification_loss,
                   'train_mmd_loss': mmd_loss,
                   'alpha': self.robust_loss.alpha()[0, 0].data,
                   'scale': self.robust_loss.scale()[0, 0].data})
        return total_loss


class LightningVAutoencoderCombineBENetwork(LightningAutoencoderCombineNetwork):

    def __init__(self, input_shape, output_shape, condition_latent_size, num_layers, latent_size, hidden_size,
                 network_type='vae'):

        super(LightningVAutoencoderCombineBENetwork, self).__init__(input_shape, output_shape,
                                                                   condition_latent_size, num_layers,
                                                                   latent_size, hidden_size, network_type)

        self.network = Network(latent_size, output_shape, num_layers, hidden_size)
        self.batch_network = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2),
            nn.ReLU(),
            nn.Linear(latent_size * 2, latent_size * 2),
            nn.ReLU(),
            nn.Linear(latent_size * 2, condition_latent_size)
        )

        self.loss = nn.L1Loss()
        self.robust_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=2, float_dtype=np.float32,
                                                                             device='cpu')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.parameters()) + list(self.robust_loss.parameters()), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_index):
        x, onehot_decoder_indexes, y = train_batch

        total_loss = 0
        # reconstruction
        autoencoder_x, autoencoder_onehot_decoder_indexes, autoencoder_y = next(self.autoencoder_dataset)
        _, reconstructed_output = self.autoencoder_network(autoencoder_x)
        reconstruction_loss = self.loss(reconstructed_output, autoencoder_x)
        kld = 4 * self.autoencoder_network.kld_loss(autoencoder_x)
        total_loss = total_loss + self._get_autoencoder_alpha() * (reconstruction_loss + kld)

        # classification
        encoded_output, _ = self.autoencoder_network(x)
        outputs = torch.sigmoid(self.network(encoded_output))
        classification_loss = (1 - self._get_autoencoder_alpha()) * torch.mean(self.robust_loss.lossfun(outputs-y))

        # batch effect correction
        onehot_output = torch.log_softmax(self.batch_network(encoded_output), 1)
        batch_loss = torch.clamp(-self.second_loss(onehot_output, onehot_decoder_indexes.argmax(1)), -1)

        total_loss = total_loss + classification_loss + 0.01 * batch_loss

        if self.type == 'vqvae':
            loss, encode_features, persplexity, _ = encoded_output
            total_loss = total_loss + loss

        self.log('train_loss', total_loss)
        self.log('train_kld_loss', kld)
        self.log('train_reconstruction_loss', reconstruction_loss)
        self.log('train_classification_loss', classification_loss)
        self.log('alpha', self.robust_loss.alpha()[0,0].data, prog_bar=True)
        self.log('scale', self.robust_loss.scale()[0,0].data, prog_bar=True)
        wandb.log({"train_loss": total_loss, 'train_kld_loss': kld,
                   'train_reconstruction_loss': reconstruction_loss, 'train_classification_loss': classification_loss})
        return total_loss

    def validation_step(self, val_batch, batch_index):
        x, onehot_decoder_indexes, y = val_batch
        encoded_output, _ = self.autoencoder_network(x)
        outputs = torch.sigmoid(self.network(encoded_output))

        classification_loss = self.binary_loss(outputs, y)
        loss = classification_loss # + kld_loss
        self.val_losses.append(loss)

        self.log('val_loss', loss)
        # self.log('train_contrast_loss', contrast_loss)
        wandb.log({"val_loss": loss})
        return loss


class LightningJointAutoencoderNetwork(LightningNetwork):

    def __init__(self, input_shape, output_shape, condition_latent_size, num_layers, hidden_size, network_type='normal'):
        # network_type can be 'normal' or 'conditional'
        super(LightningJointAutoencoderNetwork, self).__init__(input_shape, output_shape, num_layers, hidden_size)

        networks_dict = {
            'conditional': lambda: ConditionalNetwork(input_shape, condition_latent_size, output_shape, hidden_size),
            'normal': lambda: AutoencoderNetwork(input_shape, output_shape, hidden_size)
        }

        self.network_type = network_type
        self.network = networks_dict[network_type]()
        self.classify_network = Network(hidden_size+condition_latent_size, output_shape, num_layers, hidden_size)
        self.loss = nn.L1Loss()
        self.classify_loss = nn.BCELoss()
        self.train_recons = False

    def get_encoded_features(self, inputs):
        if type(inputs) != torch.Tensor:
            inputs = torch.from_numpy(inputs)
        return self.network.get_encoded_features(inputs)

    def forward(self, inputs):
        if self.network_type == 'conditional':
            onehot_decoder_indexes, current_inputs = inputs
            if type(current_inputs) != torch.Tensor:
                current_inputs = torch.from_numpy(current_inputs)
            if type(onehot_decoder_indexes) != torch.Tensor:
                onehot_decoder_indexes = torch.from_numpy(onehot_decoder_indexes)
            mean, log_var, encoded_output, reconstructed_output = self.network([onehot_decoder_indexes, current_inputs])
            output = self.classify_network(encoded_output)
            return output
        else:
            return self.network(inputs)

    def training_step(self, train_batch, batch_index):
        x, onehot_decoder_indexes, y = train_batch

        if self.global_step % 50 == 0:
            self.train_recons = not self.train_recons

        if self.network_type == 'conditional':
            mean, log_var, encoded_output, reconstructed_output = self.network([onehot_decoder_indexes, x])
        else:
            mean, log_var, encoded_output, reconstructed_output = self.network(x)
        if self.train_recons:
            reconstruction_loss = self.loss(reconstructed_output, x)
            kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss = reconstruction_loss + kld_loss
            self.log('train_autoencoder_loss', loss)
            wandb.log({"train_autoencoder_loss": loss})
        else:
            output = torch.sigmoid(self.classify_network(encoded_output))
            output_loss = self.classify_loss(output, y)
            self.log('train_output_loss', output_loss)
            wandb.log({"train_output_loss": output_loss})
            loss = output_loss

        return loss

    def validation_step(self, val_batch, batch_index):
        x, onehot_decoder_indexes, y = val_batch
        mean, log_var, encoded_output, reconstructed_output = self.network([onehot_decoder_indexes, x])

        reconstruction_loss = self.loss(reconstructed_output, x)
        kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = reconstruction_loss + kld_loss

        output = torch.sigmoid(self.classify_network(encoded_output))
        output_loss = self.classify_loss(output, y)
        loss = loss + output_loss

        self.log('val_autoencoder_loss', reconstruction_loss + kld_loss)
        self.log('val_output_loss', output_loss)
        self.log('val_total_loss', loss)

        wandb.log({"val_autoencoder_loss": loss, 'val_output_loss': output_loss, 'val_total_loss': loss})
        return loss


class LightningMultiAutoencoderNetwork(LightningAutoencoderNetwork):

    def __init__(self, input_shape, output_shape, num_layers, hidden_size, latent_size=16, multiple_type="encoder"):
        super(LightningMultiAutoencoderNetwork, self).__init__(input_shape, output_shape, None, num_layers, hidden_size)
        self.multiple_type = multiple_type
        self.network = MultiAutoencoderNetwork(input_shape, hidden_size, latent_size, multiple_type=multiple_type)
        self.loss = nn.MSELoss()
        self.decoder_index = 0

    def set_decoder_index(self, decoder_index):
        self.decoder_index = decoder_index

    def forward(self, inputs):
        current_input, decoder_index = inputs
        if type(current_input) != torch.Tensor:
            current_input = torch.from_numpy(current_input)
        mean, log_var, encoded_output, reconstructed_output = self.network(current_input, self.decoder_index)
        return reconstructed_output

    def get_encoded_features(self, inputs):
        current_inputs, encoder_indexes = inputs
        latents = []
        if self.multiple_type == 'encoder':
            for i in range(current_inputs.shape[0]):
                mean, log_var, latent = self.network.get_encoded_features(current_inputs[i], encoder_indexes[i])
                latents.append(latent)
            latents = torch.stack(latents)
        else:
            mean, log_var, latent = self.network.get_encoded_features(inputs, None)
            latents = latent

        return latents

    def training_step(self, train_batch, batch_index):
        x, y = train_batch
        mean, log_var, encoded_output, reconstructed_output = self.network(x, self.decoder_index)

        reconstruction_loss = self.loss(reconstructed_output, x)
        kld_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        loss = reconstruction_loss + kld_loss

        self.log('train_autoencoder_loss', loss)

        wandb.log({"train_autoencoder_loss": loss})
        return loss

    def validation_step(self, val_batch, batch_index):
        x, y = val_batch
        encoded_output, reconstructed_output = self.network(x, self.decoder_index)
        loss = self.loss(reconstructed_output, x)
        self.log('val_autoencoder_loss', loss)

        wandb.log({"val_autoencoder_loss": loss})
        return loss


class LightningJointMultiAutoencoderNetwork(LightningMultiAutoencoderNetwork):

    def __init__(self, input_shape, output_shape, num_layers, latent_size, hidden_size):
        super(LightningJointMultiAutoencoderNetwork, self).__init__(input_shape, input_shape, num_layers, hidden_size)
        self.network = MultiAutoencoderNetwork(input_shape, latent_size)
        self.classify_network = Network(latent_size, output_shape, num_layers, hidden_size)
        self.loss = nn.MSELoss()
        self.classify_loss = nn.BCELoss()
        self.decoder_index = 0
        self.train_recons = True

    def set_decoder_index(self, decoder_index):
        self.decoder_index = decoder_index

    def forward(self, inputs):
        current_inputs, decoder_index = inputs
        if type(current_inputs) != torch.Tensor:
            current_inputs = torch.from_numpy(current_inputs)

        reconstructed_outputs = None
        outputs = None

        if current_inputs.shape[0] == 1:
            mean, log_var, encoded_output, reconstructed_output = self.network(current_inputs, self.decoder_index)
            output = self.classify_network(encoded_output)
            reconstructed_outputs = reconstructed_output
            outputs = output
        else:
            for i in range(current_inputs.shape[0]):
                current_input = current_inputs[i].unsqueeze(0)
                current_decoder_index = decoder_index[i].item()
                mean, log_var, encoded_output, reconstructed_output = self.network(current_input, current_decoder_index)
                output = self.classify_network(encoded_output)

                if reconstructed_outputs is None:
                    reconstructed_outputs = reconstructed_output
                    outputs = output
                else:
                    reconstructed_outputs = torch.cat([reconstructed_outputs, reconstructed_output])
                    outputs = torch.cat([outputs, output])

        return reconstructed_outputs, outputs

    def get_encoded_features(self, inputs):
        mean, log_var, latent = self.network.get_encoded_features(inputs)
        return latent

    def training_step(self, train_batch, batch_index):
        if self.global_step % 50 == 0:
            self.train_recons = not self.train_recons

        x, decoder_index, y = train_batch
        # make sure batch size is one because this only work with one batch size since we have many decoders
        assert x.shape[0] == 1

        mean, log_var, encoded_output, reconstructed_output = self.network(x, decoder_index.item())
        output = torch.sigmoid(self.classify_network(encoded_output))
        if self.train_recons:
            reconstruction_loss = self.loss(reconstructed_output, x)
            kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss = reconstruction_loss + kld_loss
            self.log('train_autoencoder_loss', reconstruction_loss + kld_loss)
            self.log('train_loss', loss)
            wandb.log({"train_autoencoder_loss": reconstruction_loss + kld_loss,
                       "train_loss": loss})

        if not self.train_recons:
            output_loss = self.classify_loss(output, y)
            loss = output_loss
            self.log('train_output_loss', output_loss)
            wandb.log({"train_output_loss": output_loss,
                       "train_loss": loss})

        return loss

    def validation_step(self, val_batch, batch_index):
        x, decoder_index, y = val_batch
        # make sure batch size is one because this only work with one batch size since we have many decoders
        assert x.shape[0] == 1
        mean, log_var, encoded_output, reconstructed_output = self.network(x, decoder_index.item())
        output = torch.sigmoid(self.classify_network(encoded_output))
        reconstruction_loss = self.loss(reconstructed_output, x)
        kld_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        output_loss = self.classify_loss(output, y)

        loss = reconstruction_loss + kld_loss + output_loss

        self.log('val_autoencoder_loss', reconstruction_loss + kld_loss)
        self.log('val_output_loss', output_loss)
        self.log('val_loss', loss)

        wandb.log({"val_autoencoder_loss": reconstruction_loss + kld_loss,
                   "val_output_loss": output_loss,
                   "val_loss": loss})

        return loss


class MultiAutoencoderNetwork(nn.Module):
    def __init__(self, input_shape, hidden_size, latent_size=16, num_decoders=5, multiple_type="encoder"):
        super(MultiAutoencoderNetwork, self).__init__()

        self.multiple_type = multiple_type

        # have multiple encoders for different studies
        if multiple_type == 'encoder':
            self.encoders = []
            self.encoder_mean_layers = []
            self.encoder_var_layers = []
            encoders_param = None
            for _ in range(num_decoders):
                encoder = [nn.Linear(input_shape, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                            nn.Linear(hidden_size, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                           nn.Linear(2 * hidden_size, hidden_size)]
                encoder = nn.Sequential(*encoder)
                self.encoders.append(encoder)
                mean_layer = nn.Linear(hidden_size, latent_size)
                log_var_layer = nn.Linear(hidden_size, latent_size)
                self.encoder_mean_layers.append(mean_layer)
                self.encoder_var_layers.append(log_var_layer)
                if not encoders_param:
                    encoders_param = list(encoder.parameters())
                else:
                    encoders_param += list(encoder.parameters())

            decoder = [nn.Linear(latent_size, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                       nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                       nn.Linear(hidden_size, input_shape)]
            self.decoder = nn.Sequential(*decoder)
            self.optim = torch.optim.Adam(self.parameters())#torch.optim.Adam(list(self.decoder.parameters()) + encoders_param)
        else:
            # multiple decoders
            encoder = [nn.Linear(input_shape, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                       nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5)]
            self.encoder = nn.Sequential(*encoder)
            self.mean_layer = nn.Linear(hidden_size, latent_size)
            self.log_var_layer = nn.Linear(hidden_size, latent_size)

            self.decoders = []
            decoders_param = None
            for _ in range(num_decoders):
                decoder = [nn.Linear(latent_size, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                           nn.Linear(2 * hidden_size, input_shape)]
                decoder = nn.Sequential(*decoder)
                self.decoders.append(decoder)
                if not decoders_param:
                    decoders_param = list(decoder.parameters())
                else:
                    decoders_param += list(decoder.parameters())
            self.optim = torch.optim.Adam(self.parameters())#torch.optim.Adam(list(self.encoder.parameters()) + decoders_param)

        # loss
        self.loss = nn.BCELoss()

    def reparameterization(self, mean, log_var):
        epsilon = torch.randn_like(log_var)
        z = mean + log_var * epsilon
        return z

    def get_encoded_features(self, inputs, encoder_index):
        if self.multiple_type == 'encoder':
            encoded = self.encoders[encoder_index](inputs)
            mean_layer = self.encoder_mean_layers[encoder_index]
            var_layer = self.encoder_var_layers[encoder_index]
        else:
            encoded = self.encoder(inputs)
            mean_layer = self.mean_layer
            var_layer = self.log_var_layer

        mean, log_var = mean_layer(encoded), var_layer(encoded)
        return mean, log_var, self.reparameterization(mean, log_var)

    def forward(self, inputs, decoder_index):
        """

        :param inputs:
        :param decoder_index:   Which decoder to use
        :return:
        """
        mean, log_var, encoded_output = self.get_encoded_features(inputs, decoder_index)

        # multiple encoder
        if self.multiple_type == 'encoder':
            reconstructed_output = self.decoder(encoded_output)
        else:
            # multiple decoder
            reconstructed_output = self.decoders[decoder_index](encoded_output)
        return mean, log_var, encoded_output, reconstructed_output


class AutoencoderNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size, network_type='normal'):
        super(AutoencoderNetwork, self).__init__()
        self.network_type = network_type
        self.hidden_size = hidden_size

        self._initialize_network(input_shape, hidden_size)

    def _initialize_network(self, input_shape, hidden_size):
        encoder = [nn.Linear(input_shape, 2 * hidden_size), nn.ReLU(),
                   nn.Linear(2 * hidden_size, hidden_size), nn.ReLU()]

        if self.network_type == 'vqvae':
            encoder.append(VectorQuantizer(64, hidden_size, 0.25))
        self.encoder = nn.Sequential(*encoder)

        decoder = [nn.Linear(hidden_size, 2 * hidden_size), nn.ReLU(),
                   nn.Linear(2 * hidden_size, input_shape)]
        self.decoder = nn.Sequential(*decoder)
        # loss
        self.loss = nn.BCELoss()

    def get_encoded_features(self, inputs):
        if self.network_type == 'vqvae':
            loss, encoded, persplexity, _ = self.encoder(inputs)
            return loss, encoded, persplexity
        else:
            encoded = self.encoder(inputs)
            return encoded

    def decode(self, encoded_features):
        return self.decoder(encoded_features)

    def forward(self, inputs):
        encoded_output = self.get_encoded_features(inputs)
        if self.type == 'vqvae':
            _, encoded_features, _ = encoded_output
            reconstructed_output = self.decoder(encoded_features)
        else:
            reconstructed_output = self.decoder(encoded_output)

        return encoded_output, reconstructed_output


class VariationalAutoencoderNetwork(AutoencoderNetwork):
    
    def __init__(self, input_shape, output_shape, hidden_size, network_type='normal'):
        super(VariationalAutoencoderNetwork, self).__init__(input_shape, output_shape, hidden_size,
                                                            network_type='normal')

    def _initialize_network(self, input_shape, hidden_size):
        encoder = [nn.Linear(input_shape, 2 * hidden_size), nn.ReLU(),
                   nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(),
                   nn.Linear(hidden_size, 2 * hidden_size)]

        self.encoder = nn.Sequential(*encoder)

        decoder = [nn.Linear(hidden_size, 2 * hidden_size), nn.ReLU(),
                   nn.Linear(2 * hidden_size, input_shape)]
        self.decoder = nn.Sequential(*decoder)
        # loss
        self.loss = nn.BCELoss()
    
    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def get_encoded_features(self, inputs):
        encoded = self.encoder(inputs)
        mean, log_var = torch.split(encoded, self.hidden_size, dim=1)
        encoded = self.reparameterization(mean, log_var)
        return encoded

    def kld_loss(self, inputs):
        encoded = self.encoder(inputs)
        mean, log_var = torch.split(encoded, self.hidden_size, dim=1)
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kld


class ConditionalNetwork(AutoencoderNetwork):
    def __init__(self, input_shape, condition_latent_size, output_shape, hidden_size, network_type):
        # network_type [conditional, vqvae]
        super(ConditionalNetwork, self).__init__(input_shape, output_shape, hidden_size)
        self.condition_latent_size = condition_latent_size
        self.network_type = network_type

        encoder = [nn.Linear(input_shape, 2 * hidden_size), nn.ReLU(),
                   nn.Linear(2 * hidden_size, hidden_size), nn.ReLU()]
        if self.network_type == 'vqvae':
            encoder.append(VectorQuantizer(64, hidden_size, 0.25))
        self.encoder = nn.Sequential(*encoder)
        # self.mean_layer = nn.Linear(hidden_size, hidden_size)
        # self.log_var_layer = nn.Linear(hidden_size, hidden_size)

        decoder = [nn.Linear(hidden_size + condition_latent_size, 2 * hidden_size), nn.ReLU(),
                   nn.Linear(2 * hidden_size, input_shape)]
        self.decoder = nn.Sequential(*decoder)
        self.optim = torch.optim.Adam(self.parameters())

        # loss
        self.loss = nn.BCELoss()

    def combine_onehot_and_encoded_feature(self, condition_latents, encoded_outputs):
        return torch.cat([condition_latents, encoded_outputs], 1)

    def decode(self, encoded_features):
        # expect [ condition, and features ]
        return self.decoder(encoded_features)

    def forward(self, inputs):
        condition_latents, current_inputs = inputs
        # mean, log_var, encoded_output = self.get_encoded_features(current_inputs)
        encoded_output = self.get_encoded_features(current_inputs)
        if self.network_type == 'vqvae':
            loss, encoded_features, persplexity = encoded_output
            encoded_features = self.combine_onehot_and_encoded_feature(condition_latents, encoded_features)
            reconstructed_output = self.decoder(encoded_features)
        else:
            encoded_output = self.combine_onehot_and_encoded_feature(condition_latents, encoded_output)
            reconstructed_output = self.decoder(encoded_output)
        return encoded_output, reconstructed_output
        # return mean, log_var, encoded_output, reconstructed_output


class Network(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers, hidden_size, conditional_latent_size=0):
        super(Network, self).__init__()
        self.conditional_latent_size = conditional_latent_size
        self.attention_layer = nn.MultiheadAttention(hidden_size, 8)

        if conditional_latent_size != 0:
            self.first_layer = nn.Linear(input_shape + conditional_latent_size, hidden_size)
        else:
            self.first_layer = nn.Linear(input_shape, input_shape*2)

        all_layers = [nn.Linear(input_shape, hidden_size), nn.ReLU()]

        for i in range(num_layers-1):
            all_layers.append(nn.Linear(hidden_size, hidden_size))
            all_layers.append(nn.ReLU())
            # all_layers.append(nn.Dropout(0.5))

        self.layers = nn.Sequential(*all_layers)
        if conditional_latent_size != 0:
            self.second_last_output_layer = nn.Linear(conditional_latent_size + hidden_size, hidden_size)
        else:
            self.second_last_output_layer = nn.Linear(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_shape)

        # loss
        self.loss = nn.BCELoss()

    def get_last_features(self, inputs):
        if self.conditional_latent_size != 0:
            x, onehot = inputs
            output = self.layers(torch.cat([onehot, x], 1))
            output = torch.cat([onehot, output], 1)
        else:
            output = self.layers(inputs)

        return F.relu(self.second_last_output_layer(output))

    def forward(self, inputs):
        if self.conditional_latent_size != 0:
            x, onehot = inputs
            output = F.relu(self.first_layer(torch.cat([onehot, x], 1)))
            output = self.layers(output)
            output = torch.cat([onehot, output], 1)
        else:
            output = F.glu(self.first_layer(inputs))
            output = self.layers(output)
            output = self.attention_layer(output.unsqueeze(0), output.unsqueeze(0), output.unsqueeze(0))[0][0]

        second_last_output = F.relu(self.second_last_output_layer(output))
        final_output = self.output_layer(second_last_output)
        return final_output
