import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl
import wandb


class LightningNetwork(pl.LightningModule):
    def __init__(self, input_shape, output_shape, num_layers, hidden_size, conditional_latent_size=0):
        super(LightningNetwork, self).__init__()
        self.conditional_latent_size = conditional_latent_size
        self.network = Network(input_shape, output_shape, num_layers, hidden_size, conditional_latent_size)
        self.loss = nn.BCELoss()

    def forward(self, inputs):
        if self.conditional_latent_size != 0:
            current_inputs, onehot = inputs

        if type(current_inputs) != torch.Tensor:
            current_inputs = torch.from_numpy(current_inputs)
        if self.conditional_latent_size != 0:
            return self.network([current_inputs, onehot])

        return self.network(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
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

    def validation_step(self, val_batch, batch_index):
        if self.conditional_latent_size != 0:
            x, onehot, y = val_batch
            predicted = torch.sigmoid(self.network([x, onehot]))
        else:
            x, y = val_batch
            predicted = torch.sigmoid(self.network(x))
        loss = self.loss(predicted, y)
        self.log('val_loss', loss, prog_bar=True)

        wandb.log({"val_loss": loss})
        return loss


class LightningAutoencoderNetwork(LightningNetwork):

    def __init__(self, input_shape, output_shape, condition_latent_size, num_layers, hidden_size, network_type='normal'):
        # network_type can be 'normal' or 'conditional'
        super(LightningAutoencoderNetwork, self).__init__(input_shape, output_shape, num_layers, hidden_size)

        networks_dict = {
            'conditional': lambda: ConditionalNetwork(input_shape, condition_latent_size, output_shape, hidden_size),
            'normal': lambda: AutoencoderNetwork(input_shape, output_shape, hidden_size)
        }

        self.network_type = network_type
        self.network = networks_dict[network_type]()
        self.loss = nn.L1Loss()

    def get_encoded_features(self, inputs):
        if type(inputs) != torch.Tensor:
            inputs = torch.from_numpy(inputs)
        return self.network.get_encoded_features(inputs)

    def forward(self, inputs):
        if type(inputs) != torch.Tensor:
            inputs = torch.from_numpy(inputs)
        return self.network(inputs)

    def training_step(self, train_batch, batch_index):
        x, onehot_decoder_indexes, y = train_batch

        if self.network_type == 'conditional':
            mean, log_var, encoded_output, reconstructed_output = self.network([onehot_decoder_indexes, x])
        else:
            mean, log_var, encoded_output, reconstructed_output = self.network(x)

        reconstruction_loss = self.loss(reconstructed_output, x)
        kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = reconstruction_loss + kld_loss
        # output_loss = self.classify_loss(output, y)

        self.log('train_autoencoder_loss', loss)

        wandb.log({"train_autoencoder_loss": loss})
        return loss

    def validation_step(self, val_batch, batch_index):
        x, onehot_decoder_indexes, y = val_batch
        mean, log_var, encoded_output, reconstructed_output = self.network([onehot_decoder_indexes, x])

        reconstruction_loss = self.loss(reconstructed_output, x)
        kld_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = reconstruction_loss + kld_loss
        self.log('val_autoencoder_loss', loss)

        wandb.log({"val_autoencoder_loss": loss})
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
            self.optim = torch.optim.Adam(list(self.decoder.parameters()) + encoders_param)
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
            self.optim = torch.optim.Adam(list(self.encoder.parameters()) + decoders_param)

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
    def __init__(self, input_shape, output_shape, hidden_size):
        super(AutoencoderNetwork, self).__init__()

        encoder = [nn.Linear(input_shape, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                   nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5)]
        self.encoder = nn.Sequential(*encoder)
        self.mean_layer = nn.Linear(hidden_size, hidden_size)
        self.log_var_layer = nn.Linear(hidden_size, hidden_size)

        decoder = [nn.Linear(hidden_size, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                   nn.Linear(2 * hidden_size, input_shape)]
        self.decoder = nn.Sequential(*decoder)
        self.optim = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()))

        # loss
        self.loss = nn.BCELoss()

    def reparameterization(self, mean, log_var):
        epsilon = torch.randn_like(log_var)
        z = mean + log_var * epsilon
        return z

    def get_encoded_features(self, inputs):
        encoded = self.encoder(inputs)
        mean, log_var = self.mean_layer(encoded), self.log_var_layer(encoded)
        return mean, log_var, self.reparameterization(mean, log_var)

    def forward(self, inputs):
        mean, log_var, encoded_output = self.get_encoded_features(inputs)
        reconstructed_output = self.decoders(encoded_output)
        return mean, log_var, encoded_output, reconstructed_output


class ConditionalNetwork(AutoencoderNetwork):
    def __init__(self, input_shape, condition_latent_size, output_shape, hidden_size):
        super(ConditionalNetwork, self).__init__(input_shape, output_shape, hidden_size)
        self.condition_latent_size = condition_latent_size

        encoder = [nn.Linear(input_shape, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                   nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5)]
        self.encoder = nn.Sequential(*encoder)
        self.mean_layer = nn.Linear(hidden_size, hidden_size)
        self.log_var_layer = nn.Linear(hidden_size, hidden_size)

        decoder = [nn.Linear(hidden_size + condition_latent_size, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                   nn.Linear(2 * hidden_size, input_shape)]
        self.decoder = nn.Sequential(*decoder)
        self.optim = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()))

        # loss
        self.loss = nn.BCELoss()

    def combine_onehot_and_encoded_feature(self, condition_latents, encoded_outputs):
        return torch.cat([condition_latents, encoded_outputs], 1)

    def forward(self, inputs):
        condition_latents, current_inputs = inputs
        mean, log_var, encoded_output = self.get_encoded_features(current_inputs)
        encoded_output = self.combine_onehot_and_encoded_feature(condition_latents, encoded_output)
        reconstructed_output = self.decoder(encoded_output)
        return mean, log_var, encoded_output, reconstructed_output


class Network(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers, hidden_size, conditional_latent_size=0):
        super(Network, self).__init__()
        self.conditional_latent_size = conditional_latent_size

        if conditional_latent_size != 0:
            first_layer = nn.Linear(input_shape + conditional_latent_size, hidden_size)
        else:
            first_layer = nn.Linear(input_shape, hidden_size)

        all_layers = [first_layer, nn.ReLU()]

        for i in range(num_layers):
            all_layers.append(nn.Linear(hidden_size, hidden_size))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(0.5))

        self.layers = nn.Sequential(*all_layers)
        if conditional_latent_size != 0:
            self.second_last_output_layer = nn.Linear(conditional_latent_size + hidden_size, hidden_size)
        else:
            self.second_last_output_layer = nn.Linear(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_shape)

        self.optim = torch.optim.Adam(self.layers.parameters())

        # loss
        self.loss = nn.BCELoss()

    def forward(self, inputs):
        x, onehot = inputs
        if self.conditional_latent_size != 0:
            output = self.layers(torch.cat([onehot, x], 1))
            output = torch.cat([onehot, output], 1)
        else:
            output = self.layers(x)

        second_last_output = self.second_last_output_layer(output)
        final_output = self.output_layer(second_last_output)
        return final_output

