import torch
import torch.nn as nn

import pytorch_lightning as pl
import wandb


class LightningNetwork(pl.LightningModule):
    def __init__(self, input_shape, output_shape, num_layers, hidden_size):
        super(LightningNetwork, self).__init__()
        self.network = Network(input_shape, output_shape, num_layers, hidden_size)
        self.loss = nn.BCELoss()

    def forward(self, inputs):
        if type(inputs) != torch.Tensor:
            inputs = torch.from_numpy(inputs)
        return self.network(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def on_train_start(self):
        self.train()

    def training_step(self, train_batch, batch_index):
        x, y = train_batch
        predicted = torch.sigmoid(self.network(x))
        loss = self.loss(predicted, y)
        self.log('train_loss', loss)

        wandb.log({"train_loss": loss})
        return loss

    def on_validation_start(self):
        self.eval()

    def validation_step(self, val_batch, batch_index):
        x, y = val_batch
        predicted = torch.sigmoid(self.network(x))
        loss = self.loss(predicted, y)
        self.log('val_loss', loss, prog_bar=True)

        wandb.log({"val_loss": loss})
        return loss


class LightningAutoencoderNetwork(LightningNetwork):

    def __init__(self, input_shape, output_shape, num_layers, hidden_size):
        super(LightningAutoencoderNetwork, self).__init__(input_shape, output_shape, num_layers, hidden_size)
        self.network = AutoencoderNetwork(input_shape, output_shape, hidden_size)
        self.loss = nn.L1Loss()

    def forward(self, inputs):
        if type(inputs) != torch.Tensor:
            inputs = torch.from_numpy(inputs)
        return self.network(inputs)

    def training_step(self, train_batch, batch_index):
        x, y = train_batch
        encoded_output, reconstructed_output = self.network(x)
        loss = self.loss(reconstructed_output, x)
        self.log('train_autoencoder_loss', loss)

        wandb.log({"train_autoencoder_loss": loss})
        return loss

    def validation_step(self, val_batch, batch_index):
        x, y = val_batch
        encoded_output, reconstructed_output = self.network(x)
        loss = self.loss(reconstructed_output, x)
        self.log('val_autoencoder_loss', loss)

        wandb.log({"val_autoencoder_loss": loss})
        return loss


class LightningMultiAutoencoderNetwork(LightningAutoencoderNetwork):

    def __init__(self, input_shape, output_shape, num_layers, hidden_size):
        super(LightningMultiAutoencoderNetwork, self).__init__(input_shape, output_shape, num_layers, hidden_size)
        self.network = MultiAutoencoderNetwork(input_shape, output_shape, hidden_size)
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
        mean, log_var, latent = self.network.get_encoded_features(inputs)
        return latent

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

    def __init__(self, input_shape, output_shape, num_layers, hidden_size):
        super(LightningJointMultiAutoencoderNetwork, self).__init__(input_shape, input_shape, num_layers, hidden_size)
        self.network = MultiAutoencoderNetwork(input_shape, hidden_size)
        self.classify_network = Network(hidden_size, output_shape, num_layers, hidden_size)
        self.loss = nn.MSELoss()
        self.classify_loss = nn.BCELoss()
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
        mean, log_var, latent = self.network.get_encoded_features(inputs)
        return latent

    def training_step(self, train_batch, batch_index):
        x, y = train_batch
        # make sure batch size is one because this only work with one batch size since we have many decoders
        assert x.shape[0] == 1

        mean, log_var, encoded_output, reconstructed_output = self.network(x, self.decoder_index)
        output = self.classify_network(encoded_output)
        reconstruction_loss = self.loss(reconstructed_output, x)
        output_loss = self.classify_loss(output, y)
        kld_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        loss = reconstruction_loss + kld_loss + output_loss

        self.log('train_autoencoder_loss', reconstruction_loss + kld_loss)
        self.log('train_output_loss', output_loss)

        wandb.log({"train_autoencoder_loss": reconstruction_loss + kld_loss,
                   "train_output_loss": output_loss})
        return loss

    def validation_step(self, val_batch, batch_index):
        x, y = val_batch
        mean, log_var, encoded_output, reconstructed_output = self.network(x, self.decoder_index)
        output = self.classify_network(encoded_output)
        reconstruction_loss = self.loss(reconstructed_output, x)
        kld_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        output_loss = self.classify_loss(output, y)

        loss = reconstruction_loss + kld_loss + output_loss

        self.log('val_autoencoder_loss', reconstruction_loss + kld_loss)
        self.log('val_output_loss', output_loss)

        wandb.log({"val_autoencoder_loss": reconstruction_loss + kld_loss,
                   "val_output_loss": output_loss})

        return loss


class MultiAutoencoderNetwork(nn.Module):
    def __init__(self, input_shape, hidden_size, num_decoders=5):
        super(MultiAutoencoderNetwork, self).__init__()

        encoder = [nn.Linear(input_shape, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                   nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5)]
        self.encoder = nn.Sequential(*encoder)
        self.mean_layer = nn.Linear(hidden_size, hidden_size)
        self.log_var_layer = nn.Linear(hidden_size, hidden_size)

        # have multiple decoders for different studies
        self.decoders = []
        decoders_param = None
        for _ in range(num_decoders):
            decoder = [nn.Linear(hidden_size, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
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

    def get_encoded_features(self, inputs):
        encoded = self.encoder(inputs)
        mean, log_var = self.mean_layer(encoded), self.log_var_layer(encoded)
        return mean, log_var, self.reparameterization(mean, log_var)

    def forward(self, inputs, decoder_index):
        """

        :param inputs:
        :param decoder_index:   Which decoder to use
        :return:
        """
        mean, log_var, encoded_output = self.get_encoded_features(inputs)
        reconstructed_output = self.decoders[decoder_index](encoded_output)
        return mean, log_var, encoded_output, reconstructed_output


class AutoencoderNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size):
        super(AutoencoderNetwork, self).__init__()

        encoder = [nn.Linear(input_shape, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                   nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5)]
        self.encoder = nn.Sequential(*encoder)

        decoder = [nn.Linear(hidden_size, 2 * hidden_size), nn.ReLU(), nn.Dropout(0.5),
                   nn.Linear(2 * hidden_size, input_shape)]
        self.decoder = nn.Sequential(*decoder)
        self.optim = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()))

        # loss
        self.loss = nn.BCELoss()

    def forward(self, inputs):
        encoded_output = self.encoder(inputs)
        reconstructed_output = self.decoder(encoded_output)
        return encoded_output, reconstructed_output


class Network(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers, hidden_size):
        super(Network, self).__init__()

        first_layer = nn.Linear(input_shape, hidden_size)

        all_layers = [first_layer, nn.ReLU()]

        for i in range(num_layers):
            all_layers.append(nn.Linear(hidden_size, hidden_size))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(0.5))

        output_layer = nn.Linear(hidden_size, output_shape)
        all_layers.append(output_layer)

        self.layers = nn.Sequential(*all_layers)
        self.optim = torch.optim.Adam(self.layers.parameters())

        # loss
        self.loss = nn.BCELoss()

    def forward(self, inputs):
        output = self.layers(inputs)
        return output

