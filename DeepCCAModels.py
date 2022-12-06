import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(
                    nn.Sequential(
                        nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                        nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                        nn.Sigmoid(),
                        nn.BatchNorm1d(
                            num_features=layer_sizes[l_id + 1], affine=False
                        ),
                    )
                )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepCCA(nn.Module):
    def __init__(
        self,
        layer_sizes1,
        layer_sizes2,
        input_size1,
        input_size2,
        outdim_size,
        use_all_singular_values,
        device=torch.device("cpu"),
    ):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2


class AutoEncoder(nn.Module):
    def __init__(self, num_layers, layer_size, input_size, output_size):
        super(AutoEncoder, self).__init__()
        layers = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else layer_size
            layer_output_size = output_size if i == num_layers - 1 else layer_size
            layers.append(
                nn.Sequential(
                    nn.Linear(layer_input_size, layer_output_size), nn.Sigmoid()
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DCCAE(nn.Module):
    def __init__(self, deep_cca_model, autoencoder):
        super(DCCAE, self).__init__()
        self.deep_cca_model = deep_cca_model
        self.autoencoder = autoencoder

        self.dcca_loss = self.deep_cca_model.loss
        self.ae_loss = nn.functional.mse_loss

    def forward(self, x1, x2):
        dcca_out1, dcca_out2 = self.deep_cca_model(x1, x2)
        ae_out1, ae_out2 = self.autoencoder(dcca_out1), self.autoencoder(dcca_out2)
        return dcca_out1, dcca_out2, ae_out1, ae_out2

    def loss(self, dcca_out1, dcca_out2, ae_out1, ae_out2, y1, y2):
        dcca_loss = self.deep_cca_model.loss(dcca_out1, dcca_out2)
        ae_loss_1 = self.ae_loss(ae_out1, y1)
        ae_loss_2 = self.ae_loss(ae_out2, y2)
        return dcca_loss + ae_loss_1 + ae_loss_2
