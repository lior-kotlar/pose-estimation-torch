import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Network:
    def __init__(self, config, image_size, number_of_output_channels):
        self.config = config
        self.model_type = config['model type']
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.num_blocks = config["number of encoder decoder blocks"]
        self.kernel_size = config["convolution kernel size"]
        self.num_base_filters = config["number of base filters"]
        self.dilation_rate = config["dilation rate"]
        self.dropout = config["dropout ratio"]
        self.batches_per_epoch = config["batches per epoch"]
        self.model = self.config_model()

    class encoder_atrous(nn.Module):
        def __init__(self, img_size, num_base_filters, num_blocks,
                     kernel_size, dilation_rate, dropout):
            super(Network.encoder_atrous, self).__init__()
            layers = []
            in_channels = img_size[2]
            for block_idx in range(num_blocks):
                out_channels = num_base_filters*(2**block_idx)
                layers.append(nn.Conv2d(
                     in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     dilation=dilation_rate,
                     padding='same'
                ))
                layers.append(nn.LeakyReLU(inplace=True))
                layers.append(nn.Conv2d(
                     in_channels=out_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     dilation=dilation_rate,
                     padding='same'
                ))
                layers.append(nn.LeakyReLU(inplace=True))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # padding=0 usually
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=dropout))

                in_channels = out_channels

            out_channels = num_base_filters * (2 ** num_blocks)

            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                    padding='same',
                                    dilation=dilation_rate))
            layers.append(nn.LeakyReLU(inplace=True))

            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size,
                                    padding='same',
                                    dilation=dilation_rate))
            layers.append(nn.LeakyReLU(inplace=True))

            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size,
                                    padding='same',
                                    dilation=dilation_rate))
            layers.append(nn.LeakyReLU(inplace=True))

            layers.append(nn.Dropout(p=dropout))

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)
        
    class decoder(nn.Module):
        def __init__(self, input_channels, output_channels, 
                        num_base_filters, num_blocks, kernel_size):
            super(Network.decoder, self).__init__()
            layers = []
            in_channels = input_channels
            for block_idx in range(num_blocks-1, -1, -1):
                out_channels = num_base_filters * (2 ** block_idx)

                layers.append(nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding='same'))
                
                layers.append(nn.LeakyReLU(inplace=True))

                layers.append(nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding='same'))
                
                layers.append(nn.LeakyReLU(inplace=True))

                layers.append(nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding='same'))
                
                layers.append(nn.LeakyReLU(inplace=True))

                in_channels = out_channels

            layers.append(nn.ConvTranspose2d(
                in_channels, output_channels,
                kernel_size,
                stride=2,
                padding='same'))

            self.model = nn.Sequential(*layers)


        def forward(self, x):
            return self.model(x)

    class simple_network(nn.Module):

        def __init__(self, config, image_size, number_of_output_channels):
            super(Network.simple_network, self).__init__()
            self.model_type = config['model type']
            self.image_size = image_size
            self.number_of_output_channels = number_of_output_channels
            self.num_base_filters = config["number of base filters"]
            self.num_blocks = config["number of encoder decoder blocks"]
            self.kernel_size = config["convolution kernel size"]
            self.num_base_filters = config["number of base filters"]
            self.dilation_rate = config["dilation rate"]
            self.dropout = config["dropout ratio"]
            self.batches_per_epoch = config["batches per epoch"]
            self.encoder = Network.encoder_atrous(
                img_size=(self.image_size[0], self.image_size[1], self.image_size[2]),
                num_base_filters=self.num_base_filters,
                num_blocks=self.num_blocks,
                kernel_size=self.kernel_size,
                dilation_rate=self.dilation_rate,
                dropout=self.dropout
            )
            encoder_out_channels = self.num_base_filters * (2 ** self.num_blocks)
            self.decoder = Network.decoder(
                input_channels=encoder_out_channels,
                output_channels=self.number_of_output_channels,
                num_base_filters=self.num_base_filters,
                num_blocks=self.num_blocks,
                kernel_size=self.kernel_size
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def config_model(self):
        
        # if self.model_type == ALL_CAMS or self.model_type == ALL_CAMS_18_POINTS or self.model_type == ALL_CAMS_ALL_POINTS:
        #     model = self.all_4_cams()
        # elif self.model_type == ALL_CAMS_AND_3_GOOD_CAMS:
        #     model = self.all_3_cams()
        # elif self.model_type == TWO_WINGS_TOGATHER:
        #     model = self.two_wings_net()
        # elif self.model_type == HEAD_TAIL_ALL_CAMS:
        #     model = self.head_tail_all_cams()
        # elif self.model_type == C2F_PER_WING:
        #     model = self.C2F_per_wing()
        # elif self.model_type == COARSE_PER_WING:
        #     model = self.coarse_per_wing()
        # elif self.model_type == MODEL_18_POINTS_PER_WING_VIT or self.model_type == ALL_POINTS_MODEL_VIT:
        #     model = self.get_transformer()
        # elif self.model_type == RESNET_18_POINTS_PER_WING:
        #     model = self.resnet50_encoder_shallow_decoder()
        # else:
        #     model = self.simple_network()


        model = self.simple_network(self.config, self.image_size, self.number_of_output_channels)
        return model

    def get_model(self):
        return self.model         