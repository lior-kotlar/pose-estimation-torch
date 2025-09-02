import torch
import torch.nn as nn
import torch.nn.functional as F


class Network:
    def __init__(self, config, image_size, number_of_output_channels):
        self.model_type = config['model type']
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.num_blocks = config["number of encoder decoder blocks"]
        self.kernel_size = config["convolution kernel size"]
        self.num_base_filters = config["number of base filters"]
        self.learning_rate = config["learning rate"]
        self.loss_function = config["loss_function"]
        self.dilation_rate = config["dilation rate"]
        self.dropout = config["dropout ratio"]
        self.batches_per_epoch = config["batches per epoch"]

    
    class encoder_atrous(nn.Module):
        def __init__(self, img_size, num_base_filters, num_blocks,
                     kernel_size, dilation_rate, dropout):
            super(Network.simple_network, self).__init__()
            layers = []
            in_channels = img_size
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
            
    class simple_network:
        def __init__(self, config, image_size, number_of_output_channels):
            self.model_type = config['model type']
            self.image_size = image_size
            self.number_of_output_channels = number_of_output_channels
            self.num_base_filters = config["number of base filters"]
            self.num_blocks = config["number of encoder decoder blocks"]
            self.kernel_size = config["convolution kernel size"]
            self.num_base_filters = config["number of base filters"]
            self.learning_rate = config["learning rate"]
            self.loss_function = config["loss_function"]
            self.dilation_rate = config["dilation rate"]
            self.dropout = config["dropout ratio"]
            self.batches_per_epoch = config["batches per epoch"]

            