import torch
import torch.nn as nn
from utils import TrainConfig
from constants import ALL_CAMS_18_POINTS, ALL_CAMS_ALL_POINTS


class Network:
    def __init__(self, general_configuration: TrainConfig, image_size, number_of_output_channels):
        self.model_type = general_configuration.get_model_type()
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.model = self.config_model(general_configuration=general_configuration)

    class encoder_atrous(nn.Module):
        def __init__(
                        self,
                        img_size,
                        num_base_filters,
                        num_blocks,
                        kernel_size,
                        dilation_rate,
                        weight_init_method_str,
                        dropout
                    ):
            super(Network.encoder_atrous, self).__init__()
            weight_init_function = Network.config_init_method(self, weight_init_method_str)
            layers = []
            in_channels = img_size[0]
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

            self.out_channels = num_base_filters * (2 ** num_blocks)

            layers.append(nn.Conv2d(in_channels, self.out_channels, kernel_size,
                                    padding='same',
                                    dilation=dilation_rate))
            layers.append(nn.LeakyReLU(inplace=True))

            layers.append(nn.Conv2d(self.out_channels, self.out_channels, kernel_size,
                                    padding='same',
                                    dilation=dilation_rate))
            layers.append(nn.LeakyReLU(inplace=True))

            layers.append(nn.Conv2d(self.out_channels, self.out_channels, kernel_size,
                                    padding='same',
                                    dilation=dilation_rate))
            layers.append(nn.LeakyReLU(inplace=True))

            layers.append(nn.Dropout(p=dropout))

            self.model = nn.Sequential(*layers)

            self.model.apply(lambda m: Network.init_weights(self, m, weight_init_function))

        def forward(self, x):
            return self.model(x)
        
        def get_out_channels(self):
            return self.out_channels

    class decoder(nn.Module):
        def __init__(
                self,
                input_channels,
                output_channels,
                weight_init_method_str,
                num_base_filters,
                num_blocks,
                kernel_size
                ):
            super(Network.decoder, self).__init__()
            weight_init_function = Network.config_init_method(self, weight_init_method_str)
            layers = []
            in_channels = input_channels
            for block_idx in range(num_blocks-1, -1, -1):
                out_channels = num_base_filters * (2 ** block_idx)

                layers.append(nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1))
                
                layers.append(nn.LeakyReLU(inplace=True))

                layers.append(nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=1))
                
                layers.append(nn.LeakyReLU(inplace=True))

                layers.append(nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=1))
                
                layers.append(nn.LeakyReLU(inplace=True))

                in_channels = out_channels

            layers.append(nn.Conv2d(
                in_channels,
                10,          # fixed output channels
                kernel_size=1
            ))

            self.model = nn.Sequential(*layers)

            self.model.apply(lambda m: Network.init_weights(self, m, weight_init_function))

        def forward(self, x):
            return self.model(x)

    class simple_network(nn.Module):

        def __init__(self, general_configuration: TrainConfig, image_size, number_of_output_channels):
            super(Network.simple_network, self).__init__()
            image_size = image_size
            number_of_output_channels = number_of_output_channels

            num_base_filters,\
            num_blocks,\
            kernel_size,\
            dilation_rate,\
            enc_weight_init_str,\
            dec_weight_init_str,\
            dropout = general_configuration.get_network_configuration()
            
            self.encoder = Network.encoder_atrous(
                img_size=(image_size[0], image_size[1], image_size[2]),
                num_base_filters=num_base_filters,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                weight_init_method_str=enc_weight_init_str,
                dropout=dropout
            )
            encoder_out_channels = num_base_filters * (2 ** num_blocks)
            self.decoder = Network.decoder(
                input_channels=encoder_out_channels,
                output_channels=number_of_output_channels,
                weight_init_method_str=dec_weight_init_str,
                num_base_filters=num_base_filters,
                num_blocks=num_blocks,
                kernel_size=kernel_size
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    class FourCamsNetwork(nn.Module):
        NUM_OF_CAMS = 4
        def __init__(self, general_configuration: TrainConfig, image_size, number_of_output_channels):
            super(Network.FourCamsNetwork, self).__init__()
            image_size = image_size
            number_of_output_channels = number_of_output_channels

            num_base_filters,\
            num_blocks,\
            kernel_size,\
            dilation_rate,\
            dropout = general_configuration.get_network_configuration()

            total_input_channels = image_size[0]
            self.channels_per_cam = total_input_channels // self.NUM_OF_CAMS

            self.shared_encoder = Network.encoder_atrous(
                img_size=(image_size[0]//self.NUM_OF_CAMS, image_size[1], image_size[2]),
                num_base_filters=num_base_filters,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                dropout=dropout
            )

            encoder_out_channels = self.shared_encoder.get_out_channels()
            decoder_input_channels = (self.NUM_OF_CAMS + 1) * encoder_out_channels

            self.shared_decoder = Network.decoder(
                input_channels=decoder_input_channels,
                output_channels=number_of_output_channels//self.NUM_OF_CAMS,
                num_base_filters=num_base_filters,
                num_blocks=num_blocks,
                kernel_size=kernel_size
            )

        def forward(self, x):
            splits = torch.split(x, self.channels_per_cam, dim=1)
            x_in_split_1 = splits[0]
            x_in_split_2 = splits[1]
            x_in_split_3 = splits[2]
            x_in_split_4 = splits[3]

            # 4. Shared Encoding
            # Call the *same* module on each split
            code_out_1 = self.shared_encoder(x_in_split_1)
            code_out_2 = self.shared_encoder(x_in_split_2)
            code_out_3 = self.shared_encoder(x_in_split_3)
            code_out_4 = self.shared_encoder(x_in_split_4)

            # 5. Global Feature Merging
            # Concatenate along the channel dimension (dim=1)
            x_code_merge = torch.cat([code_out_1, code_out_2, code_out_3, code_out_4], dim=1)
            # Shape is (B, 4 * C_enc, H_feat, W_feat)

            # 6. Shared Decoding (Local + Global)
            # We also concatenate along the channel dimension (dim=1)
            map_out_1 = self.shared_decoder(torch.cat([code_out_1, x_code_merge], dim=1))
            map_out_2 = self.shared_decoder(torch.cat([code_out_2, x_code_merge], dim=1))
            map_out_3 = self.shared_decoder(torch.cat([code_out_3, x_code_merge], dim=1))
            map_out_4 = self.shared_decoder(torch.cat([code_out_4, x_code_merge], dim=1))

            # 7. Final Output Merging
            # Concatenate along the channel dimension (dim=1)
            x_maps_merge = torch.cat([map_out_1, map_out_2, map_out_3, map_out_4], dim=1)
            return x_maps_merge


    def config_model(self, general_configuration: TrainConfig):
        
        # if self.model_type == ALL_CAMS or self.model_type == ALL_CAMS_18_POINTS or self.model_type == ALL_CAMS_ALL_POINTS:
        #     model = self.all_4_cams()
        # elif self.model_type == ALL_CAMS_AND_3_GOOD_CAMS:
        #     model = self.all_3_cams()
        # else:
        #     model = self.simple_network()
        if self.model_type == ALL_CAMS_18_POINTS or self.model_type == ALL_CAMS_ALL_POINTS:
            model = self.FourCamsNetwork(
                general_configuration=general_configuration,
                image_size=self.image_size,
                number_of_output_channels=self.number_of_output_channels)
        
        else:
            model = self.simple_network(
                general_configuration=general_configuration,
                image_size=self.image_size,
                number_of_output_channels=self.number_of_output_channels)
        
        return model
    
    def config_init_method(self, weight_init_method_str):
        weight_init_method_str = weight_init_method_str.lower()
        weight_init_function = None
        if weight_init_method_str == "xavier_uniform":
            weight_init_function = nn.init.xavier_uniform_
        elif weight_init_method_str == "xavier_normal":
            weight_init_function = nn.init.xavier_normal_
        elif weight_init_method_str == "kaiming_uniform":
            weight_init_function = nn.init.kaiming_uniform_
        else:
            weight_init_function = nn.init.kaiming_normal_
        return weight_init_function

    def init_weights(self, m, weight_init_function):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            weight_init_function(m.weight)

    def get_model(self):
        return self.model         