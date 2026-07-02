import torch
import torch.nn as nn
from utils import TrainConfig
from constants import ALL_CAMS_PER_WING, ALL_CAMS_ALL_WINGS, MODEL_PER_CAM_PER_WING, MODEL_PER_CAM_PER_WING_UNET


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
                layers.append(nn.Conv2d(
                     in_channels=out_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     dilation=dilation_rate,
                     padding='same'
                ))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
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

            self.layers = nn.ModuleList(layers)

            self.layers.apply(lambda m: Network.init_weights(self, m, weight_init_function))

        def forward(self, x):
            """
            Processes the input through each layer sequentially for easy debugging.
            """
            for i, layer in enumerate(self.layers):
                # You can place a breakpoint here to inspect 'x' after each layer
                x = layer(x)
                # print(f"After layer {i}, x.shape: {x.shape}") # Optional: Print for inspection
                
            return x
        
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
                output_channels,          # fixed output channels
                kernel_size=1
            ))

            self.layers = nn.ModuleList(layers)

            self.layers.apply(lambda m: Network.init_weights(self, m, weight_init_function))

        def forward(self, x):
            """
            Processes the input through each layer sequentially for easy debugging.
            """
            for i, layer in enumerate(self.layers):
                # You can place a breakpoint here to inspect 'x' after each layer
                x = layer(x)
                # print(f"After layer {i}, x.shape: {x.shape}") # Optional: Print for inspection
                
            return x

    class simple_network(nn.Module):

        def __init__(self, general_configuration: TrainConfig, image_size, number_of_output_channels):
            super(Network.simple_network, self).__init__()
            image_size = image_size
            number_of_output_channels = number_of_output_channels

            num_base_filters,\
            num_blocks,\
            kernel_size,\
            dilation_rate,\
            weight_init_str,\
            dropout = general_configuration.get_network_configuration()
            
            self.encoder = Network.encoder_atrous(
                img_size=(image_size[0], image_size[1], image_size[2]),
                num_base_filters=num_base_filters,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                weight_init_method_str=weight_init_str,
                dropout=dropout
            )
            encoder_out_channels = num_base_filters * (2 ** num_blocks)
            self.decoder = Network.decoder(
                input_channels=encoder_out_channels,
                output_channels=number_of_output_channels,
                weight_init_method_str=weight_init_str,
                num_base_filters=num_base_filters,
                num_blocks=num_blocks,
                kernel_size=kernel_size
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        
        def get_model_type(self):
            return MODEL_PER_CAM_PER_WING

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
            weight_init_str,\
            dropout = general_configuration.get_network_configuration()

            total_input_channels = image_size[0]
            self.channels_per_cam = total_input_channels // self.NUM_OF_CAMS

            self.shared_encoder = Network.encoder_atrous(
                img_size=(image_size[0]//self.NUM_OF_CAMS, image_size[1], image_size[2]),
                num_base_filters=num_base_filters,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                weight_init_method_str=weight_init_str,
                dropout=dropout
            )

            encoder_out_channels = self.shared_encoder.get_out_channels()
            decoder_input_channels = (self.NUM_OF_CAMS + 1) * encoder_out_channels

            self.shared_decoder = Network.decoder(
                input_channels=decoder_input_channels,
                output_channels=number_of_output_channels//self.NUM_OF_CAMS,
                weight_init_method_str=weight_init_str,
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
        
        def get_model_type(self):
            return ALL_CAMS_PER_WING

    class UNet(nn.Module):
        """
        A true U-Net: encoder/decoder WITH skip connections, dilated convs and
        OPTIONAL normalization. Same input/output contract as simple_network
        (one camera, one wing):
            input  (B, C_in, H, W)  ->  output (B, num_output_channels, H, W)
        so it trains on the MODEL_PER_CAM_PER_WING preprocessing and is served
        at prediction time through the existing PER_WING_PER_CAM path.

        The skip connections are what make its errors decorrelate from the
        skip-less encoder_atrous networks (the ensemble's per-point selection
        exploits that). Dilation matches the receptive field of the proven
        encoder_atrous recipe; normalization defaults to "none" because the
        other models train fine without it at 2 blocks and GroupNorm appeared
        to over-smooth the heatmap peaks (set "normalization" in the config to
        "group"/"batch" only if you go deeper and it stops training).
        """
        def __init__(self, general_configuration: TrainConfig, image_size, number_of_output_channels):
            super(Network.UNet, self).__init__()

            num_base_filters, \
            num_blocks, \
            kernel_size, \
            dilation_rate, \
            weight_init_str, \
            dropout = general_configuration.get_network_configuration()
            norm = general_configuration.get_normalization()

            in_channels = image_size[0]

            # --- contracting path ---
            self.encoders = nn.ModuleList()
            self.pools = nn.ModuleList()
            skip_channels = []
            channels = in_channels
            for block_idx in range(num_blocks):
                out_channels = num_base_filters * (2 ** block_idx)
                self.encoders.append(Network.UNet.double_conv(channels, out_channels, kernel_size, dilation_rate, dropout, norm))
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
                skip_channels.append(out_channels)
                channels = out_channels

            # --- bottleneck ---
            bottleneck_channels = num_base_filters * (2 ** num_blocks)
            self.bottleneck = Network.UNet.double_conv(channels, bottleneck_channels, kernel_size, dilation_rate, dropout, norm)

            # --- expanding path ---
            self.ups = nn.ModuleList()
            self.decoders = nn.ModuleList()
            channels = bottleneck_channels
            for block_idx in range(num_blocks - 1, -1, -1):
                out_channels = num_base_filters * (2 ** block_idx)
                self.ups.append(nn.ConvTranspose2d(channels, out_channels,
                                                   kernel_size=4, stride=2, padding=1))
                # decoder input = upsampled features (out_channels) concatenated
                # with the matching encoder skip (skip_channels[block_idx]).
                self.decoders.append(Network.UNet.double_conv(out_channels + skip_channels[block_idx],
                                                              out_channels, kernel_size, dilation_rate, dropout, norm))
                channels = out_channels

            self.final = nn.Conv2d(channels, number_of_output_channels, kernel_size=1)

            weight_init_function = Network.config_init_method(self, weight_init_str)
            self.apply(lambda m: Network.init_weights(self, m, weight_init_function))

        @staticmethod
        def _num_groups(num_channels):
            # largest group count in {8,4,2,1} that divides the channel count
            for groups in (8, 4, 2, 1):
                if num_channels % groups == 0:
                    return groups
            return 1

        @staticmethod
        def _norm_layer(norm, num_channels):
            if norm == "group":
                return nn.GroupNorm(Network.UNet._num_groups(num_channels), num_channels)
            elif norm == "batch":
                return nn.BatchNorm2d(num_channels)
            else:  # "none" - matches the norm-free encoder_atrous/decoder nets
                return nn.Identity()

        @staticmethod
        def double_conv(in_channels, out_channels, kernel_size, dilation, dropout, norm):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', dilation=dilation),
                Network.UNet._norm_layer(norm, out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', dilation=dilation),
                Network.UNet._norm_layer(norm, out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=dropout),
            )

        def forward(self, x):
            skips = []
            for encoder, pool in zip(self.encoders, self.pools):
                x = encoder(x)
                skips.append(x)
                x = pool(x)
            x = self.bottleneck(x)
            for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
                x = up(x)
                # ConvTranspose(k=4,s=2,p=1) and MaxPool(2) both halve/double
                # exactly for the fixed 192x192 input, so spatial dims align.
                x = torch.cat([x, skip], dim=1)
                x = decoder(x)
            x = self.final(x)
            return x

        def get_model_type(self):
            return MODEL_PER_CAM_PER_WING_UNET


    def config_model(self, general_configuration: TrainConfig):
        # if self.model_type == ALL_CAMS or self.model_type == ALL_CAMS_18_POINTS or self.model_type == ALL_CAMS_ALL_POINTS:
        #     model = self.all_4_cams()
        # elif self.model_type == ALL_CAMS_AND_3_GOOD_CAMS:
        #     model = self.all_3_cams()
        # else:
        #     model = self.simple_network()
        if self.model_type == ALL_CAMS_PER_WING or self.model_type == ALL_CAMS_ALL_WINGS:
            model = self.FourCamsNetwork(
                general_configuration=general_configuration,
                image_size=self.image_size,
                number_of_output_channels=self.number_of_output_channels)
        
        elif self.model_type == MODEL_PER_CAM_PER_WING:
            model = self.simple_network(
                general_configuration=general_configuration,
                image_size=self.image_size,
                number_of_output_channels=self.number_of_output_channels)

        elif self.model_type == MODEL_PER_CAM_PER_WING_UNET:
            model = self.UNet(
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