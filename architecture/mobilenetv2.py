import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(
            inp,
            oup,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        ),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()

        hidden_dim = int(round(inp * expand_ratio))

        self.use_res_connect = (
            stride == (1, 1, 1) and inp == oup
        )

        layers = []

        # expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # depthwise
        layers.extend([
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # projection
        layers.extend([
            nn.Conv3d(
                hidden_dim,
                oup,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm3d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        if self.use_res_connect:
            return x + self.conv(x)

        return self.conv(x)


class MobileNetV2Logit(nn.Module):

    def __init__(self, config_file, architecture_config):
        super().__init__()

        num_classes = config_file["train"]["num_classes"]
        width_mult = architecture_config["width_mult"]
        dropout = architecture_config["dropout_prob"]

        input_channel = 32
        last_channel = 1280

        input_channel = int(input_channel * width_mult)

        if width_mult > 1.0:
            last_channel = int(last_channel * width_mult)

        interverted_residual_setting = [
            # t, c, n, s

            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (2, 2, 2)],
            [6, 32, 3, (2, 2, 2)],
            [6, 64, 4, (2, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        features = [
            conv_bn(
                3,
                input_channel,
                stride=(1, 2, 2)
            )
        ]

        for t, c, n, s in interverted_residual_setting:

            output_channel = int(c * width_mult)

            for i in range(n):

                stride = s if i == 0 else (1, 1, 1)

                features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t
                    )
                )

                input_channel = output_channel

        features.append(
            conv_1x1x1_bn(
                input_channel,
                last_channel
            )
        )

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = F.avg_pool3d(
            x,
            kernel_size=x.shape[-3:]
        )

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


class MobileNetV2(nn.Module):

    def __init__(self, config_file, architecture_config):
        super().__init__()

        self.model = MobileNetV2Logit(
            config_file,
            architecture_config
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.model(x)

        return self.softmax(x)


def get_fine_tuning_parameters(model, ft_portion):

    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":

        ft_module_names = ['classifier']

        parameters = []

        for k, v in model.named_parameters():

            for ft_module in ft_module_names:

                if ft_module in k:
                    parameters.append({'params': v})
                    break

            else:
                parameters.append({
                    'params': v,
                    'lr': 0.0
                })

        return parameters

    else:
        raise ValueError(
            "Unsupported ft_portion"
        )


def get_model(**kwargs):
    return MobileNetV2(**kwargs)