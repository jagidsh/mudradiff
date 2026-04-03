import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=28):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        # Initial convolution
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Downsampling layers
        self.down1 = UnetDown(n_feat, n_feat)          # 256 -> 128
        self.down2 = UnetDown(n_feat, 2 * n_feat)      # 128 -> 64
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)  # 64 -> 32

        # Update pooling for 128x128 resolution
        self.to_vec = nn.Sequential(nn.AvgPool2d(16), nn.GELU())  # 32x32 -> 1x1

        # Embedding layers
        self.timeembed1 = EmbedFC(1, 4 * n_feat)
        self.timeembed2 = EmbedFC(1, 2 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 4 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 2 * n_feat)

        # Upsampling layers
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, 16, 16),  # 1x1 -> 32x32
            nn.GroupNorm(8, 4 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(8 * n_feat, 2 * n_feat)  # 32 -> 64
        self.up2 = UnetUp(4 * n_feat, n_feat)      # 64 -> 128
        self.up3 = UnetUp(2 * n_feat, n_feat)      # 128 -> 256

        # Final output layers
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # Initial downsampling
        x = self.init_conv(x)
        down1 = self.down1(x)  # 256 -> 128
        down2 = self.down2(down1)  # 128 -> 64
        down3 = self.down3(down2)  # 64 -> 32

        # Transition to latent space
        hiddenvec = self.to_vec(down3)

        # Context embeddings
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        context_mask = context_mask[:, None].repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask))
        c = c * context_mask

        # Temporal and context embeddings
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 4, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 4, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 2, 1, 1)

        # Upsampling with skip connections
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down3)
        up3 = self.up2(cemb2 * up2 + temb2, down2)
        up4 = self.up3(up3, down1)

        # Final output
        out = self.out(torch.cat((up4, x), 1))
        return out


