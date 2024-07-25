import torch.nn as nn
import torch 
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

num_epochs = 10

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, activation=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool_out = self.pool(x)
        fc_out = self.fc(pool_out)
        act_out = self.act(fc_out)
        return x * act_out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, activation=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = activation

    def forward(self, x):
        mean_out = torch.mean(x, 1, keepdim=True)
        max_out = torch.max(x, 1, keepdim=True)[0]
        concat_out = torch.cat([mean_out, max_out], 1)
        cv1_out = self.cv1(concat_out)
        act_out = self.act(cv1_out)
        return x * act_out

class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7, activation=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, activation)
        self.spatial_attention = SpatialAttention(kernel_size, activation)

    def forward(self, x):
        ca_out = self.channel_attention(x)
        sa_out = self.spatial_attention(ca_out)
        return sa_out

class ResBlock_CBAM(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, expansion=1, downsampling=False):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        activation = nn.LeakyReLU(0.1, inplace=True)  # Activation function used in YOLOv9
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            activation,
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            activation,
            nn.Conv2d(in_channels=c2, out_channels=c2 * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2 * self.expansion),
        )
        
        self.cbam = CBAM(c1=c2 * self.expansion, activation=activation)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=c1, out_channels=c2 * self.expansion, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(c2 * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Clear the cache after each step
        torch.cuda.empty_cache()

        # Print memory usage after each step
        print(f"Memory Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Memory Cached: {torch.cuda.memory_reserved()} bytes")

