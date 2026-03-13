import torch
import torch.nn as nn


class SmallBlock(nn.Module):
    """
    Small DroNet-style block.
    First convolution downsamples with stride=2.
    Second convolution refines the features.
    """

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DirectionGateNet(nn.Module):
    """
    Lightweight DroNet-inspired CNN.

    Outputs:
      - heading in [-1, 1]
      - gate_measure

    gate_measure can represent:
        probability  -> probability of being within ~1m of the gate
        distance     -> estimated distance to the gate
        raw          -> raw regression value

    Input shape:
        [B, 1, 120, 160]
    """

    def __init__(self, output_mode="probability"):
        super().__init__()

        self.output_mode = output_mode

        # Stem (early downsampling)
        # [B,1,120,160] -> [B,8,60,80]
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),
        )

        # Block1: [B,8,60,80] -> [B,8,30,40]
        self.block1 = SmallBlock(8, 8, stride=2)

        # Block2: [B,8,30,40] -> [B,16,15,20]
        self.block2 = SmallBlock(8, 16, stride=2)

        # Block3: [B,16,15,20] -> [B,24,8,10]
        self.block3 = SmallBlock(16, 24, stride=2)

        # Pool only over height to preserve horizontal position
        # [B,24,8,10] -> [B,24,1,10]
        self.horiz_pool = nn.AdaptiveAvgPool2d((1, 10))

        # Fully connected head
        # 24 * 10 = 240 features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 10, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x):

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.horiz_pool(x)
        x = self.head(x)

        # Output 1: heading
        heading = torch.tanh(x[:, 0])

        # Output 2: gate measure
        raw_gate = x[:, 1]

        if self.output_mode == "probability":
            gate_measure = torch.sigmoid(raw_gate)

        elif self.output_mode == "distance":
            gate_measure = torch.relu(raw_gate)

        else:
            gate_measure = raw_gate

        return heading, gate_measure


if __name__ == "__main__":

    model = DirectionGateNet(output_mode="probability")

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    dummy = torch.randn(1, 1, 120, 160)

    heading, gate_measure = model(dummy)

    print(f"heading={heading.item():+.4f}, gate_measure={gate_measure.item():.4f}")