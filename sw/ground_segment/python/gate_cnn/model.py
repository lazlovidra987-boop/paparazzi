import torch
import torch.nn as nn

class AddYChannel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, _, h, w = x.size()
        # Create a gradient from -1.0 (Top/Left) to 1.0 (Bottom/Right)
        y_coords = torch.linspace(-1.0, 1.0, steps=h, device=x.device)
        # Reshape
        y_coords = y_coords.view(1, 1, h, 1).expand(batch_size, 1, h, w)
        # Combine
        return torch.cat([x, y_coords], dim=1)

class GateNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(GateNet, self).__init__()
        
        self.add_coords = AddYChannel()
        
        self.features = nn.Sequential(
        
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(

            nn.Linear(64, 4) 
        )

    def forward(self, x):
        x = self.add_coords(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
