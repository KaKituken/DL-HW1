import torch
from torch import nn

class LeNet5(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            # nn.BatchNorm2d(6),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            # nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.Tanh()
        ])

        self.classifier_head = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            # nn.Dropout(),
            nn.Linear(in_features=84, out_features=num_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, return_dict=False):
        for module in self.conv_layers:
            x = module(x)
        
        x = x.reshape(x.shape[0], -1)
        logits = self.classifier_head(x)

        if return_dict:
            return {'logits': logits}
        else:
            return logits
        

if __name__ == '__main__':
    #sanity check
    lenet = LeNet5(num_class=10)
    input = torch.rand([2, 1, 32, 32]) # [b c w h]
    print('input:', input.shape)
    logit = lenet(input)
    print('logit:', logit.shape)