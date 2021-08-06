import torch
import torch.nn as nn
from torchvision.models import resnet18


class SAT(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.encoder1 = nn.Sequential(*list(resnet18(pretrained=self.pretrained).children())[:-1])
        self.encoder2 = nn.Sequential(*list(resnet18(pretrained=self.pretrained).children())[:-1])

        self.fuse_layer = nn.Linear(in_features=1024, out_features=512, bias=True)

        self.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=12, bias=True))

        self.discriminator = nn.Sequential(nn.Linear(in_features=1024, out_features=256, bias=True),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                           nn.Linear(in_features=256, out_features=2, bias=True))

        # self.discriminator = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
        #                                                   track_running_stats=True),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
        #                                                   track_running_stats=True),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
        #                                                   track_running_stats=True),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
        #                                                   track_running_stats=True),
        #                                    nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        #
        # self.dis_cls = nn.Linear(in_features=512, out_features=2, bias=True)

        # for i in self.encoder1.parameters():
        #     i.requires_grad = False
        # for i in self.encoder2.parameters():
        #     i.requires_grad = False
        for i in self.discriminator.parameters():
            i.requires_grad = False
        # for i in self.classifier.parameters():
        #     i.requires_grad = False

    def forward(self, x):
        channel = int(x.shape[1] / 2)
        x1 = x[:, 0:channel, :, :]
        x2 = x[:, channel:, :, :]

        # x_dis = x1 + x2
        # x_dis = torch.flatten(self.discriminator(x_dis), 1)

        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        # x_sum = x1 + x2
        x_sum = self.fuse_layer(torch.cat((x1, x2), dim=-1))
        x = self.classifier(x_sum)
        x_domain = self.discriminator(torch.cat((x1, x2), dim=-1))

        return x, x_domain
