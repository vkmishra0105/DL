import torch
import torch.nn.functional as F
import numpy as py


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                stride=stride, bias=False),
                #           torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),

                #           torch.nn.BatchNorm2d(n_output),
                #
                #          torch.nn.BatchNorm2d(n_output),
                #           torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)
            )

        def forward(self, x):
            return (self.net(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=6, kernel_size=3):
        super().__init__()
        self.input_mean = torch.Tensor([0.1, 0.2, 0.4])
        self.input_std = torch.Tensor([0.15, 0.16, 0.17])

        L = []
        c = 3
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)

    def forward(self, x):
        z = self.network(
            (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device))
        return self.classifier(z.mean(dim=[2, 3]))


class FCN(torch.nn.Module):
    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.net1 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=3, padding=2),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, 2, 1))
            self.net2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, 2, 1),
            )
            self.net3 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, 2, 1),
            )
            self.net4 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, 2, 1),
            )
            self.net5 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(1, 1, 0)
            )
            self.fcn1 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU()
            )
            self.fcn2 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
            )
            self.fcn3 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU()
            )
            self.fcn4 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU()
            )
            self.fcn5 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(32, 5, kernel_size=3, stride=2, padding=1, dilation=1),
                torch.nn.BatchNorm2d(5),
                torch.nn.ReLU()
            )
            # raise NotImplementedError('FCN.__init__')

        def forward(self, x):
            """
            Your code here
            @x: torch.Tensor((B,3,H,W))
            @return: torch.Tensor((B,6,H,W))
            Hint: Apply input normalization inside the network, to make sure it is applied in the grader
            Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
                  if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
                  convolution
            """
            size = py.shape(x)
            x1 = self.net1(x)
            # print("x"+ str(py.shape(x)))
            x2 = self.net2(x1)
            x3 = self.net3(x2)
            x4 = self.net4(x3)
            x5 = self.net5(x4)
            x6 = self.fcn1(x5)
            x6 = x6[:x.size(0), :, :x4.size(2), :x4.size(3)]
            x7 = self.fcn2(torch.cat([x6, x4], 0))
            x7 = x7[:x.size(0), :, :x3.size(2), :x3.size(3)]
            x8 = self.fcn3(torch.cat([x7, x3], 0))
            x8 = x8[:x.size(0), :, :x2.size(2), :x2.size(3)]
            x9 = self.fcn4(torch.cat([x8, x2], 0))
            x9 = x9[:x.size(0), :, :x1.size(2), :x1.size(3)]
            x10 = self.fcn5(torch.cat([x9, x1], 0))
            # print("x10"+ str(py.shape(x10)))
            # x = self.fcn(x)
            xout = x10[:x.size(0), :, :size[2], :size[3]]
            return xout
            raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r

