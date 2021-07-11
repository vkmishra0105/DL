import torch
import torch.nn.functional as F
import numpy as py


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
          torch.nn.Conv2d(3, 32, kernel_size=7, padding = 2),
          torch.nn.BatchNorm2d(32),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(3,2,1),
          torch.nn.Conv2d(32, 64, kernel_size=3, padding = 1),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(3,2,1),
          torch.nn.Conv2d(64, 128, kernel_size=3, padding = 1),
          torch.nn.BatchNorm2d(128),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(3,2,1),
          torch.nn.Conv2d(128, 256, kernel_size=3, padding = 1),
          torch.nn.BatchNorm2d(256),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(3,2,1),
          torch.nn.Conv2d(256, 512, kernel_size=3, padding = 1),
          torch.nn.BatchNorm2d(512),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(1, 1, 0)
        )
        self.netlinear=torch.nn.Sequential(
          torch.nn.Linear(8192, 120),
          torch.nn.Dropout(0.05),
          torch.nn.Linear(120, 6)
          
        )
        self.downsample = None
        #if stride!=1
        #  self.downsample = torch.nn.Sequential(
        #    torch.nn.Conv2d()
        #  )

        #come back to this lol
        #raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        identity = x
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.netlinear(x)
        return x


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.net1 = torch.nn.Sequential(
          torch.nn.Conv2d(3, 32, kernel_size=3, padding = 2),
          torch.nn.BatchNorm2d(32),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(3,2,1))
        self.net2 = torch.nn.Sequential(
          torch.nn.Conv2d(32, 64, kernel_size=3, padding = 1),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(3,2,1),
        )
        self.net3 = torch.nn.Sequential(
          torch.nn.Conv2d(64, 128, kernel_size=3, padding = 1),
          torch.nn.BatchNorm2d(128),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(3,2,1),
        )
        self.net4 = torch.nn.Sequential(
          torch.nn.Conv2d(128, 256, kernel_size=3, padding = 1),
          torch.nn.BatchNorm2d(256),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(3,2,1),
        )
        self.fcn2 = torch.nn.Sequential(
          torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding = 1),
          torch.nn.BatchNorm2d(128),
          torch.nn.ReLU(),
        )
        self.fcn3 = torch.nn.Sequential(
          torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding = 1),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU()
        )
        self.fcn4 = torch.nn.Sequential(
          torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding = 1),
          torch.nn.BatchNorm2d(32),
          torch.nn.ReLU()
        )
        self.fcn5 = torch.nn.Sequential(
          torch.nn.ConvTranspose2d(32, 5, kernel_size=3, stride=2, padding = 1, dilation=1),
          torch.nn.BatchNorm2d(5),
          torch.nn.ReLU()
        )
        #raise NotImplementedError('FCN.__init__')

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
        #print("x"+ str(py.shape(x)))
        x2 = self.net2(x1)
        x3 = self.net3(x2)
        x4 = self.net4(x3)
        #x5 = self.net5(x4)
        x7 = self.fcn2(x4)
        #x6 = x6[:x.size(0), :, :x4.size(2), :x4.size(3)]
        #x7 = self.fcn2(torch.cat([x6, x4],0))
        x7 = x7[:x.size(0), :, :x3.size(2), :x3.size(3)]
        x8 = self.fcn3(torch.cat([x7, x3],0))
        x8 = x8[:x.size(0), :, :x2.size(2), :x2.size(3)]
        x9 = self.fcn4(torch.cat([x8, x2],0))
        x9 = x9[:x.size(0), :, :x1.size(2), :x1.size(3)]
        x10 = self.fcn5(torch.cat([x9, x1],0))
        #print("x10"+ str(py.shape(x10)))
       # x = self.fcn(x)
        xout = x10[:x.size(0), :, :size[2], :size[3]]
        return xout
#        raise NotImplementedError('FCN.forward')


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

