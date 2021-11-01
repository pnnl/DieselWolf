#Model definitions for the Hybrid multitask model and a basic ResNet 

import torch
import torch.nn as nn
from reformer_pytorch import Reformer

####Real Hybrid Model####

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias): 
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, activation=True):
        super(ConvUpsampleBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.activation = activation
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        if self.activation:
            out = self.relu(out)
        return out

class Hybrid(nn.Module):
    def __init__(self, input_samples: int, n_classes: int, debug = False):
        super().__init__()
        
        self.debug = debug
        
        self.encoder = nn.Sequential(
        ConvBlock(2, 128, 13, 6, False),
        nn.MaxPool1d(2),
        ConvBlock(128, 256, 13, 6, False),
        nn.MaxPool1d(2),    
        ConvBlock(256, 256, 13, 6, False),
        ConvBlock(256, 256, 13, 6, False),)
        
        self.reformer = Reformer(dim = 256, depth = 2,  heads = 8, lsh_dropout = 0.1, causal = True, bucket_size = 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*int(input_samples/4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, n_classes),)

        self.decoder = nn.Sequential(
        ConvUpsampleBlock(256, 128, 13, 6, False),
        nn.Upsample(scale_factor=2),
        ConvUpsampleBlock(128, 128, 13, 6, True),
        nn.Upsample(scale_factor=2),
        ConvUpsampleBlock(128, 64, 13, 6, True),
        ConvUpsampleBlock(64, 2, 13, 6, True,activation=False),)
        
    def forward(self, input_):
        z = self.encoder(input_)
        
        if self.debug: print(z.shape)
        
        recon = self.decoder(z)
        
        if self.debug: print(recon.shape)
        
        z = self.reformer(z.permute(0,2,1))
        y = self.classifier(z.permute(0,2,1))
        return y,recon
    
#ResNet with 13/6 kernel/padding

class BasicResidualUnit(nn.Module):
    """Basic residual unit for building ResNet with 1D input, from O'shea et. al (2018)"""
    def __init__(self, in_channels, out_channels, downsample=None):
        super(BasicResidualUnit, self).__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=13, padding=6)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=13, padding=6)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        
        input_tensor_copy = input_tensor
        
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.batch_norm1(output_tensor)
        output_tensor = self.relu1(output_tensor)
        
        output_tensor = self.conv2(output_tensor)
        output_tensor = self.batch_norm2(output_tensor)
        
        if self.downsample is not None:
            self.downsample(output_tensor)
        
        output_tensor += input_tensor_copy
        #output_tensor = self.relu2(output_tensor)
        return output_tensor
    
class BasicResidualStack(nn.Module):
    """Basic residual stack, made from 1x1 conv, ResUnit, ResUnit, MaxPool from O'shea et. al (2018)"""
    def __init__(self, in_channels, out_channels, downsample=None):
        super(BasicResidualStack, self).__init__()
        self.downsample = downsample
        
        self.conv1x1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,)
        self.basic_unit_1 = BasicResidualUnit(in_channels=out_channels, out_channels=out_channels)
        self.basic_unit_2 = BasicResidualUnit(in_channels=out_channels, out_channels=out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, input_tensor):
        
        output_tensor = self.conv1x1(input_tensor)
        output_tensor = self.basic_unit_1(output_tensor)
        output_tensor = self.basic_unit_2(output_tensor)
        output_tensor = self.maxpool(output_tensor)
        
        return output_tensor

class Basic1DModel(nn.Module):
    """Create a pytorch copy of the 1D model used in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8267032"""
    def __init__(self, in_size, in_channels, block_sizes, num_classes):
        super(Basic1DModel, self).__init__()
        self.block_sizes = block_sizes
        
        self.in_size = in_size
        num_features = in_size[1]
        
        self.basic_stack_1 = BasicResidualStack(in_channels, block_sizes[0], )
        num_features /= 2
            
        self.basic_stack_2 = BasicResidualStack(block_sizes[0], block_sizes[1], )
        num_features /= 2
        
        self.basic_stack_3 = BasicResidualStack(block_sizes[1], block_sizes[2], )
        num_features /= 2
        
        if len(self.block_sizes) >= 4:
            self.basic_stack_4 = BasicResidualStack(block_sizes[2], block_sizes[3], )
            num_features /= 2
        
        if len(self.block_sizes) >= 5:
            self.basic_stack_5 = BasicResidualStack(block_sizes[3], block_sizes[4], )
            num_features /= 2
            
        if len(self.block_sizes) >= 6:
            self.basic_stack_6 = BasicResidualStack(block_sizes[4], block_sizes[5], )
            num_features /= 2
                
        self.fc_1 = nn.Linear(in_features=int(block_sizes[-1] * num_features), out_features=128, )
        self.selu_1 = nn.SELU(inplace=True)
        self.a_dropout_1 = nn.AlphaDropout(.1)
        
        self.fc_2 = nn.Linear(128, 128)
        self.selu_2 = nn.SELU(inplace=True)
        self.a_dropout_2 = nn.AlphaDropout(.1)
        
        self.fc_last = nn.Linear(128, num_classes)       

    def forward(self, input_tensor):
        
        output_tensor = self.basic_stack_1(input_tensor)        
        
        #print("Here", output_tensor.shape)
        
        output_tensor = self.basic_stack_2(output_tensor)        
        output_tensor = self.basic_stack_3(output_tensor)
        if len(self.block_sizes) >= 4: output_tensor = self.basic_stack_4(output_tensor)
        if len(self.block_sizes) >= 5: output_tensor = self.basic_stack_5(output_tensor)
        if len(self.block_sizes) >= 6: output_tensor = self.basic_stack_6(output_tensor)
            
        output_tensor = self.fc_1(torch.flatten(output_tensor, 1))
        output_tensor = self.selu_1(output_tensor)
        output_tensor = self.a_dropout_1(output_tensor)
        
        output_tensor = self.fc_2(output_tensor)
        output_tensor = self.selu_2(output_tensor)
        output_tensor = self.a_dropout_2(output_tensor)
        
        output_tensor = self.fc_last(output_tensor)  
        
        return output_tensor