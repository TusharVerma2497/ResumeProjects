import torch.nn as nn 
import torch            
            
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze operation
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),  # Excitation
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).reshape(batch_size, channels)  # Squeeze: (batch, channels)
        y = self.fc(y).reshape(batch_size, channels, 1, 1)  # Excitation: (batch, channels, 1, 1)
        return x * y.expand_as(x)  # Reweight feature maps

    
    
    
class Atrous_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pad, dilation_rate, stride=1):
        pad = (dilation_rate * (kernel - 1)) // 2  # Calculate padding dynamically
        super(Atrous_Convolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=pad, stride=stride, dilation=dilation_rate, bias=True)

    def forward(self, x):
        return self.conv(x)
    


class ASSP_block(nn.Module):
    def __init__(self, in_channles, out_channles, stride=1, pad=0, kernel=3, dilationRates=[1,3,7,13]):
        super(ASSP_block, self).__init__()
        self.conv = nn.ModuleList()
        for i in dilationRates:
            if(i==1):
                self.conv.append(Atrous_Convolution(in_channles, out_channles, kernel=1, pad=0, dilation_rate=1))
            else:
                self.conv.append(Atrous_Convolution(in_channles, out_channles, kernel=kernel, pad=i, dilation_rate=i))
        self.final_conv=Atrous_Convolution(out_channles*len(dilationRates), out_channles, 1, pad, dilation_rate=1)

    def forward(self, x):
        out = []
        for i in self.conv:
            out.append(i(x))
        # concatination of all features
        concat = torch.cat(tuple(i for i in out),dim=1)
        x_final_conv = self.final_conv(concat)
        return x_final_conv

    
    
    
    
    
# Multiheaded attention with SE operator for block importance            
            
class multiHeaded_ASSP_SE_block(nn.Module):
    def __init__(self, in_channles, out_channles, stride=1, pad=0, kernel=3, dilationLayers=[[1,3,7],[1,5,13,15],[1,3,13,19,27,29],[1,7,15,27,37,39]], SEOperator=False):
        super(multiHeaded_ASSP_SE_block, self).__init__()
        
        assert len(dilationLayers)>0
        
        # if only one list of list then no multiheaded ASSP
        self.SEOperator=SEOperator
        self.multiHeaded = True
        if(len(dilationLayers)==1):
            self.multiHeaded = False
            self.SEOperator = False
            dilationLayers = dilationLayers[0]
        self.dilationLayers = dilationLayers
        
        # multiheaded module
        if(self.multiHeaded):
            self.heads = nn.ModuleList()
            dilationRates = set()
            for i in dilationLayers:
                self.heads.append(Atrous_Convolution(out_channles*len(i), out_channles, 1, pad, dilation_rate=1))
                dilationRates=dilationRates.union(set(i))
        else:
            dilationRates = set(dilationLayers)
            
        
        print(dilationRates)    
        self.dilatedConv = nn.ModuleDict()
        for i in dilationRates:
            if(i==1):
                self.dilatedConv['1']=Atrous_Convolution(in_channles, out_channles, kernel=1, pad=0, dilation_rate=1)
            else:
                self.dilatedConv[str(i)]=Atrous_Convolution(in_channles, out_channles, kernel=kernel, pad=i, dilation_rate=i)
        
        #putting a SE block to rreweight the important channels of concatinated channels form the heads
        if(self.SEOperator):
            self.SEBlock = SEBlock(out_channles*len(dilationLayers))
        
        self.final_conv=Atrous_Convolution(out_channles*len(dilationLayers), out_channles, 1, pad, dilation_rate=1)
        
        
        
    def forward(self, x):
        out = dict()
        for i, j in self.dilatedConv.items():
            out[i] = j(x)
        
        if(self.multiHeaded):
            heads_out = []
            for i,j in zip(self.dilationLayers, self.heads):
                concat = torch.cat(tuple(out[str(k)] for k in i),dim=1)
                heads_out.append(j(concat))

            concat = torch.cat(tuple(i for i in heads_out),dim=1)
            if(self.SEOperator):
                concat = self.SEBlock(concat)
        else:
            concat = torch.cat(tuple(out[str(k)] for k in self.dilatedConv.keys()),dim=1)
        final_out = self.final_conv(concat)
        
        return final_out