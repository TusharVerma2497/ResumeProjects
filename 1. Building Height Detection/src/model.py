from src.assp import ASSP_block as ASSP, multiHeaded_ASSP_SE_block
import torch.nn as nn
import torch
# from src.HartleySpectralPooling import SpectralPoolNd


# Custom Height Filter for outputting height always grater then 2m
class heightFilter(nn.Module):
    def __init__(self, shift=2, alpha=30):
        super().__init__()
        self.shift = shift
        self.alpha=alpha

    def forward(self, x):
        sigmoidFilter = torch.sigmoid(self.alpha*(x - self.shift))
        reluFilter = torch.relu(x- self.shift) + self.shift
        
        return reluFilter*sigmoidFilter


#Define encoder and decoder components for model definition
def pixelShuffle_upSample(upscale_factor):
    return nn.PixelShuffle(upscale_factor)


def depthwise_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),  # Depthwise convolution
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),  # Pointwise convolution
    )

def basic_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def transposed_upConv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
    )



# Define encoder decoder blocks
def encoder_block(in_channels, out_channels, dilationLayers=[1,3,7,13], SEOperator=False, ASSPKernel=3,):
    return nn.Sequential(
        basic_conv(in_channels, out_channels),
        depthwise_conv(out_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        multiHeaded_ASSP_SE_block(out_channels, out_channels, stride=1, kernel=ASSPKernel, dilationLayers=dilationLayers, SEOperator=SEOperator),
#         depthwise_conv(out_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )



def decoder_block(in_channels, out_channels):
    return nn.Sequential(
        basic_conv(in_channels, out_channels),
        depthwise_conv(out_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        transposed_upConv(out_channels, out_channels),
#         depthwise_conv(out_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )




def decoderPixelShuffler_block(in_channels, upsample_factor):
    return nn.Sequential(
        depthwise_conv(in_channels, in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        pixelShuffle_upSample(upsample_factor),
    )



class encoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        pooling = params["pooling"]
        layers = params["unet_filters"]
        layers.insert(0, params["in_channels"])
        SEOperator = params['ASSP']['SEOperator']
        ASSPKernel = params['ASSP']['kernel_size']
        self.dilationLayers = params['ASSP']['dilation_layers']
        
        self.blocks = nn.ModuleList()
        for i in range(1,len(layers),1):
            self.blocks.append(encoder_block(layers[i-1], layers[i], self.dilationLayers[i-1], SEOperator, ASSPKernel))
        
        if(pooling == 'avg'):
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        
    def forward(self, x):
        residuals = []
        
        for i, block in enumerate(self.blocks):
            out = block(x)
            if(i>0): #skipping pooling for the first block
                out = self.pool(out)
            residuals.append(out)
            x = out
        
        return residuals
    


class decoder(nn.Module):
    def __init__(self, params, layers):
        super().__init__()
        
        self.applyAttention = params['self_attention']
        produce = params['produce']
        self.finalActivation = nn.Sigmoid() if('sigmoid' in params['final_activation']) else nn.ReLU()
        self.activationWeight = 30 if('steep' in params['final_activation']) else 1
        out_channels = len(produce)

        #list of decoder blocks
        self.blocks = nn.ModuleList()
        #attention blocks
        if(self.applyAttention):
            self.attentionBlocks = nn.ModuleList()
            self.layerNorms = nn.ModuleList()
            self.attentionWeight = nn.ParameterList() 
        
        
        for i in range(1,len(layers)-1,1):
            self.blocks.append(decoder_block(2*layers[i-1], layers[i]))
            if(self.applyAttention):
                self.attentionBlocks.append(nn.MultiheadAttention(embed_dim=layers[i], num_heads=4, batch_first=True))
                self.layerNorms.append(nn.LayerNorm(layers[i]))
                self.attentionWeight.append(nn.Parameter(torch.zeros(1)))
#         self.blocks = self.blocks.to('cuda')
        
#         final processing and output layer
        self.prePixelShuffler = nn.Sequential(
            depthwise_conv(layers[-2]*2, layers[-1]),
            basic_conv(layers[-1], layers[-1]),
            nn.BatchNorm2d(layers[-1]),
            nn.ReLU(inplace=True))
        #last layer as pixelShuffler for crisp output
        self.pixelShuffler = decoderPixelShuffler_block(layers[-1], 2)
        self.postPixelShuffler = depthwise_conv((layers[-1]//4 + layers[-1]), out_channels)
#         self.postPixelShuffler = basic_conv((layers[-1]//4), out_channels)
        
        
    def use_attention(self, x, index):
        b,c,h,w = x.shape
        x_att = x.reshape(b,c,h*w).transpose(1,2)
        x_att = self.layerNorms[index](x_att)
        att_out, att_map = self.attentionBlocks[index](x_att, x_att, x_att)
        att_out = self.layerNorms[index](att_out)
        return att_out.transpose(1,2).reshape(b,c,h,w), att_map    
        
    def forward(self, x, residuals):
        
        for i, block in enumerate(self.blocks):
            x = torch.cat([x, residuals[-(i+1)]], dim=1)
#           processing using the decoder block and upsampling
            out = block(x)
#           applying self-attention
            if(self.applyAttention):
                attention = self.use_attention(out, i)
                out = self.attentionWeight[i]*attention[0] + out
#           setting to x so that we pass this output to the next decoder block
            x = out
    
        x = torch.concat([x, residuals[1]], dim=1)
        x = self.prePixelShuffler(x)
        x = self.pixelShuffler(x)
        x = torch.concat([x, residuals[0]], dim=1)
        x = self.postPixelShuffler(x)
        x = self.finalActivation(self.activationWeight*x)
        return x
    
    
class UNet(nn.Module):
    def __init__(self, encoderParams, decoderParams):
        super().__init__()
        
        #ENCOER
        self.encoder = encoder(encoderParams)
        #BOTTLENECK
        bottleneckChannel = encoderParams['unet_filters'][-1]
        self.bottleneck = nn.Sequential(
            depthwise_conv(bottleneckChannel, bottleneckChannel),
            nn.BatchNorm2d(bottleneckChannel),
            nn.GELU(),
        )
        #DECODER
        self.applyDecoderCrossAttention = decoderParams["cross_attention"]
        layers = encoderParams["unet_filters"][::-1][:-1]
        self.decoder = nn.ModuleDict({i['tag']:decoder(i,layers) for i in decoderParams['tracks']})
        self.predFormat = {i['tag']: {j:None for j in i['produce']} for i in decoderParams['tracks']}
        #Height Filter
        self.heightFilter = heightFilter()

        
    def forward(self, x):
        #copying the format that the output needs to be in
        preds = self.predFormat.copy()
        #passing through the encoder
        residuals=self.encoder(x)
        #passing through the bottleneck
        x = self.bottleneck(residuals[-1])
        #passing through decoder
        for tag, decoder in self.decoder.items():
            temp = decoder(x,residuals)
            c=0
            for j in preds[tag].keys():
                if(j =='dsm'):
                    #Applying height filter so that predicted heights is always >2 
                    preds[tag][j] = self.heightFilter(temp[:, c:c+1, :, :])
                else:
                    preds[tag][j] = temp[:, c:c+1, :, :]
                c+=1
                
        return preds