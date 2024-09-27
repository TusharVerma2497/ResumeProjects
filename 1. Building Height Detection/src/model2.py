from src.assp import ASSP_block as ASSP, multiHeaded_ASSP_SE_block
import torch.nn as nn
import torch
from src.helper import topological_sort
# from src.HartleySpectralPooling import SpectralPoolNd

class illigalConfiguration(Exception):
    pass



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
        
#         self.returnResiduals = params['return_residuals']
        self.crossAttentionParams = params['cross_attention']
        self.crossAttention = True if len(params['cross_attention'])>0 else False
        self.selfAttention = params['self_attention']
        produce = params['produce']
        self.finalActivation = nn.Sigmoid() if('sigmoid' in params['final_activation']) else nn.ReLU()
        self.activationWeight = 30 if('steep' in params['final_activation']) else 1
        out_channels = len(produce)

        #list of decoder blocks
        self.blocks = nn.ModuleList()
        #attention blocks
        if(self.selfAttention):
            self.attentionBlocks = nn.ModuleList()
            self.layerNorms = nn.ModuleList()
            self.attentionWeight = nn.ParameterList()
        if(self.crossAttention):
            self.cross_attentionBlocks = nn.ModuleList()
            self.cross_layerNorms = nn.ModuleList()
            self.cross_attentionWeight = nn.ParameterList() 
            for i in self.crossAttentionParams:
                self.cross_attentionBlocks.append(nn.ModuleList())
                self.cross_layerNorms.append(nn.ModuleList())
                self.cross_attentionWeight.append(nn.ParameterList())
        
        
        for i in range(1,len(layers)-1,1):
            self.blocks.append(decoder_block(2*layers[i-1], layers[i]))
            if(self.selfAttention):
                self.attentionBlocks.append(nn.MultiheadAttention(embed_dim=layers[i], num_heads=4, batch_first=True))
                self.layerNorms.append(nn.LayerNorm(layers[i]))
                self.attentionWeight.append(nn.Parameter(torch.full((1,), 0.2)))
#                 self.attentionWeight.append(nn.Parameter(torch.zeros(1)))
            if(self.crossAttention):
                for j in range(len(self.crossAttentionParams)):
                    self.cross_attentionBlocks[j].append(nn.MultiheadAttention(embed_dim=layers[i], num_heads=4, batch_first=True))
                    self.cross_layerNorms[j].append(nn.LayerNorm(layers[i]))
#                     self.cross_attentionWeight[j].append(nn.Parameter(torch.zeros(1)))
                    self.cross_attentionWeight[j].append(nn.Parameter(torch.full((1,), 0.2)))
        
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
        
        
    def apply_selfAttention(self, x, index):
        b,c,h,w = x.shape
        x_att = x.reshape(b,c,h*w).transpose(1,2)
        x_att = self.layerNorms[index](x_att)
        att_out, att_map = self.attentionBlocks[index](x_att, x_att, x_att)
        att_out = self.layerNorms[index](att_out)
        return att_out.transpose(1,2).reshape(b,c,h,w), att_map    
    
    def apply_crossAttention(self, q, k, v, moduleIndex, layerIndex):
        b,c,h,w = q.shape
        q_att = q.reshape(b,c,h*w).transpose(1,2)
        k_att = k.reshape(b,c,h*w).transpose(1,2)
        v_att = v.reshape(b,c,h*w).transpose(1,2)
        q_att = self.cross_layerNorms[moduleIndex][layerIndex](q_att)
        k_att = self.cross_layerNorms[moduleIndex][layerIndex](k_att)
        v_att = self.cross_layerNorms[moduleIndex][layerIndex](v_att)
        att_out, att_map = self.cross_attentionBlocks[moduleIndex][layerIndex](q_att, k_att, v_att)
        att_out = self.cross_layerNorms[moduleIndex][layerIndex](att_out)
        return att_out.transpose(1,2).reshape(b,c,h,w), att_map    
    
    def forward(self, x, encoderResiduals, decoderResiduals=None):
        
        if(decoderResiduals==None and self.crossAttention):
            raise illigalConfiguration
            
        ret = []    
        for i, block in enumerate(self.blocks):
            x = torch.cat([x, encoderResiduals[-(i+1)]], dim=1)
#           processing using the decoder block and upsampling
            out = block(x)
#           applying self-attention
            if(self.selfAttention):
                attention = self.apply_selfAttention(out, i)
                out = self.attentionWeight[i]*attention[0] + out
            if(self.crossAttention):
                for j in range(len(self.crossAttentionParams)):
                    query = out
                    key = decoderResiduals[self.crossAttentionParams[j]['key']][i]
                    value = decoderResiduals[self.crossAttentionParams[j]['value']][i]
                    attention = self.apply_crossAttention(query, key, value, j, i)
                    out = self.cross_attentionWeight[j][i]*attention[0] + out
            ret.append(out)
#           setting to x so that we pass this output to the next decoder block
            x = out
    
        x = torch.concat([x, encoderResiduals[1]], dim=1)
        x = self.prePixelShuffler(x)
        x = self.pixelShuffler(x)
        ret.append(x)
        x = torch.concat([x, encoderResiduals[0]], dim=1)
        x = self.postPixelShuffler(x)
        ret.append(x)
        x = self.finalActivation(self.activationWeight*x)
       
        return x, ret
    
    
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
                #Determining the order of the decoder execution in case of cross attention
        self.crossAttentionParams = []
        for i in decoderParams["tracks"]:
            self.crossAttentionParams+=i['cross_attention']
            
        decoderTuples = set() #This will contain tuples of decoder tag (a,b) where 'a' decoder execues before 'b' 
        for i in self.crossAttentionParams:
            decoderTuples.add((i['key'], i['query']))
            decoderTuples.add((i['value'], i['query']))
        self.decoderOrder = topological_sort(list(decoderTuples))
        
        layers = encoderParams["unet_filters"][::-1][:-1]
                #Initializing decoders
        self.decoder = nn.ModuleDict({i['tag']:decoder(i,layers) for i in decoderParams['tracks']})
        
                #If no decoder with cross attention then execute in any order
        if(len(self.decoderOrder)==0):
            self.decoderOrder = list(self.decoder.keys())
            
                #This the format in which model's output is expected of
        self.predFormat = {i['tag']: {j:None for j in i['produce']} for i in decoderParams['tracks']}
                #Height filter
        self.heightFilter = heightFilter()

        
    def forward(self, x):
        decoderResiduals = dict()
        #copying the format that the output needs to be in
        preds = self.predFormat.copy()
        #passing through the encoder
        encoderResiduals=self.encoder(x)
        #passing through the bottleneck
        x = self.bottleneck(encoderResiduals[-1])
        #passing through decoder
        for tag in self.decoderOrder:
            temp, decoderResiduals[tag] = self.decoder[tag](x,encoderResiduals, decoderResiduals)
            c=0
            for j in preds[tag].keys():
                if(j =='dsm'):
                    #Applying height filter so that predicted heights is always >2 
#                     preds[tag][j] = self.heightFilter(temp[:, c:c+1, :, :])
                    preds[tag][j] = temp[:, c:c+1, :, :]
                else:
                    preds[tag][j] = temp[:, c:c+1, :, :]
                c+=1
                
        return preds