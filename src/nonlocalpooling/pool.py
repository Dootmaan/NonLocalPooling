import torch 
from einops.layers.torch import Rearrange

class PixelShuffle3d(torch.nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale_d=2, scale_h=2, scale_w=2):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale_d = scale_d
        self.scale_h=scale_h
        self.scale_w=scale_w

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        nOut = channels // (self.scale_d*self.scale_h*self.scale_w)

        out_depth = in_depth * self.scale_d
        out_height = in_height * self.scale_h
        out_width = in_width * self.scale_w

        input_view = x.contiguous().view(batch_size, nOut, self.scale_d, self.scale_h, self.scale_w, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class MixerBlock3d(torch.nn.Module):

    def __init__(self, in_size, in_channel, out_channel=None, scale=2):
        super(MixerBlock3d,self).__init__()
        patch_size=[2*scale,2*scale,2*scale]
        
        if not isinstance(in_size,list):
            isize=in_size
            in_size=[isize,isize]
        if out_channel is None:
            out_channel=in_channel
        # self.out_size=[in_size[0]//scale, in_size[1]//scale, in_size[2]//scale]
        self.num_token=in_size[0]//patch_size[0]*in_size[1]//patch_size[1]*in_size[2]//patch_size[2]
        self.token_dim=out_channel//2
        
        self.embedding = torch.nn.Conv3d(in_channel, self.token_dim, patch_size, patch_size)
        self.token_mix = torch.nn.Sequential(
            Rearrange('b c d h w -> b c (d h w)'),
            torch.nn.LayerNorm(self.num_token), 
            torch.nn.Linear(self.num_token, self.num_token//2),
            torch.nn.GELU(),
            torch.nn.Linear(self.num_token//2, self.num_token),
            Rearrange('b c (d h w) -> b c d h w', d=in_size[0]//patch_size[0],h=in_size[1]//patch_size[1],w=in_size[2]//patch_size[2])
        )
        self.pixelshuffle=PixelShuffle3d(2,2,2)
        self.channel_mix1 = torch.nn.Sequential(
            Rearrange('b c d h w -> b (d h w) c'),
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, out_channel*4),
            Rearrange('b (d h w) c -> b c d h w',d=in_size[0]//patch_size[0],h=in_size[1]//patch_size[1],w=in_size[2]//patch_size[2])
        )
        self.channel_mix2 = torch.nn.Sequential(
            Rearrange('b c d h w -> b (d h w) c'),
            torch.nn.LayerNorm(out_channel//2),
            torch.nn.Linear(out_channel//2, out_channel),
            Rearrange('b (d h w) c -> b c d h w',d=in_size[0]//scale,h=in_size[1]//scale,w=in_size[2]//scale)
        )
        
        # self.beta=torch.Tensor([1],requires_grad=True)

    def forward(self, x):
        ebd=self.embedding(x)
        tkm=self.token_mix(ebd)
        chm=self.channel_mix1(tkm)
        ps=self.pixelshuffle(chm)
        return self.channel_mix2(ps)

 
class MixerBlock2d(torch.nn.Module):

    def __init__(self, in_size, in_channel, out_channel=None, scale=2):
        super(MixerBlock2d,self).__init__()
        patch_size=[2*scale,2*scale]
        
        if not isinstance(in_size,list):
            isize=in_size
            in_size=[isize,isize]
        
        if out_channel is None:
            out_channel=in_channel
        # self.out_size=[in_size[0]//scale, in_size[1]//scale, in_size[2]//scale]
        self.num_token=in_size[0]//patch_size[0]*in_size[1]//patch_size[1]
        self.token_dim=out_channel//2
        
        self.embedding = torch.nn.Conv2d(in_channel, self.token_dim, patch_size, patch_size)
        self.token_mix = torch.nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            torch.nn.LayerNorm(self.num_token),
            torch.nn.Linear(self.num_token, self.num_token//2),
            torch.nn.GELU(),
            torch.nn.Linear(self.num_token//2, self.num_token),
            Rearrange('b c (h w) -> b c h w', h=in_size[0]//patch_size[0],w=in_size[1]//patch_size[1])
        )
        self.pixelshuffle=torch.nn.PixelShuffle(2)
        self.channel_mix1 = torch.nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, out_channel*2), 
            Rearrange('b (h w) c -> b c h w',h=in_size[0]//patch_size[0], w=in_size[1]//patch_size[1])
        )
        self.channel_mix2 = torch.nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            torch.nn.LayerNorm(out_channel//2),
            torch.nn.Linear(out_channel//2, out_channel), 
            Rearrange('b (h w) c -> b c h w',h=in_size[0]//scale, w=in_size[1]//scale)
        )
        
        # self.beta=torch.Tensor([1],requires_grad=True)

    def forward(self, x):
        ebd=self.embedding(x)
        tkm=self.token_mix(ebd)
        chm=self.channel_mix1(tkm)
        ps=self.pixelshuffle(chm)
        return self.channel_mix2(ps)


class NonLocalPool3d(torch.nn.Module):
    '''
    This class is a 3d version of nonlocalpooling. 
    '''
    def __init__(self, in_size, in_channel, out_channel=None, scale=2):
        super(NonLocalPool3d, self).__init__()
        self.mlp=MixerBlock3d(in_size, in_channel, out_channel, scale)
        self.max_pool=torch.nn.MaxPool3d(scale)
        self.alpha=torch.nn.Parameter(torch.FloatTensor(1))
        self.add=True if out_channel is None else False
       
    def forward(self,x):
        nonlocalpool=self.mlp(x)
        if self.add:
            maxpool=self.max_pool(x)
            return maxpool*(self.alpha).expand_as(maxpool)+nonlocalpool*(1-self.alpha).expand_as(nonlocalpool)
        else:
            return nonlocalpool



class NonLocalPool2d(torch.nn.Module):
    '''
    This class is a 2d version of nonlocalpooling.
    '''
    def __init__(self, in_size, in_channel, out_channel=None, scale=2):
        super(NonLocalPool2d, self).__init__() 
        self.mlp=MixerBlock2d(in_size, in_channel, out_channel, scale)
        self.max_pool=torch.nn.MaxPool2d(scale)
        self.alpha=torch.nn.Parameter(torch.FloatTensor(1))
        self.add=True if out_channel is None else False
       
    def forward(self,x):
        nonlocalpool=self.mlp(x)
        if self.add:
            maxpool=self.max_pool(x)
            return maxpool*(self.alpha).expand_as(maxpool)+nonlocalpool*(1-self.alpha).expand_as(nonlocalpool)
        else:
            return nonlocalpool
