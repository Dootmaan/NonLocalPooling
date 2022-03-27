import torch 
from einops.layers.torch import Rearrange

class PixelShuffle3d(torch.nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale_d=8, scale_h=8, scale_w=8):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale_d = scale_d
        self.scale_h=scale_h
        self.scale_w=scale_w

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // (self.scale_d*self.scale_h*self.scale_w)

        out_depth = in_depth * self.scale_d
        out_height = in_height * self.scale_h
        out_width = in_width * self.scale_w

        input_view = input.contiguous().view(batch_size, nOut, self.scale_d, self.scale_h, self.scale_w, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class MixerBlock(torch.nn.Module):

    def __init__(self, in_channel, in_size, scale=2, patch_size=[4,6,6]):
        super(MixerBlock,self).__init__()
        # self.out_size=[in_size[0]//scale, in_size[1]//scale, in_size[2]//scale]
        self.num_token=in_size[0]//patch_size[0]*in_size[1]//patch_size[1]*in_size[2]//patch_size[2]
        self.token_dim=in_channel//16*(patch_size[0]//scale*patch_size[1]//scale*patch_size[2]//scale)
        
        self.embedding = torch.nn.Conv3d(in_channel, self.token_dim, patch_size, patch_size)
        self.token_mix = torch.nn.Sequential(
            Rearrange('b c d h w -> b c (d h w)'),
            torch.nn.LayerNorm(self.num_token), 
            torch.nn.Linear(self.num_token, self.num_token//2),
            torch.nn.GELU(),
            torch.nn.Linear(self.num_token//2, self.num_token),
            Rearrange('b c (d h w) -> b c d h w', d=in_size[0]//patch_size[0],h=in_size[1]//patch_size[1],w=in_size[2]//patch_size[2])
        )
        self.pixelshuffle=PixelShuffle3d(patch_size[0]//scale,patch_size[1]//scale,patch_size[2]//scale)
        self.channel_mix = torch.nn.Sequential(
            Rearrange('b c d h w -> b (d h w) c'),
            torch.nn.LayerNorm(self.token_dim//(patch_size[0]//scale*patch_size[1]//scale*patch_size[2]//scale)),
            torch.nn.Linear(self.token_dim//(patch_size[0]//scale*patch_size[1]//scale*patch_size[2]//scale), in_channel),
            Rearrange('b (d h w) c -> b c d h w',d=in_size[0]//scale, h=in_size[1]//scale, w=in_size[2]//scale)
        )
        
        # self.beta=torch.Tensor([1],requires_grad=True)

    def forward(self, x):
        ebd=self.embedding(x)
        tkm=self.token_mix(ebd)
        rev_tkm=self.pixelshuffle(tkm)
        return self.channel_mix(rev_tkm)


class NonLocalPooling(torch.nn.Module):
    def __init__(self, in_channel, in_size, scale=2, patch_size=[4,6,6]):
        super().__init__() 
        self.mlp=MixerBlock(in_channel, in_size, scale, patch_size)
        self.max_pool=torch.nn.MaxPool3d(scale)
        self.alpha=torch.tensor(1.0,requires_grad=True).cuda()
       
    def forward(self,x):
        maxpool=self.max_pool(x)
        nonlocalpool=self.mlp(x)
        return maxpool*(self.alpha).expand_as(maxpool)+nonlocalpool*(1-self.alpha).expand_as(nonlocalpool)
