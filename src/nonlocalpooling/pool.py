import torch 
from einops.layers.torch import Rearrange
import warnings

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

class NonLocalPool2d(torch.nn.Module):
    def __init__(self, in_size, in_channel, out_channel=None, scale=2, squeeze=False):
        super(NonLocalPool2d, self).__init__()
        self.squeeze=squeeze

        if self.squeeze:
            patch_size = [2*scale, 2*scale]
        else:
            patch_size = [scale, scale]
        
        if not isinstance(in_size, list):
            isize = in_size
            in_size = [isize, isize]
        # print(in_size)

        if out_channel is None:
            out_channel = in_channel
        # self.out_size=[in_size[0]//scale, in_size[1]//scale, in_size[2]//scale]
        self.num_token = (in_size[0]//patch_size[0])*(in_size[1]//patch_size[1])
        self.token_dim = out_channel//2 if out_channel >= 128 else out_channel

        self.embedding = torch.nn.Conv2d(
            in_channel, self.token_dim, patch_size, patch_size)
        self.token_mix = torch.nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            torch.nn.LayerNorm(self.num_token),
            torch.nn.Linear(self.num_token, self.num_token),
            torch.nn.Tanh(),
            torch.nn.Linear(self.num_token, self.num_token),
            Rearrange('b c (h w) -> b c h w',
                      h=in_size[0]//patch_size[0], w=in_size[1]//patch_size[1])
        )
        self.pixelshuffle=torch.nn.PixelShuffle(2)
        self.maxpool=torch.nn.MaxPool2d(3,2,1)
        self.point_wise=torch.nn.Conv2d(in_channel, out_channel,1,1,0)
        self.norm=torch.nn.BatchNorm2d(out_channel)
        self.alpha=torch.nn.Parameter(torch.FloatTensor(1))
        torch.nn.init.constant_(self.alpha,0.1)

        if self.squeeze:
            out_channel=scale**2*out_channel

        # self.beta=torch.Tensor([1],requires_grad=True)
        self.channel_mix = torch.nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, out_channel),
            Rearrange('b (h w) c -> b c h w',
                      h=in_size[0]//patch_size[0], w=in_size[1]//patch_size[1])
        )
        # self.channel_mix2 = nn.Sequential(
        #     Rearrange('b c h w -> b (h w) c'),
        #     nn.LayerNorm(out_channel//2),
        #     nn.Linear(out_channel//2, out_channel),
        #     Rearrange('b (h w) c -> b c h w',h=in_size[0]//scale, w=in_size[1]//scale)
        # )

    def forward(self, x):
        mxp=self.maxpool(self.point_wise(x))
        ebd = self.embedding(x)
        tkm = self.token_mix(ebd)
        chm = self.channel_mix(tkm)
        if self.squeeze:
            chm=self.pixelshuffle(chm)
        return self.norm(self.alpha.expand_as(chm)*chm+(1-self.alpha).expand_as(mxp)*mxp)
        

class NonLocalPool3d(torch.nn.Module):
    def __init__(self, in_size, in_channel, out_channel=None, scale=2, squeeze=True):
        super(NonLocalPool3d, self).__init__()
        self.squeeze=squeeze

        if self.squeeze:
            patch_size = [2*scale, 2*scale, 2*scale]
        else:
            warnings.warn('Squeeze disabled may cause dramatic video memory usage for 3D NonLocal Pooling.', UserWarning)
            patch_size = [scale, scale, scale]
        
        if not isinstance(in_size, list):
            isize = in_size
            in_size = [isize, isize, isize]
        # print(in_size)

        if out_channel is None:
            out_channel = in_channel
        # self.out_size=[in_size[0]//scale, in_size[1]//scale, in_size[2]//scale]
        self.num_token=in_size[0]//patch_size[0]*in_size[1]//patch_size[1]*in_size[2]//patch_size[2]
        self.token_dim = out_channel//2 if out_channel >= 128 else out_channel

        self.embedding = torch.nn.Conv3d(in_channel, self.token_dim, patch_size, patch_size)

        self.token_mix = torch.nn.Sequential(
            Rearrange('b c d h w -> b c (d h w)'),
            torch.nn.LayerNorm(self.num_token), 
            torch.nn.Linear(self.num_token, self.num_token),
            torch.nn.Tanh(),
            torch.nn.Linear(self.num_token, self.num_token),
            Rearrange('b c (d h w) -> b c d h w', d=in_size[0]//patch_size[0],h=in_size[1]//patch_size[1],w=in_size[2]//patch_size[2])
        )
        self.pixelshuffle=PixelShuffle3d(2,2,2)
        self.maxpool=torch.nn.MaxPool3d(3,2,1)
        self.point_wise=torch.nn.Conv3d(in_channel, out_channel,1,1,0)
        self.norm=torch.nn.BatchNorm3d(out_channel)
        self.alpha=torch.nn.Parameter(torch.FloatTensor(1))
        torch.nn.init.constant_(self.alpha,0.1)

        if self.squeeze:
            out_channel=scale**3*out_channel

        # self.beta=torch.Tensor([1],requires_grad=True)
        self.channel_mix = torch.nn.Sequential(
            Rearrange('b c d h w -> b (d h w) c'),
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, out_channel),
            Rearrange('b (d h w) c -> b c d h w',d=in_size[0]//patch_size[0],h=in_size[1]//patch_size[1],w=in_size[2]//patch_size[2])
        )
        # self.channel_mix2 = nn.Sequential(
        #     Rearrange('b c h w -> b (h w) c'),
        #     nn.LayerNorm(out_channel//2),
        #     nn.Linear(out_channel//2, out_channel),
        #     Rearrange('b (h w) c -> b c h w',h=in_size[0]//scale, w=in_size[1]//scale)
        # )

    def forward(self, x):
        mxp=self.maxpool(self.point_wise(x))
        ebd = self.embedding(x)
        tkm = self.token_mix(ebd)
        chm = self.channel_mix(tkm)
        if self.squeeze:
            chm=self.pixelshuffle(chm)
        return self.norm(self.alpha.expand_as(chm)*chm+(1-self.alpha).expand_as(mxp)*mxp)