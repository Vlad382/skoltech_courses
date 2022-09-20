import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm

device_glob = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        # print('embed_features, embed_features')
        self.embed_features = embed_features
        # self.mat_gamma = torch.randn(self.embed_features, self.num_features)
        # self.mat_bias = torch.randn(self.embed_features, self.num_features)

        self.base_bn = nn.BatchNorm2d(self.num_features, affine=False, device=device_glob)
        self.mapping_g = nn.Linear(self.embed_features, self.num_features, bias=False, device=device_glob)
        self.mapping_b = nn.Linear(self.embed_features, self.num_features, bias=False, device=device_glob)

    def forward(self, inputs, embeds):
        # print('inputs.shape', inputs.shape)
        # output_features = int(inputs.shape[1] * inputs.shape[2] * inputs.shape[3])

        # print('inputs.shape', inputs.shape)
        # print('embeds.shape', embeds.shape)
        # print('self.num_features', self.num_features)
        # print('self.embed_features', self.embed_features)
        # print()

        gamma = self.mapping_g(embeds)
        bias =  self.mapping_b(embeds)

        # gamma = gamma.reshape((inputs.shape))
        # bias = bias.reshape((inputs.shape))

        # gamma = torch.matmul(embeds, self.mat_gamma)
        # bias = torch.matmul(embeds, self.mat_bias)

        # print('gamma, inputs', gamma.shape, inputs.shape)
        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = self.base_bn(inputs)

        return outputs * gamma[..., None, None] + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_channels = embed_channels
        self.upsample = upsample
        self.downsample = downsample

        if batchnorm:
            self.pre_activation_res_block = nn.ModuleList([
                AdaptiveBatchNorm(in_channels, embed_channels),
                nn.ReLU(),
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, device=device_glob)),
                AdaptiveBatchNorm(out_channels, embed_channels),
                nn.ReLU(),
                spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, device=device_glob)),
            ])

        else:
            self.pre_activation_res_block = nn.ModuleList([
                nn.ReLU(),
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, device=device_glob)),
                nn.ReLU(),
                spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, device=device_glob)),
            ])

        self.up = nn.Upsample(scale_factor=(2, 2), mode='nearest')
        self.skip = spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, device=device_glob))
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.pre_activation_res_block = nn.Sequential(
        #     self.batchnorm_layer1,
        #     nn.ReLU(),
        #     nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
        #     self.batchnorm_layer2,
        #     nn.RelU(),
        #     nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
        # )

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        x1 = inputs.clone()
        if self.upsample:
            # print(x1.shape)
            x1 = self.up(x1)
            # print(x1.shape)
            
        x2 = x1.clone()
        for layer in self.pre_activation_res_block:
            if (embeds != None) & ('AdaptiveBatchNorm' in str(layer)):
                x2 = layer(x2, embeds)
            else:
                x2 = layer(x2)
        
        if self.in_channels != self.out_channels:
            x1 = self.skip(x1)
        
        outputs = x1 + x2

        if self.downsample:
            outputs = self.down(outputs)
        
        return outputs


class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.output_size = 4 * 2**num_blocks

        self.min_channels = min_channels
        self.max_channels = max_channels
        self.noise_channels = noise_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.use_class_condition = use_class_condition

        start = torch.log(torch.Tensor([min_channels])) / torch.log(torch.Tensor([2]))
        end = torch.log(torch.Tensor([max_channels])) / torch.log(torch.Tensor([2]))
        self.channels = torch.logspace(start.item(), end.item(), num_blocks+1, base=2., dtype=int).sort(descending=True)[0]

        if self.use_class_condition:
            self.embeds = torch.nn.Embedding(self.num_classes, self.noise_channels, device=device_glob)

        self.head = nn.Sequential(
            nn.BatchNorm2d(self.min_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(self.min_channels, 3, kernel_size=3, padding=1, device=device_glob)),
            nn.Sigmoid()
        )

        self.blocks = nn.ModuleList()
        self.output_features = int(self.max_channels * 4 * 4)

        if self.use_class_condition:
            self.embed_to_input = nn.Linear(self.noise_channels*2, self.output_features, device=device_glob)

            for i in range(self.num_blocks):
                self.blocks.append(PreActResBlock(in_channels=self.channels[i].item(),
                                    out_channels=self.channels[i+1].item(),
                                    embed_channels=self.noise_channels*2,
                                    batchnorm=True,
                                    upsample=True,
                                    downsample=False))
        else:
            self.embed_to_input = nn.Linear(self.noise_channels, self.output_features, device=device_glob)   

            for i in range(self.num_blocks):
                self.blocks.append(PreActResBlock(in_channels=self.channels[i].item(),
                                    out_channels=self.channels[i+1].item(),
                                    embed_channels=self.noise_channels,
                                    batchnorm=True,
                                    upsample=True,
                                    downsample=False))         

    def forward(self, noise, labels):
        if self.use_class_condition:
            input_embeds = self.embeds(labels).to(device_glob)
            input_embeds = torch.concat((input_embeds, noise), dim=1)
        else:
            input_embeds = noise

        input_embeds = input_embeds.to(device_glob).float()
        # print('input_embeds.shape[1], self.max_channels * 4 * 4 / input_embeds.shape[0])', input_embeds.shape[1], self.max_channels * 4 * 4 / input_embeds.shape[0])
        
        input_tensor = self.embed_to_input(input_embeds)
        input_tensor = input_tensor.reshape((noise.shape[0], self.max_channels, 4, 4))
        
        for b in self.blocks:
            # print('g', b)
            input_tensor = b(input_tensor, input_embeds)

        outputs = self.head(input_tensor)

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()

        self.min_channels = min_channels
        self.max_channels = max_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.use_projection_head = use_projection_head

        start = torch.log(torch.Tensor([min_channels])) / torch.log(torch.Tensor([2]))
        end = torch.log(torch.Tensor([max_channels])) / torch.log(torch.Tensor([2]))
        self.channels = torch.logspace(start.item(), end.item(), num_blocks+1, base=2., dtype=int) 

        if self.use_projection_head:
            self.embeds = spectral_norm(torch.nn.Embedding(self.num_classes, self.max_channels, device=device_glob))

        self.head = nn.Sequential(
            # nn.BatchNorm2d(3),
            # nn.ReLU(),
            spectral_norm(nn.Conv2d(3, self.min_channels, kernel_size=3, padding=1, device=device_glob))
        )

        self.block_after_conv = nn.Sequential(
            nn.ReLU(),
            nn.LPPool2d(1, kernel_size=4), # Sum pooling
        )

        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            # if i != (self.num_blocks - 1):
            self.blocks.append(PreActResBlock(in_channels=self.channels[i].item(),
                                out_channels=self.channels[i+1].item(),
                                embed_channels=None,
                                batchnorm=False,
                                upsample=False,
                                downsample=True))
            # else:
            #     self.blocks.append(PreActResBlock(in_channels=self.channels[i].item(),
            #                         out_channels=self.channels[i].item(),
            #                         embed_channels=None,
            #                         batchnorm=False,
            #                         upsample=False,
            #                         downsample=True))

        self.psi = spectral_norm(nn.Linear(self.max_channels, 1, device=device_glob))

    def forward(self, inputs, labels):
        if self.use_projection_head:
            class_embeddings = self.embeds(labels)

            x = self.head(inputs)
            for b in self.blocks:
                # print('d', b)
                x = b(x)
            
            x = self.block_after_conv(x)
            x = x.squeeze()

            x1 = self.psi(x).squeeze()

            # x.shape: [B, max_channels]
            # class_embeddings.shape: [B, max_channels]

            # x = torch.bmm(x, class_embeddings.view(class_embeddings.shape[0], self.max_channels, 1))
            x = (x * class_embeddings).sum(dim=1)
            # print('x', x.shape)
            # print('x1', x1.shape)
            scores = x + x1 # TODO

        else:
            scores = self.head(inputs)
            for b in self.blocks:
                # print('d', b)
                scores = b(scores)
            
            scores = self.block_after_conv(scores)
            scores = scores.squeeze()

            scores = self.psi(scores)
            
        scores = scores.squeeze()
        assert scores.shape == (inputs.shape[0],)
        return scores