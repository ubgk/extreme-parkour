import torch
import torch.nn as nn
import sys
import torchvision

from typing import Tuple

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

class StackDepthEncoder(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )

        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=env_cfg.depth.buffer_len, out_channels=16, kernel_size=4, stride=2),  # (30 - 4) / 2 + 1 = 14,
                                    activation,
                                    nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2), # 14-2+1 = 13,
                                    activation)
        self.mlp = nn.Sequential(nn.Linear(16*14, 32), 
                                 activation)
        
    def forward(self, depth_image, proprioception):
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(None, depth_image.flatten(0, 1), None)  # [batch_size * num, 32]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 32]
        depth_latent = self.conv1d(depth_latent)
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        return depth_latent

    
class AttentionEncoder(nn.Module):
    def __init__(
        self,
        num_obs: int,
        hidden_dim: int = 64,
        output_dim: int = -1,
        height_points: torch.Tensor = None,
        exteroception_dims: Tuple[int, int] = (12, 11),
        activation: str = "elu",
        conv_params: dict = {"kernel_size": 5, "stride": 1, "padding": "same"}  # Default parameters for convolution,
    ):
        """Attention-based encoder for proprioception and exteroception data.

        The encoder can be used to model both the actor and critic networks in an actor-critic architecture.

        Args:
            num_obs (int): Dimension of the proprioception input.
            hidden_dim (int, optional): Dimension of the hidden layer. Defaults to 64.
            output_dim (int, optional): Dimension of the output layer. If -1, defaults to hidden_dim. Defaults to -1.
            height_points (torch.Tensor | None, optional): Tensor containing the positions of the height points in the grid. Defaults to None.
            exteroception_dims (tuple[int, int]): Dimensions of the exteroception input (dim1, dim2).
            activation (str, optional): Activation function to use. Defaults to "elu".
            conv_params (dict, optional): Parameters for the convolutional layer. Defaults to {'kernel_size': 5, 'stride': 1}.

        Raises:
            AssertionError: If the exteroception dimensions are not divisible by the kernel size.
        """
        super().__init__()
        self.num_obs = num_obs
        self.num_patches = height_points.shape[-2]
        self.exteroception_dims = exteroception_dims
        self.height_points = height_points[0][..., :2].clone()
        
        assert conv_params["padding"] == "same", \
            "Padding must be set to 'same' to ensure the output dimensions match the input dimensions \
            for the convolutional layers."

        self.activation = nn.ELU() if activation == "elu" else nn.ReLU()
        self.proprioception_encoder = nn.Linear(num_obs, hidden_dim)

        self.position_encoder = nn.Embedding(
            num_embeddings = self.num_patches,
            embedding_dim = hidden_dim
        )

        # Convolutional encoder for exteroception
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, **conv_params),
            self.activation,
            nn.Conv2d(16, hidden_dim - 2, **conv_params),
            self.activation,
            nn.Flatten(-2),  # Flatten the last two dimensions to get patches
        )

        self.attention = nn.MultiheadAttention(
            embed_dim = hidden_dim, # Hidden dimension + 2 for yaw prediction
            num_heads = 16,
            batch_first = True,
        )

        self.output_dim = hidden_dim if output_dim == -1 else output_dim
        self.out_projection = nn.Linear(hidden_dim, self.output_dim)  # Project to the desired output dimension

        self.att_scores: torch.Tensor | None = None  # Placeholder for attention scores



    def forward(self, exteroception: torch.Tensor, proprioception: torch.Tensor, need_weights: bool = True) -> torch.Tensor:
        # breakpoint()
        num_envs = proprioception.shape[0]

        # Fold exteroception by the number of history steps
        exteroception = exteroception.view(
            num_envs,
            1, # Height measurement channel
            self.exteroception_dims[0], # x dimension
            self.exteroception_dims[1], # y dimension
        )  # (num_envs, channels, dim1, dim2)

        # Proprioception encoding
        proprio_encoded = self.activation(self.proprioception_encoder(proprioception))

        # Exteroception encoding
        extero_encoded = self.conv(exteroception) # (num_envs, hidden_dim, num_patches)
        extero_encoded = extero_encoded.permute(0, 2, 1)  # (num_envs, num_patches, hidden_dim - 2)
        extero_encoded = torch.cat([self.height_points.expand(num_envs, -1, -1), extero_encoded], -1) # (num_envs, num_patches, hidden_dim), add grid indices to the exteroception encodin

        # Unsqueeze to add a target sequence length dimension for attention
        proprio_encoded = proprio_encoded.unsqueeze(1)  # (num_envs, 1, hidden_dim)

        # Compute attention
        att_output, self.att_scores = self.attention(
            query = proprio_encoded,  # Query: (num_envs, 1, hidden_dim)
            key   = extero_encoded,   # Key (num_envs, num_patches, hidden_dim)
            value = extero_encoded,   # Value (num_envs, num_patches, hidden_dim)
            need_weights=need_weights
        ) # Output shape: (num_envs, 1, hidden_dim), (att_scores shape: (num_envs, 1, num_patches)

        att_output = att_output.squeeze(1)  # (num_envs, hidden_dim)

        # Output projection
        output = self.out_projection(self.activation(att_output))  # (num_envs, hidden_dim + 2)

        return output

class CriticWrapper(nn.Module):
    def __init__(self, encoder: AttentionEncoder, backbone: nn.Module, 
                 num_prop: int = -1, num_scan: int = -1, 
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.activation = activation

        assert num_prop >= 0 and num_scan >= 0, "num_prop and num_scan must be specified"
        self.num_prop = num_prop
        self.num_scan = num_scan

        assert self.encoder.num_obs == num_prop, "Encoder num_obs must match num_prop"
        assert self.encoder.exteroception_dims[0] * self.encoder.exteroception_dims[1] == num_scan, "Encoder exteroception_dims must match num_scan"


    def forward(self, critic_obs) -> torch.Tensor:
        proprioception = critic_obs[:, :self.num_prop]
        exteroception = critic_obs[:, self.num_prop:self.num_prop + self.num_scan]
        other_obs = critic_obs[:, self.num_prop + self.num_scan:]

        encoding_yaw = self.encoder(exteroception, proprioception)
        encoding = encoding_yaw[:, :-2]
        encoding = self.activation(encoding)
        encoding = torch.cat([proprioception, encoding, other_obs], dim=-1)

        output = self.backbone(encoding)
        return output