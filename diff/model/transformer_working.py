import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from flash_attn import flash_attn_func


from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    get_bias_dropout_add_scale, 
    modulate_fused,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout
        

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    
    def forward(self, x, rotary_cos_sin, c, attention_mask=None):
        
        def check_nan(tensor, name):
            if torch.isnan(tensor).any():
                print(f"NaN detected in {name}")
                # from remote_pdb import RemotePdb; RemotePdb('127.0.0.1', 4444).set_trace()
                return True
            return False
        
        # if torch.isnan(x).any():
        #         print("NaN in final input_2")
        #         print("input_2 stats:", x[~torch.isnan(x)].min().item(), 
        #             x[~torch.isnan(x)].max().item())
        
        batch_size, seq_len = x.shape[0], x.shape[1]

        # check_nan(x, "input x")

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # print("Modulation parameter stats:")
        # print(f"shift_msa: [{shift_msa.min().item()}, {shift_msa.max().item()}]")
        # print(f"scale_msa: [{scale_msa.min().item()}, {scale_msa.max().item()}]")
        # print(f"gate_msa: [{gate_msa.min().item()}, {gate_msa.max().item()}]")
        
        
        def check_numerical_stability(tensor, name, max_val=1e2):
            if tensor.abs().max() > max_val:
                print(f"Large values detected in {name}: {tensor.abs().max().item()}")
                return True
            if torch.isnan(tensor).any():
                print(f"NaN detected in {name}")
                return True
            if torch.isinf(tensor).any():
                print(f"Inf detected in {name}")
                return True
            
            return False
        
        
        
        for name, param in zip(
                ['shift_msa', 'scale_msa', 'gate_msa', 'shift_mlp', 'scale_mlp', 'gate_mlp'],
                [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
            ):
            check_nan(param, name)
        
        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # check_nan(x, "after modulate_fused")
        # if torch.isnan(x).any():
        #         print("NaN in final input_3")
        #         print("input_3 stats:", x[~torch.isnan(x)].min().item(), 
        #             x[~torch.isnan(x)].max().item())
        
        qkv = self.attn_qkv(x)
        # print("before clamping")
        # if check_numerical_stability(qkv, "QKV computation"):
        #     # Clip values if needed
        #     print("Clamping values")
        #     qkv = torch.clamp(qkv, min=-5, max=5)
        
        # check_nan(qkv, "after attn_qkv")
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        # check_nan(qkv, "after rearrange")
        
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
            # check_nan(qkv, "after rotary")


        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        # check_nan(qkv, "before flash attention")
        
        # Calculate cu_seqlens based on attention mask if provided
        if attention_mask is not None:
            # print("entering attentionmaks in cu_seqlens computation")
            # Get actual sequence lengths for each batch item
            seq_lens = attention_mask.sum(dim=1).to(torch.int32)
            cu_seqlens = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=qkv.device),
                seq_lens.cumsum(0,dtype=torch.int32) # modified by foobar to add dtype for cumsum as typically it upcasts
            ])
            # https://github.com/pytorch/pytorch/issues/128294
            # print(f"dtype of cu_seqlens in the attentionmaks if condition is {cu_seqlens.dtype}")
            # print(f"dtype of seqlens in the attentionmaks if condition is {seq_lens.dtype}")
            # print(f"dtype of seqlens.cumsum in the attentionmaks if condition is {seq_lens.cumsum(0).dtype}")
            # print(f"attention_maks is {attention_mask}, seq_lens is {seq_lens}, cumsum is {seq_lens.cumsum(0)}")
            # import pdb; pdb.set_trace()
            # from remote_pdb import RemotePdb; RemotePdb('127.0.0.1', 4444).set_trace()
        else:
            # print("Not entering attentionmaks in cu_seqlens computation") # TODO - I have to understand why during the sampling at every 500th step I am not entering here
            # import traceback
            # traceback.print_stack()
            # Original behavior - all sequences are full length
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        
        
        # Use max_seqlen from attention mask if provided
        seq_lens = attention_mask.sum(dim=1).to(torch.int32)
        max_seqlen = seq_lens.max().item() if attention_mask is not None else seq_len
        
        
        # DEBUGGING: Force fixed cu_seqlens and max_seqlen
        # print("Original cu_seqlens would have been:", 
        #     torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, 
        #                 dtype=torch.int32, device=qkv.device))
        
        # Force fixed values
        # cu_seqlens = torch.tensor([0, 128, 256, 384, 512], 
        #                         device=qkv.device, 
        #                         dtype=torch.int32)
        
        # cu_seqlens = torch.arange(
        #         0, (batch_size + 1) * seq_len, step=seq_len,
        #         dtype=torch.int32, device=qkv.device
        #     )
        
        # max_seqlen = seq_len
        
        # print("QKV shape before flash attention:", qkv.shape)
        # print("Using fixed cu_seqlens:", cu_seqlens)
        # print("Using fixed max_seqlen:", max_seqlen)

        
        
        
        # print(cu_seqlens,max_seqlen)
        
        
        
        # print(f"dtype of cu_seqlens is {cu_seqlens.dtype}")
        
        # print("QKV stats before flash attention:")
        # print(f"- Shape: {qkv.shape}")
        # print(f"- Range: [{qkv.min().item()}, {qkv.max().item()}]")
        # print(f"- Mean: {qkv.mean().item()}")

        
        
        
        ###########################################################################################################################
        
        x = flash_attn_varlen_qkvpacked_func(
            qkv, 
            cu_seqlens,
            max_seqlen,
            0.,
            causal=False, # modified by foobar from False
        )
        
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        
        ###########################################################################################################################
        
        # if check_numerical_stability(x, "Flash attention output"):
        #         # If unstable, try with scaled inputs
        #         qkv_scaled = qkv * 0.1
        #         x = flash_attn_varlen_qkvpacked_func(
        #             qkv_scaled,
        #             cu_seqlens,
        #             max_seqlen,
        #             0.,
        #             causal=True # modified by foobar from False
        #         ) * 10
        
        
        
        # print("Output stats after flash attention:")
        # print(f"- Shape: {x.shape}")
        # print(f"- Range: [{x.min().item()}, {x.max().item()}]")
        # print(f"- Mean: {x.mean().item()}")
        
        # check_nan(x, "after flash attention")

        
       
        
        
        
        # import time; time.sleep(5)
        
        # quit()
        # if torch.isnan(x).any():
        #         print("NaN in final input_4")
        #         print("input_4 stats:", x[~torch.isnan(x)].min().item(), 
        #             x[~torch.isnan(x)].max().item())
                # import traceback
                # traceback.print_stack()
        
        

        # Apply attention mask to output if provided - TODO uncomment this
        # if attention_mask is not None:
        #     x = x * attention_mask.unsqueeze(-1)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation 
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        
        # check_nan(x, "after final dropout")
        
        # Apply attention mask to final output if provided - TODO uncomment this
        # if attention_mask is not None:
        #     x = x * attention_mask.unsqueeze(-1)
            
        return x







    # def forward(self, x, rotary_cos_sin, c, seqlens=None): # modified by foobar
    #     batch_size, seq_len = x.shape[0], x.shape[1]

    #     bias_dropout_scale_fn = self._get_bias_dropout_scale()

    #     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    #     # attention operation
    #     x_skip = x
    #     x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    #     # dtype0 = x.dtype

    #     qkv = self.attn_qkv(x)
    #     qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    #     with torch.cuda.amp.autocast(enabled=False):
    #         cos, sin = rotary_cos_sin
    #         qkv = rotary.apply_rotary_pos_emb(
    #             qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
    #         )
    #     qkv = rearrange(qkv, 'b s ... -> (b s) ...')
    #     if seqlens is None:
    #         cu_seqlens = torch.arange(
    #             0, (batch_size + 1) * seq_len, step=seq_len,
    #             dtype=torch.int32, device=qkv.device
    #         )
    #     else:
    #         cu_seqlens = seqlens.cumsum(-1)
    #     x = flash_attn_varlen_qkvpacked_func(
    #         qkv, cu_seqlens, seq_len, 0., causal=False)
        
    #     x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

    #     x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

    #     # mlp operation
    #     x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
    #     return x



class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SEDD(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout) for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim)
        self.scale_by_sigma = config.model.scale_by_sigma

    
    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    # def forward(self, indices, sigma): # modified by foobar

    #     x = self.vocab_embed(indices)
    #     c = F.silu(self.sigma_map(sigma))

    #     rotary_cos_sin = self.rotary_emb(x)

    #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #         for i in range(len(self.blocks)):
    #             x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)

    #         x = self.output_layer(x, c)


    #     if self.scale_by_sigma:
    #         assert self.absorb, "Haven't configured this to work."
    #         esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
    #         x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0
            
    #     x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

    #     return x
    
    
    
    def forward(self, indices, sigma, attention_mask=None): 
        
        
        
        x = self.vocab_embed(indices) 
        
        if torch.isnan(x).any():
            print("NaN in final input")
            # print("input stats:", x[~torch.isnan(x)].min().item(), 
            #     x[~torch.isnan(x)].max().item())
            # import traceback
            # traceback.print_stack()
        
        c = F.silu(self.sigma_map(sigma))
        # print("Embedding stats:", x.min().item(), x.max().item())
        # print("Conditioning stats:", c.min().item(), c.max().item())
        
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                # Pass attention_mask to blocks
                x = self.blocks[i](x, rotary_cos_sin, c, attention_mask=attention_mask)

            
            
            
            x = self.output_layer(x, c)

        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            # print("Scale factor stats:", esigm1_log.min().item(), esigm1_log.max().item())
            x = x - esigm1_log - np.log(x.shape[-1] - 1)
            
            
        # if torch.isnan(x).any():
        #     print("NaN in final output")
        #     print("Output stats:", x[~torch.isnan(x)].min().item(), 
        #         x[~torch.isnan(x)].max().item())
        
        
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        # Apply attention mask to output if provided - TODO uncomment this
        # if attention_mask is not None:
        #     x = x * attention_mask.unsqueeze(-1)

        return x
