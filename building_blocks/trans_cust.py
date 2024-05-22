import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn



def precompute_rotary_emb(dim, max_positions):
    """
    RoPE uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 1/10000^(-2(i-1)/dim) for i in [1, dim/2]

    Since the maximum length of sequences is known, we can precompute
    these values to speed up training.

    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """

    rope_cache = None
    # TODO: [part g]
    ### YOUR CODE HERE ###
    
    theta_i = torch.tensor([10000**(-2*(i-1)/dim) for i in range(1, dim//2 + 1)])
    

    t = torch.tensor([i for i in range(0, max_positions)])


    t_theta = torch.outer(t, theta_i)


    rope_cache = torch.stack((torch.cos(t_theta), torch.sin(t_theta)), dim=-1)  
    
    return


    
    
    ### END YOUR CODE ###
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    # TODO: [part g]
    # You might find the following functions useful to convert
    # between real and complex numbers:

    # torch.view_as_real - https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html

    # Note that during inference, the length of the sequence might be different
    # from the length of the precomputed values. In this case, you should use
    # truncate the precomputed values to match the length of the sequence.

    rotated_x = None
    ### YOUR CODE HERE ###
    nb, nh, t, dim = x.size()  # Unpack dimensions from the input tensor

    if t > rope_cache.shape[0]:
        raise ValueError("The sequence length of the input tensor exceeds the maximum positions in the rope_cache.")

    rope_cache = rope_cache[:t, :]  # Truncate rope_cache to match the sequence length

    # Convert the input tensor to a complex number format
    # Reshape x to separate real and imaginary parts for each pair
    x_reshaped = x.view(nb, nh, t, dim // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)  # Convert to complex numbers

    # Convert rope_cache to complex numbers
    # rope_cache should be (t, dim/2, 2) to convert correctly
    rope_cache_reshaped = rope_cache.view(t, dim // 2, 2)
    rope_complex = torch.view_as_complex(rope_cache_reshaped)

    # Apply the rotary embeddings
    # rope_complex is (t, dim/2) and x_complex is (nb, nh, t, dim/2),
    # we use broadcasting to multiply across nb, nh and t
    #rotated_x_complex = x_complex * rope_complex.unsqueeze(0).unsqueeze(0)  # Broadcasting over nb and nh
    block_size = 1000  # Example block size for the third dimension

    # Initialize an empty tensor for the result
    rotated_x_complex = torch.empty_like(x_complex)

    # Perform block-wise multiplication
    for i in range(0, x_complex.shape[2], block_size):
        # Determine the end of the current block
        end = min(i + block_size, x_complex.shape[2])
        
        # Extract the block from x_complex
        x_block = x_complex[:, :, i:end, :]
        
        # Extract the corresponding block from rope_complex
        rope_block = rope_complex[i:end, :].unsqueeze(0).unsqueeze(0)
        
        # Perform the multiplication for the current block
        rotated_x_complex[:, :, i:end, :] = x_block * rope_block

    # Convert the complex-number results back to real format
    rotated_x = torch.view_as_real(rotated_x_complex)  # Convert back to real numbers

    # Reshape back to the original shape (nb, nh, t, dim)
    rotated_x = rotated_x.view(nb, nh, t, dim)

    ### END YOUR CODE ###
    return rotated_x

d_model = 1024
nhead = 8
max_seq_length = 1000
block_size = 1000

rope_cache_shared = precompute_rotary_emb(d_model // nhead, max_seq_length)

# Define a module to hold the shared rope_cache
class SharedCache(nn.Module):
    def __init__(self, rope_cache):
        super().__init__()
        self.register_buffer("rope_cache", rope_cache)

shared_cache_module = SharedCache(rope_cache_shared)


class ChunkedLinear(nn.Module):
    def __init__(self, seq_len,chunk_size=2000, out_features=1000):
        super(ChunkedLinear, self).__init__()
        self.linear_blocks = nn.ModuleList()
        self.block_size = chunk_size
        self.num_blocks = None 
        self.out_features = out_features
        self.seq_len = seq_len
        
        self.num_blocks = (seq_len + self.block_size - 1) // self.block_size
        self.linear_blocks = nn.ModuleList(
            [nn.Linear(min(self.chunk_size, seq_len - i * self.chunk_size), self.out_features)
             for i in range(self.num_blocks)])    

    def forward(self, x):
     
        # Initialize the accumulated output
        accumulated_output = []
        
        # Handle the remaining elements
        for i in range(self.num_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, self.seq_len)
            chunk = x[:,:,:, start_idx:end_idx]
            chunk_output = self.linear_blocks[i](chunk)
            accumulated_output.append(chunk_output)
            '''
            if accumulated_output is None:
                accumulated_output = chunk_output
            else:
                accumulated_output = accumulated_output + chunk_output 
            '''   
        accumulated_output = torch.cat(accumulated_output, dim=-1)    
        return accumulated_output
    
    def to(self, device):
        # Move the model itself to the device
        super(ChunkedLinear, self).to(device)
        # Move each linear block to the device
        for linear in self.linear_blocks:
            linear.to(device)
        return self
    
    
class CausalSelfAttention(nn.Module):
    def __init__(self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        normalize_before=False,
        max_seq_length=3000,
        use_rotary=False,
        ):    
        super().__init__()     
        k = 1000
        self.proj_k = ChunkedLinear(chunk_size=block_size, out_features=1000)
        self.proj_v = ChunkedLinear(chunk_size=block_size, out_features=1000)   
        self.key = nn.Linear(d_model, d_model,bias=False)
        self.query = nn.Linear(d_model, d_model,bias=False)
        self.value = nn.Linear(d_model, d_model,bias=False)
        
        self.rope_cache = shared_cache_module.rope_cache if use_rotary and shared_cache_module else None
        
        self.initialized = False
        '''
        if use_rotary:
            assert (d_model % nhead) % 2 == 0

            rope_cache = None
            rope_cache = precompute_rotary_emb(d_model // nhead, max_seq_length)
            
            self.register_buffer("rope_cache", rope_cache)
        '''   
        self.rope = use_rotary
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(d_model, d_model)

        self.n_head = nhead
        
        if self.rope:
            rope_cache = precompute_rotary_emb( d_model//nhead, max_seq_length)
        
        def initialize_layers(self, input_size):
            self.proj_k = ChunkedLinear(input_size=input_size,chunk_size=block_size, out_features=1000)
            self.proj_v = ChunkedLinear(input_size=input_size,chunk_size=block_size, out_features=1000) 
            self.initialized = True

    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        k = self.proj_k(k.transpose(-1, -2)).transpose(-1, -2)
        v = self.proj_v(v.transpose(-1, -2)).transpose(-1, -2)

        if self.rope:
            # TODO: [part g] Apply RoPE to the query and key.
            ### YOUR CODE HERE ###            
            rope_cache = self.rope_cache.to(x.device)
            q = apply_rotary_emb(q, rope_cache)
            k = apply_rotary_emb(k, rope_cache)
            ### END YOUR CODE ###
        '''
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        #att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        '''


        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.size(-1))
        
        attn = torch.softmax(attn_scores, dim=-1)
        
        attn = self.attn_drop(attn)
        
        y = torch.matmul(attn, v)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return y
        #y = self.resid_drop(self.proj(y))
        
    def to(self, device):
        super(CausalSelfAttention, self).to(device)
        self.proj_k.to(device)
        self.proj_v.to(device)
        if self.rope_cache is not None:
            self.rope_cache = self.rope_cache.to(device)
        return self



class SelfAttentionTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        max_seq_length=1000,
        pos_embedding_method = 'rotary'
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before,max_seq_length)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.pos_embedding_method = pos_embedding_method
        '''
        self.position_embedding = {}
        if pos_embedding_method == "vanilla":
            self.position_embedding["vanilla"] = self._generate_sinusoidal_embeddings(max_seq_length, d_model)
        elif pos_embedding_method == "rotary":
            self.position_embedding["rotary"] = self._generate_rotary_embeddings(max_seq_length, d_model)
        '''
        #self.pos_encoding = TrainablePositionalEncoding(d_model, max_seq_length, dropout)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_sinusoidal_embeddings(self, seq_length, d_model):
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pos_embedding = torch.zeros(seq_length, d_model)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding.unsqueeze(1)
    
      

    def forward(self, src, mask=None):
        src_shape = src.shape
        #src = src.permute(1, 0, 2)
        if mask is not None:
            mask = mask.flatten(1)

        '''
        if len(self.position_embedding.keys())>0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=self.position_embedding)
        else:
            memory = self.encoder(src, src_key_padding_mask=mask)   
        '''             
        memory = self.encoder(src, src_key_padding_mask=mask)  
        return memory


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        normalize_before=False,
        max_seq_length=512
    ):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model,nhead,dropout=dropout,use_rotary=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
   
      

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        #x_pos = self.with_pos_embed(src, pos)
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src





def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")



class TransformerPass(nn.Module):
    def __init__(self, input_channels, d_model,pos_embedding_method=False):
        super(TransformerPass, self).__init__()
        self.output_channels = input_channels

        # Linear layers to convert to and from transformer dimensions
        self.linear_to_transformer = nn.Linear(input_channels, d_model)
        self.linear_from_transformer = nn.Linear(d_model, input_channels)
        self.transformer = SelfAttentionTransformer(d_model,pos_embedding_method=pos_embedding_method)
        
    def forward(self, x):
        batch_size = x.size(0)
        x_shape = x.shape
        # Flatten spatial dimensions
        x = x.view(batch_size, self.output_channels, -1).permute(0,2,1)

        # Apply linear layer to get desired transformer dimension
        x = self.linear_to_transformer(x)

        # Apply transformer
        x = self.transformer(x)

        # Apply the reverse linear layer to convert back to the original conv output dimension
        x = self.linear_from_transformer(x).permute(0,2,1)

        # Reshape back to original conv output shape
        spatial_dims = x_shape[2:]
        x = x.view(batch_size, self.output_channels, *spatial_dims)

        return x