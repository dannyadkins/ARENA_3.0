# %%

import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import webbrowser

# Make sure exercises are in the path
chapter = r"chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_transformer_from_scratch"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
# import part1_transformer_from_scratch.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == '__main__'

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
# %%
"""
Naive autoregressive decoding 
"""

if (MAIN):
    reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    tokens = reference_gpt2.to_tokens(reference_text).to(device)

    logits, cache = reference_gpt2.run_with_cache(tokens)

    next_token = logits[0, -1].argmax(dim=-1)
    next_char = reference_gpt2.to_string(next_token)
    print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

    for i in range(10):
        print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
        # Define new input sequence, by appending the previously generated token
        tokens = t.cat([tokens, next_token[None, None]], dim=-1)
        # Pass our new sequence through the model, to get new output
        logits = reference_gpt2(tokens)
        # Get the predicted token at the end of our sequence
        next_token = logits[0, -1].argmax(dim=-1)
        # Decode and print the result
        next_char = reference_gpt2.to_string(next_token)
    
# %%

if (MAIN):
    # Activations of reference model
    for activation_name, activation in cache.items():
        # Only print for first layer
        if ".0." in activation_name or "blocks" not in activation_name:
            print(f"{activation_name:30} {tuple(activation.shape)}")

    for name, param in reference_gpt2.named_parameters():
        # Only print for first layer
        if ".0." in name or "blocks" not in name:
            print(f"{name:18} {tuple(param.shape)}")

# Output: 
# hook_embed                     (1, 35, 768)
# hook_pos_embed                 (1, 35, 768)
# blocks.0.hook_resid_pre        (1, 35, 768)
# blocks.0.ln1.hook_scale        (1, 35, 1)
# blocks.0.ln1.hook_normalized   (1, 35, 768)
# blocks.0.attn.hook_q           (1, 35, 12, 64)
# blocks.0.attn.hook_k           (1, 35, 12, 64)
# blocks.0.attn.hook_v           (1, 35, 12, 64)
# blocks.0.attn.hook_attn_scores (1, 12, 35, 35)
# blocks.0.attn.hook_pattern     (1, 12, 35, 35)
# blocks.0.attn.hook_z           (1, 35, 12, 64)
# blocks.0.hook_attn_out         (1, 35, 768)
# blocks.0.hook_resid_mid        (1, 35, 768)
# blocks.0.ln2.hook_scale        (1, 35, 1)
# blocks.0.ln2.hook_normalized   (1, 35, 768)
# blocks.0.mlp.hook_pre          (1, 35, 3072)
# blocks.0.mlp.hook_post         (1, 35, 3072)
# blocks.0.hook_mlp_out          (1, 35, 768)
# blocks.0.hook_resid_post       (1, 35, 768)
# ln_final.hook_scale            (1, 35, 1)
# ln_final.hook_normalized       (1, 35, 768)
# embed.W_E          (50257, 768)
# pos_embed.W_pos    (1024, 768)
# blocks.0.ln1.w     (768,)
# blocks.0.ln1.b     (768,)
# ...
# ln_final.w         (768,)
# ln_final.b         (768,)
# unembed.W_U        (768, 50257)
# unembed.b_U        (50257,)
# %%

# config

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

# %%

# tests 
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape)
    try: reference_output = gpt2_layer(input)
    except: reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")


# %%

# EXERCISE: This is where the fun starts. These are my implementations:

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    # Make mean 0
    # Normalize to have variance 1
    # Scale with learned weights
    # Translate with learned bias
    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        # the keepdim preserves the d_model
        mean = residual.mean(dim=-1, keepdim=True)
        eps = self.cfg.layer_norm_eps
        variance = residual.var(dim=-1, keepdim=True, unbiased=False)  
        # learnable 
        w = self.w
        b = self.b

        out = (residual - mean ) / t.sqrt(variance + eps) * w + b
        print(out.shape)
        return out          

rand_float_test(LayerNorm, [2, 4, 768])
load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])
# %%

# EXERCISE: Embedding! 

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        # tokens are token IDs, so we could do self.W_E[tokens]
        # alternatively, we could onehot encode them and do a matmul? would this be functionally different?
        one_hot_tokens = t.nn.functional.one_hot(tokens, num_classes=self.cfg.d_vocab)
        return t.matmul(one_hot_tokens.float(), self.W_E)

rand_int_test(Embed, [2, 4])
load_gpt2_test(Embed, reference_gpt2.embed, tokens)
# %%

# EXERCISE: positional embedding

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        # also just look up, but instead use the sequence position
        sequence_positions = t.arange(tokens.shape[1], device=tokens.device)
        return self.W_pos[sequence_positions]

rand_int_test(PosEmbed, [2, 4])
load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
# %%
