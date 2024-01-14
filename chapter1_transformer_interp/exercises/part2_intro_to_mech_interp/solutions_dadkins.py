# %% 

import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_intro_to_mech_interp"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")


# %%

# Exercise: finding model parameters 

print(gpt2_small.cfg)

n_layers = gpt2_small.cfg.n_layers
n_heads = gpt2_small.cfg.n_heads
n_ctx = gpt2_small.cfg.n_ctx 
# %%
print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17]))
# %%

model_description_text = '''## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)

logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]

print(prediction.shape)
input_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
print("Correct token count: ", (input_tokens == prediction).bool().sum())
# %%
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

attn_patterns_layer_0 = gpt2_cache["pattern", 0]

# %% 
import math 
# Exercise: verify activations

layer0_pattern_from_cache = gpt2_cache["pattern", 0]

# My code
# q, k caches are [num_tokens, num_heads, d_head]
q = gpt2_cache["blocks.0.attn.hook_q"]
k = gpt2_cache["blocks.0.attn.hook_k"]

qk = einops.einsum(q, k, 'ntq nh dh, ntk nh dh -> nh ntq ntk')

mask = t.triu(t.ones(q.shape[0], k.shape[0]), diagonal=1).bool().to(qk.device)
qk_masked = t.masked_fill(qk, mask, -1e9)

layer0_pattern_from_q_and_k = t.nn.functional.softmax(qk_masked / math.sqrt(gpt2_small.cfg.d_head), dim=-1)
# pattern should be: [num_heads, num_tokens, num_tokens]

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Manual activations tests passed!")

# %%

# Visualizing attention heads
 
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)

gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(cv.attention.attention_patterns(
    tokens=gpt2_str_tokens, 
    attention=attention_pattern,
    attention_head_names=[f"L0H{i}" for i in range(12)],
))

# note: it looks like H8 focuses on commas
# H7 is a previous token head? 

# %% 

# Neuron activations

neuron_activations_for_all_layers = t.stack([
    gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)
# shape = (seq_pos, layers, neurons)

cv.activations.text_neuron_activations(
    tokens=gpt2_str_tokens,
    activations=neuron_activations_for_all_layers
)

# %% 

neuron_activations_for_all_layers_rearranged = utils.to_numpy(einops.rearrange(neuron_activations_for_all_layers, "seq layers neurons -> 1 layers seq neurons"))

cv.topk_tokens.topk_tokens(
    # Some weird indexing required here ¯\_(ツ)_/¯
    tokens=[gpt2_str_tokens], 
    activations=neuron_activations_for_all_layers_rearranged,
    max_k=7, 
    first_dimension_name="Layer", 
    third_dimension_name="Neuron",
    first_dimension_labels=list(range(12))
)
# %%

### SECTION: Induction Heads

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True, # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b", 
    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer"
)

from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device)
model.load_state_dict(pretrained_weights)
# %%

# Exercise: viz attention heads

text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

display(cv.attention.attention_patterns(
    tokens=model.to_str_tokens(text), 
    attention=cache['pattern', 1],
    attention_head_names=[f"L0H{i}" for i in range(12)],
))


# %%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    out = []
    for layer in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            pattern = cache['pattern', layer][head]
            diag = t.diag(pattern)
            avg_current_value = diag.mean()
            # print("Average current value for ", layer, head, " is: ", avg_current_value)
            if (avg_current_value > 0.3):
                out.append(str(layer) + "." + str(head))
    return out 

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    out = []
    for layer in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            pattern = cache['pattern', layer][head]
            diag = t.diag(pattern, diagonal=-1)
            avg_prev_value = diag.mean()
            # print("Average current value for ", layer, head, " is: ", avg_current_value)
            if (avg_prev_value > 0.3):
                out.append(str(layer) + "." + str(head))
    return out 


def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    out = []
    for layer in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            pattern = cache['pattern', layer][head]
            avg_prev_value = pattern[:, 0].mean()
            # print("Average current value for ", layer, head, " is: ", avg_current_value)
            if (avg_prev_value > 0.5):
                out.append(str(layer) + "." + str(head))
    return out 


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))