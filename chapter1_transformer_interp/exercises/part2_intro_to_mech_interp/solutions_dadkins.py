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

# %%

# Exercise: induction heads, plotting per-token loss on repeated data

def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    half = t.randint(0, model.tokenizer.vocab_size, (batch, seq_len))
    repeat = half.clone()
    return t.cat([prefix, half, repeat], dim=1).to(device)

def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache


seq_len = 50
batch = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)

# %% 

# Exercise: looking for induction heads with charts

# We see heads 4 and 10 look like induction heads (or they are using positional encoding, but could easily invalidate that)
display(cv.attention.attention_patterns(attention=rep_cache["pattern", 1], tokens=rep_str))

# %% 

# Exercise: induction head detector

def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    out = []
    for layer in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            head_name = str(layer) + "." + str(head)
            attn_pattern = cache['pattern', layer][head]
            attn_diag = t.diag(attn_pattern, diagonal=-1*int((attn_pattern.shape[1]-1)/2 - 1))
            if (attn_diag.mean() > 0.3):
                out.append(head_name)
    return out 

print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %% 

tokens = "Here is a hooked transformer function"

def hook_function(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint
) -> Float[Tensor, "batch heads seqQ seqK"]:

    # modify attn_pattern (can be inplace)
    return attn_pattern

loss = model.run_with_hooks(
    tokens, 
    return_type="loss",
    fwd_hooks=[
        ('blocks.1.attn.hook_pattern', hook_function)
    ]
)

print(loss)
# %%

seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    attn_diag = pattern.diagonal(dim1=-2, dim2=-1, offset=-1*int((pattern.size(-1)-1)/2 - 1))
    
    # average across batch and position
    score = einops.reduce(attn_diag, "batch head_index position -> head_index", "mean")
    
    induction_score_store[hook.layer(), :] = score



pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=400
)
# %%

gpt2_small_induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)

def gpt2_small_induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    attn_diag = pattern.diagonal(dim1=-2, dim2=-1, offset=-1*int((pattern.size(-1)-1)/2 - 1))
    
    # average across batch and position
    score = einops.reduce(attn_diag, "batch head_index position -> head_index", "mean")
    
    gpt2_small_induction_score_store[hook.layer(), :] = score

def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), 
            attention=pattern.mean(0)
        )
    )

# here, we find induction in gpt2_small 
gpt2_small.run_with_hooks(
    rep_tokens_10,
    fwd_hooks=[(
        lambda name: name.endswith("pattern"),
        gpt2_small_induction_score_hook
    )]
)

imshow(
    gpt2_small_induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="GPT2_small induction score by head", 
    text_auto=".2f",
    width=900, height=400
)

# Induction heads in  5.0, 5.1, 5.5, 6.9, 7.3, 7.10, and some weaker ones in L10/L11

# %% 

def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    '''
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    '''
    W_U_correct_tokens = W_U[:, tokens[1:]]
    
    # we want to multiply embed by W_U for each position

    # this finds out, for each logit in the sequence, how much the embedding layer contributing to that logit 
    embed_attribution = einops.einsum(W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq")

    # how much each head in L1 contributed to that logit 
    l1_attribution =  einops.einsum(W_U_correct_tokens, l1_results[:-1], "d_model seq_len, seq_len n_heads d_model -> seq_len n_heads")

    l2_attribution =  einops.einsum(W_U_correct_tokens, l2_results[:-1], "d_model seq_len, seq_len n_heads d_model -> seq_len n_heads")

    return t.concat([embed_attribution.unsqueeze(-1), l1_attribution, l2_attribution], dim=-1)
    


text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")
# %%

embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])

plot_logit_attribution(model, logit_attr, tokens)

# attributions that are heavy in the direct path are just easy bigrams, because that is all that can be approximated in that layer 

# %% 

seq_len = 50

embed = rep_cache["embed"]
l1_results = rep_cache["result", 0]
l2_results = rep_cache["result", 1]
first_half_tokens = rep_tokens[0, : 1 + seq_len]
second_half_tokens = rep_tokens[0, seq_len:]

print("Shape of embed: ", embed.shape)
print("Shape of l1_results: ", l1_results.shape)
print("Shape of l2_results: ", l2_results.shape)
print("Shape of first_half_tokens: ", first_half_tokens.shape)
print("Shape of second_half_tokens: ", second_half_tokens.shape)
print("Shape of W_U: ", model.W_U.shape)


first_half_logit_attr = logit_attribution(embed[:seq_len+1], l1_results[:seq_len+1], l2_results[:seq_len+1], model.W_U, first_half_tokens)
second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)

assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)

plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")

# %%
