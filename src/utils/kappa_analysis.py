'''Mechanistic-interpretability helpers specific to the kappa operator.

κ(N x E y) = (NE · x) ⊠ y only rearranges the input symbols, so every output symbol has a well-defined
source ("provenance") position in the input (see src.utils.dyck_maps.kappa_provenance). These helpers
test whether the trained model's cross-attention and encoder states reflect that structure:

    - teacher_forced_cross_attention: cross-attention for all output steps in one forward pass
    - provenance_alignment: how often the most-attended input position is the true source position
    - plot_provenance_example: cross-attention heatmap with the true provenance overlaid
    - probe_regions: linear probe from encoder states to the region {root N, x, root E, y}

Token convention (same as Dataset): position 0 is BOS, input symbol j sits at position j + 1,
token id = symbol + 2.
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils.dyck_maps import kappa_provenance, kappa_regions

REGION_NAMES = ['root N', 'x', 'root E', 'y']
REGION_COLORS = ['tab:red', 'tab:blue', 'tab:purple', 'tab:orange']


def _symbols(token_row):
    '''Strip BOS and map token ids back to 0/1 symbols.'''
    return [int(t) - 2 for t in token_row[1:]]


@torch.no_grad()
def teacher_forced_cross_attention(model, inputs, targets, layer=0, head=0):
    '''Cross-attention [B, T, S] for every output step, using the ground-truth prefix.

    Row t is the query that predicts output symbol t (0-based); column s is input position s (0 = BOS).
    '''
    model.eval()
    model(src=inputs, tgt=targets[:, :-1])
    att = model._attention_scores()['decoder'][layer]['cross_attention']  # [B, H, T, S]
    return att[:, head].float().cpu()


def provenance_alignment(model, examples, num_examples=1000, batch_size=250, head=0):
    '''Compare the argmax of cross-attention with the true source position of each output symbol.

    Returns a dict with:
        top1: fraction of (example, step) pairs where the most-attended input position is the source
        within1: same, but allowing the argmax to be off by one position
        mass: mean attention weight placed on the source position
        top1_by_region: top-1 rate split by the region (root N / x / root E / y) of the source
        top1_by_step: top-1 rate per output step
    '''
    inputs, targets = examples['inputs'][:num_examples], examples['targets'][:num_examples]
    hits, near, mass, regions, steps = [], [], [], [], []
    for b in range(0, len(inputs), batch_size):
        src, tgt = inputs[b:b + batch_size], targets[b:b + batch_size]
        att = teacher_forced_cross_attention(model, src, tgt, head=head)  # [B, T, S]
        for i in range(att.shape[0]):
            w = _symbols(src[i].tolist())
            prov = np.array(kappa_provenance(w)) + 1  # +1 for BOS
            reg = np.array(kappa_regions(w))
            a = att[i].numpy()
            arg = a.argmax(-1)
            hits.append(arg == prov)
            near.append(np.abs(arg - prov) <= 1)
            mass.append(a[np.arange(len(prov)), prov])
            regions.append(reg[prov - 1])
            steps.append(np.arange(len(prov)))
    hits, near, mass = map(np.concatenate, (hits, near, mass))
    regions, steps = np.concatenate(regions), np.concatenate(steps)
    return {
        'top1': hits.mean(),
        'within1': near.mean(),
        'mass': mass.mean(),
        'top1_by_region': {REGION_NAMES[r]: hits[regions == r].mean() for r in range(4) if (regions == r).any()},
        'top1_by_step': np.array([hits[steps == s].mean() for s in range(steps.max() + 1)]),
    }


def plot_provenance_example(model, examples, ex_idx=0, head=0, figsize=(9, 8), cmap='Blues'):
    '''Teacher-forced cross-attention heatmap for one example, with the true provenance as red circles.

    Input positions are colored along the x-axis by region (root N, x, root E, y).
    '''
    src = examples['inputs'][ex_idx:ex_idx + 1]
    tgt = examples['targets'][ex_idx:ex_idx + 1]
    att = teacher_forced_cross_attention(model, src, tgt, head=head)[0].numpy()
    w = _symbols(src[0].tolist())
    out = _symbols(tgt[0].tolist())
    prov = np.array(kappa_provenance(w)) + 1
    reg = kappa_regions(w)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(att, cmap=cmap, aspect='auto')
    ax.scatter(prov, np.arange(len(prov)), s=60, facecolors='none', edgecolors='red', linewidths=1.5,
               label='true source position')
    for j, r in enumerate(reg):
        ax.axvspan(j + 0.5, j + 1.5, ymin=0, ymax=0.015, color=REGION_COLORS[r])
    ax.set_xticks(range(att.shape[1]))
    ax.set_xticklabels(['bos'] + [str(s) for s in w], fontsize=7)
    ax.set_yticks(range(att.shape[0]))
    ax.set_yticklabels([f'{t}:{s}' for t, s in enumerate(out)], fontsize=7)
    ax.set_xlabel('Input position (symbol); bottom strip = region: red root N, blue x, purple root E, orange y')
    ax.set_ylabel('Output step : symbol')
    ok = (att.argmax(-1) == prov).mean()
    ax.set_title(f'Cross-attention vs. κ provenance, example {ex_idx} (argmax = source on {ok:.0%} of steps)')
    ax.legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    return att, prov


@torch.no_grad()
def encoder_states(model, inputs):
    '''Encoder output [B, S, d] for a batch of token sequences.'''
    model.eval()
    h = model.src_pos_encoding(model.src_embedding(inputs))
    for layer in model.encoder:
        h = layer(h)
    return h.float().cpu()


def probe_regions(model, examples, num_examples=500, test_size=0.2, seed=42):
    '''Linear probe: encoder hidden state at input position -> region of that position under w = N x E y.

    Returns (accuracy, confusion_matrix, probe). BOS positions are excluded.
    '''
    inputs = examples['inputs'][:num_examples]
    H = encoder_states(model, inputs)[:, 1:, :]  # drop BOS
    X = H.reshape(-1, H.shape[-1]).numpy()
    y = np.concatenate([kappa_regions(_symbols(row.tolist())) for row in inputs])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    probe = LogisticRegression(max_iter=2000, random_state=seed).fit(X_tr, y_tr)
    pred = probe.predict(X_te)
    return accuracy_score(y_te, pred), confusion_matrix(y_te, pred, labels=range(4)), probe
