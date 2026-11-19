'''Mechanistic-interpretability helpers specific to the recursive involution sieve sigma_n.

sigma_n (src/utils/lemma_sieve.py) swaps (area, dinv), is an involution, fixes diagonal paths, and pairs
each path by the first applicable rule in a fixed priority order (lift, chain, lone, frame swap fx,
exchange fy, k-block swaps kb2..kb4). These helpers ask what the trained model has actually learned:

    - rule_lookup / examples_by_rule: which sieve rule produced each target
    - generate: greedy model outputs as 0/1 words
    - behaviour_report: exact accuracy per rule, plus whether outputs are valid Dyck words, swap
      (area, dinv) even when not exactly sigma(w), and whether the learned map is an involution
    - probe_statistics: linear probes from pooled encoder states to area(w) and dinv(w)

Token convention (same as Dataset): position 0 is BOS, token id = symbol + 2.
'''
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.utils.dyck_maps import area, dinv

RULE_ORDER = ['diagonal', 'lift1', 'lift2', 'lift3', 'lift4', 'chain', 'lone', 'loneback', 'fx', 'fy',
              'kb2', 'kb3', 'kb4']


def _symbols(token_row):
    '''Strip BOS and map token ids back to 0/1 symbols.'''
    return [int(t) - 2 for t in token_row[1:]]


def _is_dyck(w, n):
    h = 0
    for s in w:
        if s not in (0, 1):
            return False
        h += 1 if s == 1 else -1
        if h < 0:
            return False
    return h == 0 and len(w) == 2 * n


def rule_lookup(data):
    '''Map input word (tuple of 0/1) -> name of the sieve rule that paired it, from the raw data pickle.'''
    return {tuple(w): r for w, r in zip(data['inputs'].tolist(), data['rules'])}


def examples_by_rule(examples, lookup):
    '''Map rule name -> list of indices into examples['inputs'] whose target was produced by that rule.'''
    out = defaultdict(list)
    for i, row in enumerate(examples['inputs'].tolist()):
        out[lookup[tuple(_symbols(row))]].append(i)
    return dict(out)


@torch.no_grad()
def generate(model, src_tokens, n, batch_size=500):
    '''Greedy outputs of the model for a batch of token sequences, as 0/1 words (no BOS).'''
    model.eval()
    outs = []
    for b in range(0, len(src_tokens), batch_size):
        src = src_tokens[b:b + batch_size]
        bos = torch.zeros((len(src), 1), dtype=torch.long, device=src.device)
        gen = model.generate(src=src, tgt=bos, max_new_tokens=2 * n)
        outs += [[int(t) - 2 for t in row[-2 * n:]] for row in gen.tolist()]
    return outs


def behaviour_report(model, examples, lookup, n, num_examples=None):
    '''Exact accuracy per sieve rule, and structural properties of the model's outputs.

    Returns a dict with:
        accuracy: exact match with sigma(w)
        valid: output is a Dyck word of semilength n
        stat_swap: output is valid and (area, dinv)(output) == (dinv, area)(w)   [what sigma must satisfy]
        stat_swap_when_wrong: same, restricted to outputs that differ from sigma(w)
        involution: model(model(w)) == w, among valid outputs
        by_rule: {rule: (count, accuracy, stat_swap rate)}
    '''
    src = examples['inputs'][:num_examples] if num_examples else examples['inputs']
    tgt = examples['targets'][:len(src)]
    words = [_symbols(r) for r in src.tolist()]
    targets = [_symbols(r) for r in tgt.tolist()]
    preds = generate(model, src, n)

    exact = np.array([p == t for p, t in zip(preds, targets)])
    valid = np.array([_is_dyck(p, n) for p in preds])
    swap = np.array([v and (area(p), dinv(p)) == (dinv(w), area(w)) for p, w, v in zip(preds, words, valid)])

    # involution: feed valid outputs back into the model
    idx = np.where(valid)[0]
    back_src = torch.tensor([[0] + [s + 2 for s in preds[i]] for i in idx], dtype=torch.long, device=src.device)
    back = generate(model, back_src, n) if len(idx) else []
    invol = np.array([back[k] == words[i] for k, i in enumerate(idx)])

    rules = np.array([lookup[tuple(w)] for w in words])
    by_rule = {}
    for r in RULE_ORDER + sorted(set(rules) - set(RULE_ORDER)):
        m = rules == r
        if m.any():
            by_rule[r] = (int(m.sum()), float(exact[m].mean()), float(swap[m].mean()))
    return {
        'accuracy': float(exact.mean()),
        'valid': float(valid.mean()),
        'stat_swap': float(swap.mean()),
        'stat_swap_when_wrong': float(swap[~exact].mean()) if (~exact).any() else float('nan'),
        'involution': float(invol.mean()) if len(invol) else float('nan'),
        'by_rule': by_rule,
        'preds': preds,
    }


@torch.no_grad()
def encoder_states(model, inputs):
    '''Encoder output [B, S, d] for a batch of token sequences.'''
    model.eval()
    h = model.src_pos_encoding(model.src_embedding(inputs))
    for layer in model.encoder:
        h = layer(h)
    return h.float().cpu()


def probe_statistics(model, examples, num_examples=3000, test_size=0.2, seed=42):
    '''Linear probes from the mean-pooled encoder state of each input to area(w) and dinv(w).

    The statistics are treated as class labels (logistic regression), like the level probe in Analysis.ipynb.
    Returns {'area': accuracy, 'dinv': accuracy, 'area_baseline': ..., 'dinv_baseline': ...}, where the
    baseline is the accuracy of always predicting the most common value.
    '''
    inputs = examples['inputs'][:num_examples]
    X = encoder_states(model, inputs).mean(dim=1).numpy()
    words = [_symbols(r) for r in inputs.tolist()]
    out = {}
    for name, f in (('area', area), ('dinv', dinv)):
        y = np.array([f(w) for w in words])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        probe = LogisticRegression(max_iter=3000, random_state=seed).fit(X_tr, y_tr)
        out[name] = float((probe.predict(X_te) == y_te).mean())
        vals, counts = np.unique(y_tr, return_counts=True)
        out[name + '_baseline'] = float((y_te == vals[counts.argmax()]).mean())
    return out
