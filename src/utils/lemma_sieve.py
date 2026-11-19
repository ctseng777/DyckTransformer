"""A recursive involution sieve that swaps (area, dinv) on Dyck paths.

Self-contained reference implementation of the map sigma_n from the report
"Exact local moves for the q,t-Catalan symmetry: a recursive involution sieve"
(lemma_sieve.tex).  Pure Python 3, no dependencies.

    python lemma_sieve.py 9                # check n = 1..9 (about 15 seconds)
    python lemma_sieve.py 10               # about 2 minutes
    python lemma_sieve.py 11 --workers 10  # about 20 minutes on 10 cores

For every n up to N it computes sigma_n on all Dyck paths, checks that it swaps
(area, dinv) and is an involution, and reports how many non-diagonal paths are
paired and by which rule.  The sieve is total for n <= 11.  At n = 12 it leaves
28 paths unpaired.

Conventions
-----------
A Dyck path of semilength n is its area sequence v = (v_0, ..., v_{n-1}), a
tuple with v_0 = 0 and 0 <= v_{i+1} <= v_i + 1.

    area(v) = sum(v),   dinv(v) = #{i < j : v_i - v_j in {0, 1}}.

Every rule below is a partial map R with st(R(v)) = (dinv v, area v), where
st = (area, dinv), restricted to the points where R(R(v)) = v.  The sieve
(Lemma 6) pairs v at the first rule in a fixed priority order whose image is
not already paired by an earlier rule.  Pieces of a path (frame and blocks)
are swapped by the sieve itself at smaller sizes, so the definition is an
induction on n.

Section numbers in the comments refer to the report.
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from collections import Counter

# Largest number of blocks tried in the k-block frame swap (stages k = 2..KMAX).
KMAX = 4

# Global memo tables.  The map is deterministic, so the cache never changes a result.
_MEMO: dict = {}
# Final answers sigma(v) for sizes that are already complete (filled by main()).
_SOLVED: dict = {}
_UNPAIRED = "unpaired"


def _cached(tag, key, thunk):
    k = (tag, key)
    if k not in _MEMO:
        _MEMO[k] = thunk()
    return _MEMO[k]


# ---------------------------------------------------------------------------
# Dyck paths as area sequences
# ---------------------------------------------------------------------------

def all_paths(n):
    """Every area sequence of length n."""
    def rec(prefix):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for x in range(prefix[-1] + 2):
            prefix.append(x)
            yield from rec(prefix)
            prefix.pop()
    if n == 0:
        yield ()
    else:
        yield from rec([0])


def is_area_seq(v):
    return all(v[i + 1] <= v[i] + 1 for i in range(len(v) - 1))


def area(v):
    return sum(v)


def dinv(v):
    n = len(v)
    return sum(1 for i in range(n) for j in range(i + 1, n) if v[i] - v[j] in (0, 1))


def st(v):
    """(area, dinv)."""
    return _cached("st", v, lambda: (area(v), dinv(v)))


# ---------------------------------------------------------------------------
# Partition coordinates and the nu operator (Lee--Li--Loehr), Section 1
# ---------------------------------------------------------------------------
#
# The cells between the path and the staircase form a partition gamma inside
# Delta_n = (n-1, ..., 1):  gamma_i = (n-1-i) - v_{n-1-i}.  Then
# area_n = C(n,2) - |gamma| and dinv does not depend on n.  One nu-step adds
# one cell and one unit of dinv, i.e. it changes (area, dinv) by (-1, +1).

def normalize(g):
    """Strip trailing zeros; raise ValueError unless g is a partition."""
    g = list(g)
    while g and g[-1] == 0:
        g.pop()
    if any(g[i] < g[i + 1] for i in range(len(g) - 1)) or (g and g[-1] < 0):
        raise ValueError(f"not a partition: {g}")
    return tuple(g)


def to_partition(v):
    n = len(v)
    return normalize([(n - 1 - i) - v[n - 1 - i] for i in range(n)])


def fits(g, n):
    """gamma fits inside the staircase Delta_n."""
    return len(g) < n and all(gi <= n - 1 - i for i, gi in enumerate(g))


def from_partition(g, n):
    """The area sequence of length n with partition g, or None if g does not fit."""
    if not fits(g, n):
        return None
    pad = tuple(g) + (0,) * (n - len(g))
    return tuple(k - pad[n - 1 - k] for k in range(n))


def nu(g):
    """LLL: nu(gamma) = (l+1, g_1 - 1, ..., g_l - 1), defined when g_1 <= l + 2."""
    ell = len(g)
    if g and g[0] > ell + 2:
        raise ValueError("nu undefined")
    return normalize([ell + 1] + [x - 1 for x in g])


def nu_inverse(g):
    if not g:
        raise ValueError("() is not in the image of nu")
    ell = g[0] - 1
    rest = list(g[1:])
    if ell < 0 or len(rest) > ell:
        raise ValueError("not in the image of nu")
    cand = normalize([x + 1 for x in rest + [0] * (ell - len(rest))])
    try:
        if nu(cand) == g:
            return cand
    except ValueError:
        pass
    raise ValueError("not in the image of nu")


# Second-order extension of nu, used where nu itself is undefined.  It is
# given by a small set of templates on the "quasi Dyck vector" of gamma, and
# changes the statistics exactly like nu (checked on all 849 applicable
# partitions for n <= 10, and on every pair the sieve produces up to n = 12).

def _qdv(g, n):
    pad = tuple(g) + (0,) * (n - len(g))
    return tuple(i - pad[n - 1 - i] for i in range(n))


def _qdv_partition(v):
    return normalize(tuple(i - v[i] for i in range(len(v) - 1, -1, -1)))


def _nu2(g):
    if not g or g[0] <= len(g):
        raise ValueError("no second-order successor")
    v = _qdv(g, g[0])
    if v[:3] != (0, 1, 2) or v[-1] != -1 or max(v) > 2:
        raise ValueError("outside second-order templates")
    a = 0
    while 2 + a < len(v) and v[2 + a] == 2:
        a += 1
    b = 0
    while b < len(v) and v[-1 - b] == -1:
        b += 1
    if a > b:
        h = b + 1
        w = (0,) * h + (1,) + v[2 + h:len(v) - b] + (1,) * h
    else:
        w = (0,) * (a + 1) + v[2 + a:len(v) - a] + (0,) + (1,) * a
    return _qdv_partition(w)


def _nu2_inverse(g):
    v = _qdv(g, len(g) + 1)
    h = 0
    while h < len(v) and v[h] == 0:
        h += 1
    k = 0
    while k < len(v) and v[-1 - k] == 1:
        k += 1
    if h < 2 or k < 1 or max(v) > 2:
        raise ValueError("outside second-order inverse templates")
    if k >= h:
        if v[h] != 1:
            raise ValueError("inverse template a")
        w = (0, 1) + (2,) * h + v[h + 1:len(v) - h] + (-1,) * (h - 1)
    else:
        if v[-k - 1] != 0:
            raise ValueError("inverse template b")
        w = (0, 1) + (2,) * k + v[k + 1:len(v) - k - 1] + (-1,) * k
    q = _qdv_partition(w)
    if _nu2(q) != tuple(g):
        raise ValueError("inverse validation")
    return q


def _nu_step(g, sign):
    """One step along the extended nu-chain: forward (sign > 0) or backward."""
    for f in ((nu, _nu2) if sign > 0 else (nu_inverse, _nu2_inverse)):
        try:
            return f(g)
        except ValueError:
            pass
    return None


def shift(v, amount):
    """Slide v by `amount` steps along its nu-chain (each step: area -1, dinv +1).
    None if the chain ends or leaves the staircase."""
    def go():
        g = to_partition(v)
        for _ in range(abs(amount)):
            g = _nu_step(g, amount)
            if g is None:
                return None
        return from_partition(g, len(v))
    return _cached("shift", (v, amount), go)


# ---------------------------------------------------------------------------
# Rules, Section 2.  Each takes an area sequence and returns one (or None).
# ---------------------------------------------------------------------------

def rule_lift(v, m):
    """Lift (Lemma 4).  If gamma(v) fits in Delta_m, m < n: swap it at size m, then
    slide Delta = C(n,2) - C(m,2) steps along nu.  The other direction slides
    back first, then swaps at size m."""
    n = len(v)
    if not 1 <= m < n:
        return None
    delta = n * (n - 1) // 2 - m * (m - 1) // 2
    g = to_partition(v)
    if fits(g, m):
        u = swap(from_partition(g, m))
        if u is not None:
            w = from_partition(to_partition(u), n)
            return None if w is None else shift(w, delta)
    w = shift(v, -delta)
    if w is not None and fits(to_partition(w), m):
        u = swap(from_partition(to_partition(w), m))
        if u is not None:
            return from_partition(to_partition(u), n)
    return None


def rule_chain(v):
    """Chain (Lemma 5): slide area - dinv steps along nu."""
    a, d = st(v)
    return shift(v, a - d)


def rule_lone_front(v):
    """Lone front return (twist, Lemma 3): v = (0, 0, C + 1) has area C + n - 2 and
    dinv C + 1.  Partner (0, 0, C' + 1), C' = swap(C) slid n - 3 steps along nu."""
    n = len(v)
    if n < 4 or v[1] != 0 or min(v[2:]) < 1:
        return None
    c = tuple(x - 1 for x in v[2:])
    a, d = st(v)
    if a > d:
        s = swap(c)
        c2 = None if s is None else shift(s, n - 3)
    else:
        b = shift(c, -(n - 3))
        c2 = None if b is None else swap(b)
    if c2 is None:
        return None
    return (0, 0) + tuple(x + 1 for x in c2)


def rule_lone_back(v):
    """Lone back return (twist, Lemma 3): v = (0, C + 1, 0), dinv = dinv C + 1 + z(C),
    z = number of zeros.  Partner (0, D + 1, 0), D = swap(C) slid n - 3 - z(C) steps;
    exact iff z(D) = z(C)."""
    n = len(v)
    if n < 4 or v[-1] != 0 or min(v[1:-1]) < 1:
        return None
    c = tuple(x - 1 for x in v[1:-1])
    zc = c.count(0)
    a, d = st(v)
    if a > d:
        s = swap(c)
        e = None if s is None else shift(s, n - 3 - zc)
    else:
        b = shift(c, -(n - 3 - zc))
        e = None if b is None else swap(b)
    if e is None or e.count(0) != zc:
        return None
    return (0,) + tuple(x + 1 for x in e) + (0,)


# --- Frame swap (fx) and exchange (fy): Lemma 2 with k = 1 -----------------
#
# Cut v = P ++ (E + c) ++ Q: a block E (an area sequence) at level c, at
# positions [i, j), and the frame P ++ Q.  The level term is c|E| and the cross
# term X is the number of dinv pairs between the block and the frame.

def _cross_total(P, Q, block):
    """Dinv pairs between a block (absolute values) and the frame parts before (P) / after (Q) it."""
    return sum(sum(1 for y in P if y - x in (0, 1)) + sum(1 for y in Q if x - y in (0, 1))
               for x in block)


def _hole_cross(outer, blk):
    """For each hole h = 0..len(outer): the cross dinv of blk inserted into outer at h."""
    top = max(max(outer), max(blk)) + 2
    cnt = [0] * (top + 1)
    for x in blk:
        cnt[x] += 1
    after = sum(cnt[y] + cnt[y + 1] for y in outer)
    out = [after]
    for y in outer:
        after += (cnt[y - 1] if y else 0) - cnt[y + 1]
        out.append(after)
    return out


def _cuts(v, min_len):
    """All one-block cuts (i, j, c, E, frame, X) of v."""
    n = len(v)
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            m = j - i
            if m >= n - 1 or m < min_len:
                continue
            c = v[i]
            e = tuple(x - c for x in v[i:j])
            if min(e) < 0:
                continue
            frame = v[:i] + v[j:]
            if not is_area_seq(frame):
                continue
            yield i, j, c, e, frame, _cross_total(v[:i], v[j:], v[i:j])


def frame_swap_partners(v):
    """fx: T = swap(frame) with swap(E) + c2 inserted at hole h.  Exact iff
    c2 |E| = X(v) (area side) and X(T) = c |E| (dinv side)."""
    def go():
        out = []
        for i, j, c, e, frame, X in _cuts(v, 1):
            m = j - i
            if X % m:
                continue
            c2 = X // m
            f, g = swap(e), swap(frame)
            if f is None or g is None:
                continue
            blk = tuple(x + c2 for x in f)
            hc = _hole_cross(g, blk)
            for h in range(1, len(g) + 1):
                if hc[h] != c * m or c2 > g[h - 1] + 1 or (h < len(g) and g[h] > blk[-1] + 1):
                    continue
                out.append(((i, j, h), g[:h] + blk + g[h:]))
        return tuple(out)
    return _cached("fx", v, go)


def exchange_partners(v):
    """fy: swap(E) becomes the new frame, swap(frame) + c2 the new block at hole h.
    Exact iff c2 |frame| = X(v) and X(T) = c |E|."""
    def go():
        n = len(v)
        out = []
        for i, j, c, e, frame, X in _cuts(v, 2):
            m = j - i
            if X % (n - m):
                continue
            c2 = X // (n - m)
            f, g = swap(e), swap(frame)
            if f is None or g is None:
                continue
            blk = tuple(x + c2 for x in g)
            hc = _hole_cross(f, blk)
            for h in range(1, m + 1):
                if hc[h] != c * m or c2 > f[h - 1] + 1 or (h < m and f[h] > blk[-1] + 1):
                    continue
                out.append(((i, j, h), f[:h] + blk + f[h:]))
        return tuple(out)
    return _cached("fy", v, go)


def frame_table(v):
    """The 1-block instances of v as single-valued rules, indexed by keys that are
    invariant under inversion: (family, block length, min(i, h), max(i, h)).  A key
    is kept only where it gives v exactly one target."""
    def go():
        n = len(v)
        tab = {}
        for (i, j, h), t in frame_swap_partners(v):
            tab.setdefault(("fx", j - i, min(i, h), max(i, h)), set()).add(t)
        for (i, j, h), t in exchange_partners(v):
            m = min(j - i, n - (j - i))
            tab.setdefault(("fy", m, min(i, h), max(i, h)), set()).add(t)
        return {k: next(iter(ts)) for k, ts in tab.items() if len(ts) == 1}
    return _cached("frame_table", v, go)


# --- k-block frame swap with role permutation: Lemma 2 ----------------------

def kblock_partners(v, k):
    """Cut k blocks out of v: v = R0 (E1+c1) R1 ... (Ek+ck) Rk, frame A0 = R0 R1 ... Rk,
    blocks Ab = Eb.  Swap every piece at smaller size, pick a permutation pi of the
    k+1 pieces, and rebuild T = swap(A[pi 0]) with the blocks swap(A[pi b]) + d_b
    inserted in order at holes h_1 <= ... <= h_k.  With S(x) = cross dinv between
    the blocks of x and everything else:
        area(T) = dinv(v)  iff  sum_b d_b |A[pi b]| = S(v),
        dinv(T) = area(v)  iff  S(T) = sum_b c_b |E_b|."""
    N = len(v)
    out = []

    def block_cross(t, spans):
        owner = [-1] * len(t)
        for b, (i, j) in enumerate(spans):
            for x in range(i, j):
                owner[x] = b
        return sum(1 for i in range(len(t)) for j in range(i + 1, len(t))
                   if t[i] - t[j] in (0, 1) and (owner[i] >= 0 or owner[j] >= 0)
                   and owner[i] != owner[j])

    def spans_from(start, kk):
        if kk == 0:
            yield ()
            return
        for i in range(start, N):
            for j in range(i + 1, N + 1):
                for rest in spans_from(j, kk - 1):
                    yield ((i, j),) + rest

    for spans in spans_from(1, k):
        if sum(j - i for i, j in spans) > N - 1:
            continue
        es = [tuple(x - v[i] for x in v[i:j]) for i, j in spans]
        if any(min(e) < 0 for e in es):
            continue
        inside = {x for i, j in spans for x in range(i, j)}
        R = tuple(v[x] for x in range(N) if x not in inside)
        if not is_area_seq(R):
            continue
        S = block_cross(v, spans)
        target = sum(v[i] * (j - i) for i, j in spans)
        sw = [swap(x) for x in [R] + es]
        if None in sw:
            continue
        for perm in itertools.permutations(range(k + 1)):
            g, fs = sw[perm[0]], [sw[x] for x in perm[1:]]
            lens = [len(f) for f in fs]

            def place(b, lo, prefix, budget, holes, pos):
                # prefix = T built so far; g[lo:] is the part of the new frame not yet placed.
                if b == k:
                    if lo < len(g) and g[lo] > prefix[-1] + 1:
                        return
                    t = prefix + g[lo:]
                    if budget == 0 and block_cross(t, pos) == target:
                        out.append(((spans, perm, holes), t))
                    return
                for h in range(max(lo, 1), len(g) + 1):
                    if h > lo and prefix and g[lo] > prefix[-1] + 1:
                        break
                    pre = prefix + g[lo:h]
                    for d in range(0, min(pre[-1] + 1, budget // lens[b]) + 1):
                        if b == k - 1 and d * lens[b] != budget:
                            continue
                        blk = tuple(x + d for x in fs[b])
                        place(b + 1, h, pre + blk, budget - d * lens[b], holes + (h,),
                              pos + ((len(pre), len(pre) + lens[b]),))

            place(0, 0, (), S, (), ())
    return out


def kblock_table(v, k):
    """k-block instances of v as single-valued rules.  The key of an instance is the
    sorted pair {its description, the description of the inverse instance seen from T},
    so v -> T and T -> v share a key."""
    def go():
        N = len(v)
        tab = {}
        for (spans, perm, holes), t in kblock_partners(v, k):
            lens_v = [N - sum(j - i for i, j in spans)] + [j - i for i, j in spans]
            fl = [lens_v[x] for x in perm[1:]]
            pos, sh = [], 0
            for b, h in enumerate(holes):
                pos.append((h + sh, h + sh + fl[b]))
                sh += fl[b]
            inv = tuple(perm.index(i) for i in range(len(perm)))
            hv, sh = [], 0
            for i, j in spans:
                hv.append(i - sh)
                sh += j - i
            key = tuple(sorted([(tuple(spans), tuple(perm), tuple(holes)),
                                (tuple(pos), inv, tuple(hv))]))
            tab.setdefault(key, set()).add(t)
        return {key: next(iter(ts)) for key, ts in tab.items()
                if len(ts) == 1 and next(iter(ts)) != v}
    return _cached(("kb", k), v, go)


# ---------------------------------------------------------------------------
# The sieve (Lemma 6, Theorem 1)
# ---------------------------------------------------------------------------
#
# Priority positions are tuples, compared lexicographically:
#   stage 1:  (1, (r,))            r = 0..6: lift m=n-1..n-4, chain, lone front, lone back
#             (1, (7, -m, a, b))   frame swap fx, widest block first
#             (1, (8, -m, a, b))   exchange fy
#   stage k:  (k, key)             k-block frame swaps, k = 2..KMAX

_FIXED = [
    ("lift1", lambda v: rule_lift(v, len(v) - 1)),
    ("lift2", lambda v: rule_lift(v, len(v) - 2)),
    ("lift3", lambda v: rule_lift(v, len(v) - 3)),
    ("lift4", lambda v: rule_lift(v, len(v) - 4)),
    ("chain", rule_chain),
    ("lone", rule_lone_front),
    ("loneback", rule_lone_back),
]


def rule_name(pos):
    stage, sub = pos
    if stage > 1:
        return f"kb{stage}"
    if sub[0] < len(_FIXED):
        return _FIXED[sub[0]][0]
    return "fx" if sub[0] == len(_FIXED) else "fy"


def _stage1_positions(v):
    def go():
        fixed = [(1, (r,)) for r in range(len(_FIXED))]
        frames = [(1, (len(_FIXED) + (fam == "fy"), -m, a, b)) for (fam, m, a, b) in frame_table(v)]
        return sorted(fixed + frames)
    return _cached("stage1", v, go)


def _apply_stage1(sub, v):
    if sub[0] < len(_FIXED):
        return _FIXED[sub[0]][1](v)
    fam = "fx" if sub[0] == len(_FIXED) else "fy"
    return frame_table(v).get((fam, -sub[1], sub[2], sub[3]))


def rule_at(pos, v):
    """The rule at position pos, as a partial involution that swaps (area, dinv)."""
    stage, sub = pos
    if stage == 1:
        def go():
            try:
                q = _apply_stage1(sub, v)
                if q is None or len(q) != len(v) or st(q) != st(v)[::-1]:
                    return None
                return q if _apply_stage1(sub, q) == v else None
            except (ValueError, IndexError, KeyError):
                return None
        return _cached(("rule", sub), v, go)
    q = kblock_table(v, stage).get(sub)
    if q is None or st(q) != st(v)[::-1]:
        return None
    if assign(q, (stage,)) is not None:  # q already paired before stage k: skip building its table
        return None
    return q if kblock_table(q, stage).get(sub) == v else None


def assign(v, bound):
    """Partner of v using only rules at positions < bound (None if there is none).
    v is paired at the first position whose rule maps it to a path that no earlier
    position has paired.  The memo records how far the scan has got."""
    key = ("scan", v)
    idx, partner, at = _MEMO.get(key, (0, None, None))
    if partner is not None:
        return partner if at < bound else None
    cands = _stage1_positions(v)
    stage, i = 1, idx
    while True:
        if i >= len(cands):
            # Extend lazily: v's k-block table is built only once the scan reaches stage k.
            stage += 1
            if stage > KMAX or (stage,) >= bound:
                break
            cands = cands + [(stage, k) for k in sorted(kblock_table(v, stage))]
            continue
        pos = cands[i]
        if pos >= bound:
            break
        r = rule_at(pos, v)
        if r is not None and assign(r, pos) is None:
            _MEMO[key] = (i + 1, r, pos)
            return r
        i += 1
        _MEMO[key] = (i, None, None)
    return None


def swap(v):
    """sigma_n(v): the partner of v, v itself if area = dinv, None if unpaired."""
    v = tuple(v)
    if v in _SOLVED:
        r = _SOLVED[v]
        return None if r == _UNPAIRED else r
    a, d = st(v)
    if a == d:
        return v
    return assign(v, (KMAX + 1,))


def pairing_rule(v):
    """Name of the rule that paired v (call swap(v) first)."""
    entry = _MEMO.get(("scan", tuple(v)))
    return None if entry is None or entry[1] is None else rule_name(entry[2])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _solve_chunk(paths):
    out = []
    for v in paths:
        a, d = st(v)
        if a == d:
            out.append((v, v, "diagonal"))
        else:
            out.append((v, swap(v), pairing_rule(v)))
    return out


def _solve_chunk_big_stack(paths):
    return _with_big_stack(_solve_chunk, paths)


def _with_big_stack(fn, *args):
    """The sieve recurses deeply; run fn in a thread with a large C stack."""
    sys.setrecursionlimit(200000)
    threading.stack_size(512 * 1024 * 1024)
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", fn(*args)))
    t.start()
    t.join()
    return box["r"]


def solve_size(n, workers=1):
    """{v: (partner or None, rule)} for every path of semilength n."""
    paths = list(all_paths(n))
    if workers <= 1 or n < 8:
        results = _with_big_stack(_solve_chunk, paths)
    else:
        import multiprocessing as mp
        nchunks = workers * 4
        size = -(-len(paths) // nchunks)
        chunks = [paths[i:i + size] for i in range(0, len(paths), size)]
        # fork: workers inherit the answers for all smaller sizes.
        with mp.get_context("fork").Pool(workers) as pool:
            results = [r for part in pool.imap_unordered(_solve_chunk_big_stack, chunks) for r in part]
    return {v: (q, rule) for v, q, rule in results}


def check(n, res):
    """Verify that the map swaps (area, dinv) and is an involution; return counters."""
    nondiag = paired = 0
    usage = Counter()
    for v, (q, rule) in res.items():
        a, d = st(v)
        if a == d:
            assert q == v
            continue
        nondiag += 1
        if q is None:
            continue
        assert st(q) == (d, a), f"statistics not swapped: {v} -> {q}"
        assert res[q][0] == v, f"not an involution: {v} -> {q} -> {res[q][0]}"
        paired += 1
        usage[rule] += 1
    return nondiag, paired, usage


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("N", type=int, nargs="?", default=9, help="largest semilength to check")
    ap.add_argument("--workers", type=int, default=1, help="processes to use for n >= 8")
    ap.add_argument("--kmax", type=int, default=KMAX, help="largest block count k (default 4)")
    args = ap.parse_args()
    globals()["KMAX"] = args.kmax

    print(f"{'n':>3} {'non-diagonal':>13} {'paired':>9} {'time':>8}  rule usage (paths)")
    for n in range(1, args.N + 1):
        t0 = time.time()
        res = solve_size(n, args.workers)
        nondiag, paired, usage = check(n, res)
        for v, (q, _) in res.items():
            _SOLVED[v] = _UNPAIRED if q is None else q
        use = ", ".join(f"{k} {c}" for k, c in usage.most_common())
        print(f"{n:>3} {nondiag:>13} {paired:>9} {time.time() - t0:>7.1f}s  {use}", flush=True)
        if paired < nondiag:
            left = sorted(v for v, (q, _) in res.items() if q is None)
            print(f"    {len(left)} unpaired, e.g. {left[:4]}")


if __name__ == "__main__":
    main()
