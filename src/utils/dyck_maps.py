'''Pure-Python bijections on Dyck words (no SageMath needed).

Words are lists of 0/1 with 1 = North (up) step and 0 = East (down) step,
the same convention as the dataset tensors.

Implements the operators from the "Dyck path - Nov 13th, 2026" notes:
    - area_sequence, first_return, prime_factors
    - bounce_blocks, segments, zip_product (u ⊠ v)
    - kappa:  κ(N x E y) = (NE · x) ⊠ y            (notes §5.2.3)
    - zeta:   Haglund's zeta map ζ                   (notes §5.1)
    - inner_zeta: Z(N x E y) = N ζ(x) E ζ(y), so that ζ = κ ∘ Z
    - rev:    read backward and swap 0/1 (Sage's area_dinv_to_bounce_area_map = rev ∘ ζ)
    - dyck_words(n): all Dyck words of semilength n, in the same order as Sage's DyckWords(n)
'''

N, E = 1, 0


def area_sequence(w):
    '''Area sequence (a_1..a_n): full cells between path and diagonal in each row, bottom to top.'''
    a, x, y = [], 0, 0
    for s in w:
        if s == N:
            a.append(y - x)
            y += 1
        else:
            x += 1
    return a


def first_return(w):
    '''Split non-empty w = N x E y at its first return to the diagonal; return (x, y).'''
    h = 0
    for i, s in enumerate(w):
        h += 1 if s == N else -1
        if h == 0:
            return list(w[1:i]), list(w[i + 1:])
    raise ValueError(f'not a Dyck word: {w}')


def prime_factors(w):
    '''Prime factorization w = p1 p2 ... pk (split at every return to the diagonal).'''
    out, h, start = [], 0, 0
    for i, s in enumerate(w):
        h += 1 if s == N else -1
        if h == 0:
            out.append(list(w[start:i + 1]))
            start = i + 1
    return out


def bounce_blocks(u):
    '''Bounce block sizes (c_0, ..., c_{r-1}) of u (notes Algorithm 1).'''
    a = area_sequence(u)
    n = len(a)
    b, Z = 0, [1]
    for i in range(2, n + 1):
        if b + 1 <= a[i - 1]:
            b += 1
        else:
            b = 0
            Z.append(i)
    Z.append(n + 1)
    return [Z[k + 1] - Z[k] for k in range(len(Z) - 1)]


def segments(u):
    '''Cut u into r+1 pieces of lengths c_{k-1} + c_k (c_{-1} = c_r = 0).'''
    c = bounce_blocks(u)
    c = [0] + c + [0]
    pieces, pos = [], 0
    for k in range(1, len(c)):
        L = c[k - 1] + c[k]
        pieces.append(list(u[pos:pos + L]))
        pos += L
    return pieces


def zip_product(u, v):
    '''Zip product u ⊠ v = s0(u) s0(v) s1(u) s1(v) ...'''
    if not u:
        return list(v)
    if not v:
        return list(u)
    s, t = segments(u), segments(v)
    m = max(len(s), len(t))
    s += [[]] * (m - len(s))
    t += [[]] * (m - len(t))
    out = []
    for a, b in zip(s, t):
        out += a + b
    return out


def kappa(w):
    '''κ(ε) = ε,  κ(N x E y) = (NE · x) ⊠ y.'''
    if not w:
        return []
    x, y = first_return(w)
    return zip_product([N, E] + x, y)


def kappa_regions(w):
    '''Label each input position of w = N x E y: 0 = root N, 1 = inside x, 2 = root E, 3 = inside y.'''
    x, _ = first_return(w)
    i = len(x) + 1  # index of the root E
    return [0] + [1] * len(x) + [2] + [3] * (len(w) - i - 1)


def kappa_provenance(w):
    '''For each output symbol of κ(w), the index (0-based) of the input symbol of w it is copied from.

    κ(N x E y) = (NE · x) ⊠ y only rearranges symbols: the output "N E" of NE·x are the root N and
    root E of w, the rest of NE·x is x, and the zip interleaves bounce segments of NE·x and y.
    Guarantees [w[j] for j in kappa_provenance(w)] == kappa(w).
    '''
    x, y = first_return(w)
    i = len(x) + 1
    u, u_idx = [N, E] + x, [0, i] + list(range(1, i))
    v_idx = list(range(i + 1, len(w)))
    if not y:
        return u_idx

    def cut(word, idx):
        out, pos = [], 0
        for piece in segments(word):
            out.append(idx[pos:pos + len(piece)])
            pos += len(piece)
        return out

    s, t = cut(u, u_idx), cut(y, v_idx)
    m = max(len(s), len(t))
    s += [[]] * (m - len(s))
    t += [[]] * (m - len(t))
    out = []
    for a, b in zip(s, t):
        out += a + b
    return out


def zeta(w):
    '''Haglund's zeta map: for k = 0..max(a)+1 scan a(w), write E for a_i = k-1 and N for a_i = k.'''
    if not w:
        return []
    a = area_sequence(w)
    out = []
    for k in range(max(a) + 2):
        for v in a:
            if v == k - 1:
                out.append(E)
            elif v == k:
                out.append(N)
    return out


def inner_zeta(w):
    '''Z(N x E y) = N ζ(x) E ζ(y).'''
    if not w:
        return []
    x, y = first_return(w)
    return [N] + zeta(x) + [E] + zeta(y)


def rev(w):
    '''Read backward and swap 0/1 (reflection across y = -x).'''
    return [1 - s for s in reversed(w)]


def dyck_words(n):
    '''All Dyck words of semilength n, ordered like Sage's DyckWords(n) (1010... first, 11..00 last).'''
    out = []

    def rec(prefix, ups, downs):
        if ups == n and downs == n:
            out.append(list(prefix))
            return
        # Try a down step (0) before an up step (1) to produce lexicographic order.
        if downs < ups:
            prefix.append(E); rec(prefix, ups, downs + 1); prefix.pop()
        if ups < n:
            prefix.append(N); rec(prefix, ups + 1, downs); prefix.pop()

    rec([], 0, 0)  # lexicographic with 0 < 1, i.e. 1010...10 first and 11..100..0 last
    return out


MAPS = {
    'kappa': kappa,
    'zeta': zeta,
    'rev_zeta': lambda w: rev(zeta(w)),  # == Sage area_dinv_to_bounce_area_map
}
