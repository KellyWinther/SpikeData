import numpy as np
import pandas as pd
from tqdm import tqdm
import os


def _preprocess_lists(
    l1: list,
    l2: list,
) -> tuple:
    """
    Convert inputs to Python ints and ensure the shorter list is first.

    Parameters
    ----------
    l1, l2 :: list
        Lists of integers or values convertible to integers.

    Returns
    -------
    tuple
        (l_shorter, l_longer)
    """
    l1 = list(map(int, l1))
    l2 = list(map(int, l2))

    return (l1, l2) if len(l1) <= len(l2) else (l2, l1)


def _compute_pow_base(
    n: int,
    base: int,
    mod: int,
) -> list:
    """
    Precompute powers of the base modulo `mod`.

    Parameters
    ----------
    n :: int
        Maximum exponent needed.
    base :: int
        Rolling hash base.
    mod :: int
        Rolling hash modulus.

    Returns
    -------
    pow_base :: list
        pow_base[i] = base^i % mod
    """
    pow_base = [1] * (n + 1)
    for i in range(1, n + 1):
        pow_base[i] = (pow_base[i - 1] * base) % mod
    return pow_base


def _prefix_hash(
    arr: list,
    base: int,
    mod: int,
) -> list:
    """
    Compute prefix hashes for an array.

    Parameters
    ----------
    arr :: list
        Input list of integers.
    base :: int
        Rolling hash base.
    mod :: int
        Rolling hash modulus.

    Returns
    -------
    list
        Prefix hash array where H[i] hashes arr[:i].
    """
    H = [0] * (len(arr) + 1)
    for i, v in enumerate(arr):
        H[i + 1] = (H[i] * base + (v + 1)) % mod
    return H


def _get_hash(
    H: list,
    i: int,
    L: int,
    pow_base: list,
    mod: int,
) -> int:
    """
    Retrieve hash for subarray of length L starting at index i.

    Parameters
    ----------
    H :: list
        Prefix hash array.
    i :: int
        Start index.
    L :: int
        Length of the window.
    pow_base :: list
        Precomputed powers of base modulo mod.
    mod :: int
        Modulus.

    Returns
    -------
    int
        Hash value.
    """
    return (H[i + L] - (H[i] * pow_base[L]) % mod) % mod


def _has_match(
    L: int,
    n1: int,
    n2: int,
    H1: list,
    H2: list,
    pow_base: list,
    mod: int,
) -> bool:
    """
    Determine whether a contiguous match of length L exists in both lists.

    Parameters
    ----------
    L : int
        Length to test.
    n1, n2 : int
        Lengths of the two lists.
    H1, H2 : list
        Prefix hash arrays.
    pow_base : list
        Precomputed powers of the hash base.
    mod : int
        Modulus.

    Returns
    -------
    bool
        True if a match of length L exists, False otherwise.
    """
    if L == 0:
        return True
    if L > n1:
        return False

    seen = {
        _get_hash(H1, i, L, pow_base, mod)
        for i in range(n1 - L + 1)
    }

    for j in range(n2 - L + 1):
        if _get_hash(H2, j, L, pow_base, mod) in seen:
            return True

    return False


def _recover_sequence(
    L: int,
    l1: list,
    l2: list,
    H1: list,
    H2: list,
    n1: int,
    n2: int,
    pow_base: list,
    mod: int,
) -> list:
    """
    Recover the actual longest common contiguous subsequence of length L.

    Parameters
    ----------
    L : int
        Length of desired subsequence.
    l1, l2 : list
        Original lists.
    H1, H2 : list
        Prefix hash arrays.
    n1, n2 : int
        Lengths of the lists.
    pow_base : list
        Precomputed powers of the base.
    mod : int
        Modulus.

    Returns
    -------
    list
        The overlapping subsequence, or [] if none (unlikely unless collision).
    """
    if L == 0:
        return []

    # Build hash table for l1 windows
    seen = {}
    for i in range(n1 - L + 1):
        h = _get_hash(H1, i, L, pow_base, mod)
        seen.setdefault(h, []).append(i)

    # Check matches in l2
    for j in range(n2 - L + 1):
        h = _get_hash(H2, j, L, pow_base, mod)
        if h in seen:
            for i in seen[h]:
                if l1[i:i + L] == l2[j:j + L]:
                    return l1[i:i + L]

    return []  # Should not happen unless hash collision


def lccs_rabin_karp(
    l1: list,
    l2: list,
    normalize: bool = True,
    return_sequence: bool = False
) -> tuple[int | float, list]:
    """
    Finds the longest contiguous sequence of integers shared between
    two lists using Rabin–Karp rolling hashing with binary search.

    Parameters
    ----------
    l1, l2 : list
        Input lists of ints (or values convertible to int).
    normalize : bool, optional
        If True, return length normalized by length of the shorter list.
    return_sequence : bool, optional
        If True, also return the subsequence itself.

    Returns
    -------
    int or float, or (length, list)
        If return_sequence is False, returns the length.
        If True, returns (length, subsequence).
    """

    # Preprocess and ensure l1 is shorter
    l1, l2 = _preprocess_lists(l1, l2)
    n1, n2 = len(l1), len(l2)

    if n1 == 0:
        return (0, []) if return_sequence else 0

    # Hash parameters
    base = 257
    mod = (1 << 61) - 1

    pow_base = _compute_pow_base(n1, base, mod)
    H1 = _prefix_hash(l1, base, mod)
    H2 = _prefix_hash(l2, base, mod)

    # Binary search for max L
    lo, hi = 0, n1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _has_match(mid, n1, n2, H1, H2, pow_base, mod):
            lo = mid
        else:
            hi = mid - 1

    L = lo

    if not return_sequence:
        return L / n1 if normalize else L

    seq = _recover_sequence(L, l1, l2, H1, H2, n1, n2, pow_base, mod)
    length_out = L / n1 if normalize else L
    return length_out, seq
