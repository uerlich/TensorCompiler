from typing import List, Tuple
import numpy as np

def hungarian_rect(cost: np.ndarray) -> Tuple[List[int], int]:
    m, n = cost.shape
    assert m <= n
    if m < n:
        pad = np.zeros((n-m, n), dtype=cost.dtype)
        sq = np.vstack([cost, pad])
    else:
        sq = cost.copy()
    row_assign, col_assign = hungarian_square(sq)
    cols = []
    total = 0
    for r in range(m):
        c = col_assign[r]
        cols.append(c)
        total += int(cost[r, c])
    return cols, total

def hungarian_square(cost: np.ndarray):
    n, m = cost.shape
    assert n == m
    C = cost.copy().astype(float)
    C -= C.min(axis=1, keepdims=True)
    C -= C.min(axis=0, keepdims=True)

    n = C.shape[0]
    starred = np.zeros_like(C, dtype=bool)
    primed = np.zeros_like(C, dtype=bool)
    covered_rows = np.zeros(n, dtype=bool)
    covered_cols = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if C[i,j] == 0 and not covered_rows[i] and not covered_cols[j]:
                starred[i,j] = True
                covered_rows[i] = True
                covered_cols[j] = True
    covered_rows[:] = False
    covered_cols[:] = False

    def cover_columns_with_starred_zeroes():
        for j in range(n):
            if starred[:,j].any():
                covered_cols[j] = True
    def find_a_zero():
        for i in range(n):
            if not covered_rows[i]:
                for j in range(n):
                    if not covered_cols[j] and C[i,j] == 0 and not starred[i,j]:
                        return i,j
        return None
    def find_star_in_row(r):
        for j in range(n):
            if starred[r,j]: return j
        return None
    def find_star_in_col(c):
        for i in range(n):
            if starred[i,c]: return i
        return None
    def find_prime_in_row(r):
        for j in range(n):
            if primed[r,j]: return j
        return None
    def augment_path(path):
        for (i,j) in path:
            starred[i,j] = not starred[i,j]
        primed[:] = False
        covered_rows[:] = False
        covered_cols[:] = False
    def smallest_uncovered_value():
        mv = float('inf')
        for i in range(n):
            if not covered_rows[i]:
                for j in range(n):
                    if not covered_cols[j]:
                        if C[i,j] < mv: mv = C[i,j]
        return 0.0 if mv == float('inf') else mv

    cover_columns_with_starred_zeroes()
    while covered_cols.sum() < n:
        z = find_a_zero()
        while z is None:
            d = smallest_uncovered_value()
            for i in range(n):
                if covered_rows[i]: C[i,:] += d
            for j in range(n):
                if not covered_cols[j]: C[:,j] -= d
            z = find_a_zero()
        i,j = z
        primed[i,j] = True
        sc = find_star_in_row(i)
        if sc is None:
            path = [(i,j)]
            c = j
            r = find_star_in_col(c)
            while r is not None:
                path.append((r,c))
                p = find_prime_in_row(r)
                path.append((r,p))
                c = p
                r = find_star_in_col(c)
            augment_path(path)
            cover_columns_with_starred_zeroes()
        else:
            covered_rows[i] = True
            covered_cols[sc] = False

    row_assign = [-1]*n
    col_assign = [-1]*n
    for i in range(n):
        for j in range(n):
            if starred[i,j]:
                row_assign[i] = j
                col_assign[i] = j
                break
    return row_assign, col_assign
