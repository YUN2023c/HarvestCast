# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""纯 Python 数值基座。

设计约束：
- 不依赖 numpy / pandas / scipy，任何 Python 3.9+ 环境可直接运行；
- 只用标准库 math / statistics / random，保证「零重依赖」与可复现；
- 所有函数均为无状态纯函数，方便单测。

若安装可选依赖（requirements-optional.txt），模型层会自动切到
statsmodels / LightGBM 的加速实现，本模块仍作为兜底。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

# --------------------------------------------------------------------------
# 描述统计
# --------------------------------------------------------------------------


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def variance(xs: Sequence[float], ddof: int = 1) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return float("nan")
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def stdev(xs: Sequence[float], ddof: int = 1) -> float:
    v = variance(xs, ddof)
    return math.sqrt(v) if v == v else float("nan")


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mad(xs: Sequence[float]) -> float:
    """中位数绝对偏差，抗离群。"""
    m = median(xs)
    return median([abs(x - m) for x in xs])


def quantile(xs: Sequence[float], q: float) -> float:
    """线性插值分位数（与 numpy 默认 method='linear' 一致）。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def zscore(xs: Sequence[float]) -> List[float]:
    m, s = mean(xs), stdev(xs)
    if not s or s != s:
        return [0.0 for _ in xs]
    return [(x - m) / s for x in xs]


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


# --------------------------------------------------------------------------
# 变换
# --------------------------------------------------------------------------


def diff(xs: Sequence[float]) -> List[float]:
    return [xs[i] - xs[i - 1] for i in range(1, len(xs))]


def log_series(xs: Sequence[float], floor: float = 1e-9) -> List[float]:
    return [math.log(max(x, floor)) for x in xs]


def exp_series(xs: Sequence[float]) -> List[float]:
    return [math.exp(x) for x in xs]


def cumsum(xs: Sequence[float], start: float = 0.0) -> List[float]:
    out, acc = [], start
    for x in xs:
        acc += x
        out.append(acc)
    return out


def pct(a: float, b: float) -> float:
    """b 相对 a 的变化率。"""
    return (b - a) / a if a else float("nan")


# --------------------------------------------------------------------------
# 正态分布
# --------------------------------------------------------------------------


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """正态分位数（Acklam 有理逼近，误差 < 1.2e-9）。"""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# --------------------------------------------------------------------------
# 线性代数 / 最小二乘
# --------------------------------------------------------------------------


def solve_spd(A: List[List[float]], b: List[float]) -> List[float]:
    """高斯消元（部分主元），A 为方阵。数值上等价于解 A x = b。"""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("矩阵奇异，无法求解")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            factor = M[r][col] / pv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / M[r][r]
    return x


def ols(X: Sequence[Sequence[float]], y: Sequence[float], ridge: float = 1e-8) -> dict:
    """带轻微岭正则的最小二乘（截距列由调用方自行加入）。

    返回 beta / fitted / resid / sigma / r2。
    """
    n = len(X)
    k = len(X[0]) if n else 0
    if n == 0 or k == 0:
        raise ValueError("空设计矩阵")
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    for a in range(k):
        XtX[a][a] += ridge
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    beta = solve_spd(XtX, Xty)
    fitted = [sum(X[i][a] * beta[a] for a in range(k)) for i in range(n)]
    resid = [y[i] - fitted[i] for i in range(n)]
    dof = max(1, n - k)
    sigma = math.sqrt(sum(r * r for r in resid) / dof)
    ybar = mean(y)
    sst = sum((v - ybar) ** 2 for v in y)
    sse = sum(r * r for r in resid)
    r2 = 1.0 - sse / sst if sst else 0.0
    return {"beta": beta, "fitted": fitted, "resid": resid, "sigma": sigma, "r2": r2, "n": n, "k": k}


def standardize(cols: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    """按列 z-score，返回标准化矩阵与均值/标准差，供预测期还原。"""
    k = len(cols)
    means, sds = [], []
    for c in cols:
        m, s = mean(c), stdev(c) if len(c) > 1 else 0.0
        means.append(m)
        sds.append(s if s and s == s else 1.0)
    n = len(cols[0]) if k else 0
    out = [[(cols[j][i] - means[j]) / sds[j] for j in range(k)] for i in range(n)]
    return out, means, sds


# --------------------------------------------------------------------------
# 序列诊断
# --------------------------------------------------------------------------


def acf(xs: Sequence[float], lag: int) -> float:
    n = len(xs)
    if lag <= 0 or n <= lag:
        return 0.0
    m = mean(xs)
    denom = sum((x - m) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    num = sum((xs[i] - m) * (xs[i - lag] - m) for i in range(lag, n))
    return num / denom


def durbin_watson(resid: Sequence[float]) -> float:
    num = sum((resid[i] - resid[i - 1]) ** 2 for i in range(1, len(resid)))
    den = sum(r * r for r in resid)
    return num / den if den else float("nan")


def _gammainc_lower_reg(s: float, x: float) -> float:
    """正则化下不完全伽马函数 P(s,x)，用于卡方检验 p 值。"""
    if x <= 0:
        return 0.0
    if x < s + 1:
        term, total, n = 1.0 / s, 1.0 / s, 0
        while n < 200:
            n += 1
            term *= x / (s + n)
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        return total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    # 连分式
    tiny = 1e-300
    b = x + 1 - s
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    q = math.exp(-x + s * math.log(x) - math.lgamma(s)) * h
    return 1.0 - q


def ljung_box_pvalue(resid: Sequence[float], lags: int = 10) -> float:
    """Ljung-Box 白噪声检验 p 值；p 小说明残差仍有自相关。"""
    n = len(resid)
    if n <= lags + 1:
        return float("nan")
    q = 0.0
    for k in range(1, lags + 1):
        r = acf(resid, k)
        q += r * r / (n - k)
    q *= n * (n + 2)
    # 卡方上尾概率 = 1 - P(k/2, q/2)
    return 1.0 - _gammainc_lower_reg(lags / 2.0, q / 2.0)


# --------------------------------------------------------------------------
# 误差指标
# --------------------------------------------------------------------------


def mae(actual: Sequence[float], pred: Sequence[float]) -> float:
    return mean([abs(a - p) for a, p in zip(actual, pred)])


def rmse(actual: Sequence[float], pred: Sequence[float]) -> float:
    return math.sqrt(mean([(a - p) ** 2 for a, p in zip(actual, pred)]))


def mape(actual: Sequence[float], pred: Sequence[float]) -> float:
    vals = [abs((a - p) / a) for a, p in zip(actual, pred) if a]
    return mean(vals) * 100 if vals else float("nan")


def smape(actual: Sequence[float], pred: Sequence[float]) -> float:
    vals = [2 * abs(a - p) / (abs(a) + abs(p)) for a, p in zip(actual, pred) if (abs(a) + abs(p))]
    return mean(vals) * 100 if vals else float("nan")


def mase(actual: Sequence[float], pred: Sequence[float], insample: Sequence[float], season: int = 12) -> float:
    """相对季节朴素基准的平均绝对误差（<1 说明优于季节朴素）。"""
    if len(insample) <= season:
        return float("nan")
    naive = [abs(insample[i] - insample[i - season]) for i in range(season, len(insample))]
    scale = mean(naive)
    return mae(actual, pred) / scale if scale else float("nan")


def direction_accuracy(actual: Sequence[float], pred: Sequence[float], base: Sequence[float]) -> float:
    """相对基期（预测发起时价格）的方向命中率。"""
    hits = 0
    n = 0
    for a, p, b in zip(actual, pred, base):
        if a == b or p == b:
            continue
        n += 1
        if (a > b) == (p > b):
            hits += 1
    return hits / n * 100 if n else float("nan")
