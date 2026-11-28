# Copyright (c) 2026 李硕
# SPDX-License-Identifier: MIT
"""梯度提升树（GBM）预测器。

两种后端：
- 后端 A（默认，零依赖）：本文件内的直方图分裂 GBRT，深度 3、叶子最小 8 样本，
  对 100~150 条月度样本足够，且完全确定性（固定随机种子）。
- 后端 B（可选）：若安装 lightgbm（MIT 协议），自动切换为 LGBMRegressor，
  参数与后端 A 语义一致，便于规模化换用。

建模方式为**直接多步**（direct multi-horizon）：每个跨度单独训练一个模型，
目标为「t+h 月收获窗口的均价（对数）」。相比递归预测，直接法不会把预测误差
逐月反馈放大，更适合 6-12 个月的长跨度。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core import numeric, timeutil
from ..data import features as feat
from ..data.schema import DatasetBundle
from .base import Forecast, Forecaster, clamp_sigma

try:  # 可选依赖
    import lightgbm as _lgb  # type: ignore
    LGB_AVAILABLE = True
except Exception:  # noqa: BLE001
    _lgb = None
    LGB_AVAILABLE = False


# --------------------------------------------------------------------------
# 极简 GBRT：平方损失 + 直方图分裂
# --------------------------------------------------------------------------


class _Tree:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self) -> None:
        self.feature = -1
        self.threshold = 0.0
        self.left: Optional["_Tree"] = None
        self.right: Optional["_Tree"] = None
        self.value = 0.0

    def predict(self, x: Sequence[float]) -> float:
        node = self
        while node.feature >= 0 and node.left is not None and node.right is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value


def _sse(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    m = numeric.mean(vals)
    return sum((v - m) ** 2 for v in vals)


def _build(X: List[List[float]], y: List[float], idx: List[int], depth: int,
           params: Dict[str, Any], rnd: random.Random, n_features: int) -> _Tree:
    """单节点分裂：按特征排序 + 前缀和，O(n log n) 找最优切点。"""
    node = _Tree()
    node.value = numeric.mean([y[i] for i in idx])
    max_depth = int(params.get("max_depth", 3))
    min_leaf = int(params.get("min_leaf", 8))
    n = len(idx)
    if depth >= max_depth or n < 2 * min_leaf:
        return node
    frac = float(params.get("feature_fraction", 0.85))
    k = max(1, int(round(n_features * frac)))
    feats = rnd.sample(range(n_features), min(k, n_features))

    ys = [y[i] for i in idx]
    total_sum = sum(ys)
    total_sq = sum(v * v for v in ys)
    parent_sse = total_sq - total_sum * total_sum / n

    best_gain, best_f, best_t = 0.0, -1, 0.0
    for f in feats:
        order = sorted(range(n), key=lambda j: X[idx[j]][f])
        left_sum = 0.0
        left_sq = 0.0
        for p in range(1, n):
            j = order[p - 1]
            v = ys[j]
            left_sum += v
            left_sq += v * v
            if p < min_leaf or n - p < min_leaf:
                continue
            if X[idx[order[p]]][f] == X[idx[order[p - 1]]][f]:
                continue  # 相同取值不能切
            right_sum = total_sum - left_sum
            right_sq = total_sq - left_sq
            sse_l = left_sq - left_sum * left_sum / p
            sse_r = right_sq - right_sum * right_sum / (n - p)
            gain = parent_sse - sse_l - sse_r
            if gain > best_gain:
                best_gain = gain
                best_f = f
                best_t = (X[idx[order[p]]][f] + X[idx[order[p - 1]]][f]) / 2.0
    if best_f < 0 or best_gain <= 1e-12:
        return node
    left = [i for i in idx if X[i][best_f] <= best_t]
    right = [i for i in idx if X[i][best_f] > best_t]
    if len(left) < min_leaf or len(right) < min_leaf:
        return node
    node.feature, node.threshold = best_f, best_t
    node.left = _build(X, y, left, depth + 1, params, rnd, n_features)
    node.right = _build(X, y, right, depth + 1, params, rnd, n_features)
    return node


class PureGBRT:
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params
        self.trees: List[_Tree] = []
        self.base = 0.0
        self.n_features = 0

    def fit(self, X: List[List[float]], y: List[float]) -> "PureGBRT":
        rnd = random.Random(int(self.params.get("seed", 42)))
        n = len(X)
        self.n_features = len(X[0]) if X else 0
        self.base = numeric.mean(y)
        pred = [self.base] * n
        lr = float(self.params.get("learning_rate", 0.05))
        n_est = int(self.params.get("n_estimators", 220))
        sub = float(self.params.get("subsample", 0.85))
        for _ in range(n_est):
            resid = [y[i] - pred[i] for i in range(n)]
            idx = [i for i in range(n) if rnd.random() < sub] or list(range(n))
            tree = _build(X, resid, idx, 0, self.params, rnd, self.n_features)
            self.trees.append(tree)
            for i in range(n):
                pred[i] += lr * tree.predict(X[i])
        return self

    def predict(self, x: Sequence[float]) -> float:
        return self.base + sum(self.params.get("learning_rate", 0.05) * t.predict(x) for t in self.trees)


# --------------------------------------------------------------------------
# 预测器
# --------------------------------------------------------------------------


class GbmForecaster(Forecaster):
    name = "gbm"
    needs_cols = True

    def __init__(self) -> None:
        self._models: Dict[int, Any] = {}
        self._cols: List[str] = []
        self._targets: List[Dict[str, Any]] = []
        self._X: List[List[float]] = []
        self._sigma: float = 0.09
        self._resid: List[float] = []
        self._n = 0
        self._horizon = 6
        self._harvest: Sequence[int] = []
        self._backend = "pure"
        self._importance: Dict[str, float] = {}
        self._params: Dict[str, Any] = {}
        self._last_x: Optional[List[float]] = None

    def fit(self, ds: DatasetBundle, harvest_months: Sequence[int], horizon: int,
            cols: Sequence[str], settings: Dict[str, Any]) -> "GbmForecaster":
        self._cols = list(cols)
        self._n = ds.n()
        self._horizon = horizon
        self._harvest = harvest_months
        self._params = dict(settings.get("model", {}).get("gbm") or {})
        self._X = feat.design_matrix(ds, self._cols)
        self._models = {}
        self._backend = "pure"
        if LGB_AVAILABLE and len(self._X) >= 60:
            self._backend = "lightgbm"

        for h in (horizon,):
            targets = feat.build_targets(ds, harvest_months, h)
            if len(targets) < 30:
                continue
            Xs = [self._X[t["t"]] for t in targets]
            ys = [t["y"] for t in targets]
            model = self._fit_one(Xs, ys)
            self._models[h] = model
            self._targets = targets
            preds = [model.predict(x) for x in Xs]
            self._resid = [y - p for y, p in zip(ys, preds)]
            q10, q90 = numeric.quantile(self._resid, 0.1), numeric.quantile(self._resid, 0.9)
            sigma_in = (q90 - q10) / (2 * 1.2816) if q90 > q10 else 0.09
            # 时间顺序留出验证：训练集残差必然偏小，用它定区间会系统性过窄。
            # 这里在前 75% 样本上重训、在后 25% 上取残差标准差，作为「可比口径」的不确定性。
            sigma_ho = None
            cut = int(len(Xs) * 0.75)
            if len(Xs) - cut >= 12:
                keep_imp = dict(self._importance)
                m_ho = self._fit_one(Xs[:cut], ys[:cut])
                self._importance = keep_imp
                res_ho = [ys[i] - m_ho.predict(Xs[i]) for i in range(cut, len(Xs))]
                if len(res_ho) >= 6:
                    sigma_ho = numeric.stdev(res_ho)
            self._sigma = clamp_sigma(max(sigma_in, sigma_ho or 0.0) * 1.02)
        return self

    def _fit_one(self, Xs: List[List[float]], ys: List[float]) -> Any:
        if self._backend == "lightgbm":
            p = self._params
            model = _lgb.LGBMRegressor(
                n_estimators=int(p.get("n_estimators", 220)),
                learning_rate=float(p.get("learning_rate", 0.05)),
                max_depth=int(p.get("max_depth", 3)),
                min_child_samples=int(p.get("min_leaf", 8)),
                subsample=float(p.get("subsample", 0.85)),
                colsample_bytree=float(p.get("feature_fraction", 0.85)),
                reg_lambda=0.1, random_state=int(p.get("seed", 42)), verbose=-1,
            )
            model.fit(Xs, ys)
            imp = getattr(model, "feature_importances_", None)
            if imp is not None:
                self._importance = {c: float(v) for c, v in zip(self._cols, imp)}
            return model
        model = PureGBRT(self._params).fit(Xs, ys)
        # 简化重要性：按树分裂次数
        counts: Dict[int, int] = {}
        stack = list(model.trees)
        while stack:
            node = stack.pop()
            if node.feature >= 0:
                counts[node.feature] = counts.get(node.feature, 0) + 1
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
        total = sum(counts.values()) or 1
        self._importance = {c: round(counts.get(i, 0) / total, 4) for i, c in enumerate(self._cols)}
        return model

    # ------------------------------------------------------------------
    def predict_vector(self, x: List[float]) -> float:
        model = self._models.get(self._horizon)
        if model is None:
            return numeric.mean([t["y"] for t in self._targets]) if self._targets else 0.0
        return float(model.predict(x))

    def forecast_at(self, origin_t: int) -> Forecast:
        if not self._models:
            raise RuntimeError("GBM 未训练（样本不足）")
        origin_t = max(0, min(origin_t, self._n - 1))
        x = self._X[origin_t]
        self._last_x = x
        mean_log = self.predict_vector(x)
        top = sorted(self._importance.items(), key=lambda kv: -kv[1])[:6]
        return Forecast(model=self.name, mean_log=mean_log, sigma=clamp_sigma(self._sigma),
                        horizon=self._horizon,
                        diagnostics={"backend": self._backend, "n_train": len(self._targets),
                                     "sigma": round(self._sigma, 4),
                                     "feature_importance": dict(top)},
                        detail={"features": self._cols, "x": [round(v, 4) for v in x]})

    def scenario_predict(self, shift: Dict[str, float]) -> Optional[float]:
        """情景模拟：对指定特征施加位移（单位=标准差）后重新推理。"""
        if self._last_x is None:
            return None
        x = list(self._last_x)
        for col, delta in shift.items():
            if col in self._cols:
                x[self._cols.index(col)] += delta
        return self.predict_vector(x)

    @property
    def importance(self) -> Dict[str, float]:
        return self._importance
