"""GenericEM: 领域无关的期望最大化 (EM) 迭代框架。

把生成模型抽象为「观测 X ← (隐变量 Z, 参数 θ)」, EM 在两者间迭代:

  E 步: q(Z) = P(Z | X, θ_t)                  (隐变量软后验/责任度)
  M 步: θ_{t+1} = argmax E_q[log P(X, Z | θ)]  (参数极大化)

框架只负责循环、温度 (E 步置信锐化)、阻尼 (M 步更新稳定) 与收敛监控;
具体生成模型通过 `GenerativeModel` 协议注入 —— 透明层叠加、几何↔光照、
软对应 (ICP)、分割↔位姿 等都是同一工作母机的不同实例。

与 MixtureSPN 学习层的 EM (质心压缩, 已在 §2.2 实测退役) 无关: 那里
否定的是「小数据弯曲流形上把点平均到流形外」的压缩; 这里是推理期已知
生成模型下的后验迭代, 问题层不同, 不构成回退。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class GenerativeModel(Protocol):
    """EM 所需的生成模型接口 (领域适配器实现)。"""

    def responsibilities(self, params: Any, observation: Any, temperature: float = 1.0) -> Any:
        """E 步: 观测 + 当前参数 → 隐变量软后验 q(Z)。"""
        ...

    def maximize(
        self,
        responsibilities: Any,
        observation: Any,
        params: Any,
        damping: float = 0.0,
    ) -> Any:
        """M 步: 隐变量后验 + 观测 + 当前参数 → 新参数 (阻尼可稳定更新)。"""
        ...

    def log_likelihood(self, params: Any, observation: Any) -> float:
        """当前参数下的观测对数似然 (收敛监控; EM 下应单调不减)。"""
        ...

    def sample(self, params: Any, rng: Any = None) -> Any:
        """正向模型: 参数 → 合成观测 (验证/探针用)。"""
        ...


@dataclass(frozen=True)
class EMResult:
    """一次 EM 循环的完整结果。"""

    params: Any
    responsibilities: Any
    log_likelihood: float
    iterations: int
    trajectory: tuple[float, ...]


class EMLoop:
    """EM 循环 + 温度/阻尼 + 收敛监控。"""

    def __init__(
        self,
        model: GenerativeModel,
        max_iters: int = 50,
        tol: float = 1e-6,
        temperature: float = 1.0,
        damping: float = 0.0,
    ):
        if max_iters < 1:
            raise ValueError("max_iters 必须 >=1")
        self.model = model
        self.max_iters = max_iters
        self.tol = tol
        self.temperature = temperature
        self.damping = damping

    def run(self, observation: Any, init_params: Any) -> EMResult:
        """从 init_params 出发迭代 E/M, 返回收敛状态 + 轨迹。"""
        params = init_params
        resp = None
        trajectory: list[float] = []
        prev_ll = -float("inf")
        for _ in range(self.max_iters):
            resp = self.model.responsibilities(params, observation, self.temperature)
            params = self.model.maximize(resp, observation, params, self.damping)
            ll = float(self.model.log_likelihood(params, observation))
            trajectory.append(ll)
            if len(trajectory) > 1 and ll - prev_ll < self.tol:
                break
            prev_ll = ll
        return EMResult(
            params=params,
            responsibilities=resp,
            log_likelihood=trajectory[-1],
            iterations=len(trajectory),
            trajectory=tuple(trajectory),
        )
