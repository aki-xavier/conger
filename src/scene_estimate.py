"""视觉兼容别名: SceneEstimate = StructuredHypothesis。

保留旧模块路径和类名, 防止视觉调用方分叉; 通用实现见
`structured_hypothesis.py`。
"""

from structured_hypothesis import HypothesisCandidate, StructuredHypothesis

SceneHypothesis = HypothesisCandidate
SceneEstimate = StructuredHypothesis
