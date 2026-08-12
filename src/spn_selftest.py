"""spn 自检: python src/spn_selftest.py (7 组)。

覆盖: G 检验 / 独立结构 / 混合后验 / 码先验 / 序列化 / 在线等价 / 在线生长。
"""

from __future__ import annotations

import math
import os
import tempfile

import mlx.core as mx

from online_spn import OnlineSPN
from product import Product
from spn import SPN
from spn_learner import SPNLearner


class SelfTest:
    """spn.py 自检 (python src/spn.py): 7 组。"""

    @staticmethod
    def run() -> None:
        key = mx.random.key(7)

        # 1) G 检验: 独立对 → 不相关; 相关对 → 相关 (α=0.05)
        n = 500
        a = mx.random.normal(shape=(n,), key=key)
        b = mx.random.normal(shape=(n,), key=mx.random.key(8))
        dep = SPNLearner.gtest(
            SPNLearner.binarize(mx.stack([a, b], axis=1), [0, 1], set())
        )
        assert not bool(dep[0, 1].item()), "G: 独立变量误判相关"
        c = a + 0.15 * mx.random.normal(shape=(n,), key=mx.random.key(9))
        dep2 = SPNLearner.gtest(
            SPNLearner.binarize(mx.stack([a, c], axis=1), [0, 1], set())
        )
        assert bool(dep2[0, 1].item()), "G: 相关变量误判独立"
        print("  ok  G 检验: 独立→独立, 相关→相关")

        # 2) 独立变量 → 根为 Product, log 密度 = 边缘对数之和
        x = mx.random.normal(shape=(4000, 3), key=key)
        spn = SPNLearner(set(), min_n=64).learn(x)
        assert isinstance(spn.root, Product), (
            f"根应为 Product, 实际 {type(spn.root).__name__}"
        )
        pt = mx.array([[0.3, -0.7, 1.1]])
        got = float(spn.eval_log(pt)[0])
        want = sum(
            -0.5 * v * v - 0.5 * math.log(2.0 * math.pi) for v in (0.3, -0.7, 1.1)
        )
        # n=4000 → 叶参数 MLE 误差 ~1/sqrt(2n)≈0.011, 容差 0.05 富余
        assert abs(got - want) < 0.05, (got, want)
        print("  ok  独立结构: 根 Product, log 密度 = 边缘乘积")

        # 3) 混合 + 离散标签 → 后验从连续证据恢复类
        n = 400
        lab = mx.concatenate([mx.zeros((n // 2,)), mx.ones((n // 2,))])
        f = lab * 4.0 + mx.random.normal(shape=(n,), key=key)
        spn3 = SPNLearner({1}, card={1: 2}, min_n=16).learn(mx.stack([f, lab], axis=1))
        feats = mx.array([[-4.0], [0.0], [4.0]])
        codes = mx.array([[0.0], [1.0]])
        post = spn3.posterior(feats, codes)  # (3, 2)
        assert float(post[0, 1]) < math.log(0.01), "x=−4 应属类 0"
        assert float(post[2, 1]) > math.log(0.99), "x=+4 应属类 1"
        assert abs(float(mx.exp(post[1]).sum()) - 1.0) < 1e-5, "后验行未归一"
        print("  ok  混合后验: 类标签从连续证据恢复, 行归一")

        # 4) 码先验注入: P(c|x) ∝ P(x|c)·P(c), 先验改变后验但保持归一
        prior = mx.array([math.log(0.9), math.log(0.1)])  # 强偏好类 0
        post_p = spn3.posterior(feats, codes, log_prior=prior)
        norm = float(mx.exp(post_p).sum(axis=1)[1])
        assert abs(norm - 1.0) < 1e-5, "先验注入后未归一"
        # x=0 处似然两分类相近, 先验应把后验推向类 0
        assert float(post_p[1, 0]) > float(post[1, 0]), "先验未提高类 0 后验"
        print("  ok  码先验: P(c|x) ∝ P(x|c)·P(c), 注入后行归一仍成立")

        # 5) 序列化 roundtrip (safetensors): save → load → eval 逐位一致

        fd, tmp = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        try:
            spn3.save(tmp, {"mu": mx.array([0.5]), "sd": mx.array([1.0])})
            spn4, extra = SPN.load(tmp)
            xq = mx.array([[-4.0, 0.0], [4.0, 1.0]])
            a = spn3.eval_log(xq)
            b = spn4.eval_log(xq)
            assert mx.all(mx.abs(a - b) < 1e-6), "roundtrip 后 eval 不一致"
            assert float(extra["mu"][0]) == 0.5, "extra 未随模型保存"
        finally:
            os.unlink(tmp)
        print("  ok  序列化: save → load → eval 一致, extra 随存")

        # 6) 在线参数等价: 同结构同数据, OnlineSPN 吸收 ≈ learn_spn MLE
        on = OnlineSPN(spn3.root, n_vars=2, code_cols=(1,), cards=(2,))
        x6 = mx.stack([f, lab], axis=1)
        on.absorb(x6, grow=False)
        xq = mx.array([[-4.0, 0.0], [0.0, 0.0], [4.0, 1.0]])
        d = float(mx.max(mx.abs(spn3.eval_log(xq) - on.to_spn().eval_log(xq))))
        assert d < 0.05, f"在线参数应≈批量 MLE: {d}"
        print(f"  ok  在线等价: 同结构同数据, |Δeval| = {d:.2e}")

        # 7) 生长: 码混合叶 + 计数显著 → 分裂, 后验仍恢复类
        # (打乱行序: 两批都含两类, 模拟真实增量; 有序喂入会让后分裂的
        # 子叶当批无本类播种行, 继承的混合高斯残留 —— 实验数据本就打乱)
        base7 = SPNLearner({1}, card={1: 2}, min_n=300).learn(x6)  # 强制浅树
        assert len(base7.root.leaf_blocks()) == 1, "应为单叶块 (码混合)"
        perm = mx.random.permutation(x6.shape[0], key=mx.random.key(5))
        x7 = x6[perm]
        on7 = OnlineSPN(base7.root, n_vars=2, code_cols=(1,), cards=(2,))
        on7.absorb(x7[:200])
        on7.absorb(x7[200:])
        assert len(on7.root.leaf_blocks()) == 2, "码混合叶应已分裂"
        post7 = on7.to_spn().posterior(feats, codes)
        assert float(post7[0, 1]) < math.log(0.01), "生长后 x=−4 应属类 0"
        assert float(post7[2, 1]) > math.log(0.99), "生长后 x=+4 应属类 1"
        print("  ok  在线生长: 码混合叶分裂, 后验仍恢复类")

if __name__ == "__main__":
    SelfTest.run()
    print("spn.py: 7 组自检 ✓")
