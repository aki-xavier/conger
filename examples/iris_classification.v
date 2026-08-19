module main

// iris_classification.v — 经典 Fisher Iris 鸢尾花三分类示例(单文件, module main)。
//
// 运行: v -gc boehm -no-memory-limit run examples/iris_classification.v
//
// 数据集: Fisher (1936) / UCI Machine Learning Repository 的 Iris 标准数值
// (https://archive.ics.uci.edu/ml/datasets/iris), 150 样本 × 4 特征, 单位 cm。
// 特征顺序: [花萼长 sepal length, 花萼宽 sepal width,
//
//	花瓣长 petal length, 花瓣宽 petal width]
//
// 类顺序: 0=setosa / 1=versicolor / 2=virginica, 各 50 条, 按类连续排列。
//
// 建模思路: Iris 是纯表格分类, 对应 MixtureSPN 的「离散场景因子 ≡ 条件后验
// 分类」: f = 4 维特征, scene_classes = 类别列 (n,1) int, cat_sizes = [3],
// stratum = 类别索引(逐类 tied 方差), t 用 zeros(n,1) 占位(本例不用连续
// 目标)。predict 返回的第二项 P(scene|x) 即三类后验, argmax 得预测类别。
import math
import os
import mlx
import conger

const class_names = ['setosa', 'versicolor', 'virginica']

// 150 样本 × 4 特征(每行一个样本: 花萼长, 花萼宽, 花瓣长, 花瓣宽), 每 50 行为一个类。
const iris_data = [
	// --- setosa (0-49) ---
	[5.1, 3.5, 1.4, 0.2],
	[4.9, 3.0, 1.4, 0.2],
	[4.7, 3.2, 1.3, 0.2],
	[4.6, 3.1, 1.5, 0.2],
	[5.0, 3.6, 1.4, 0.2],
	[5.4, 3.9, 1.7, 0.4],
	[4.6, 3.4, 1.4, 0.3],
	[5.0, 3.4, 1.5, 0.2],
	[4.4, 2.9, 1.4, 0.2],
	[4.9, 3.1, 1.5, 0.1],
	[5.4, 3.7, 1.5, 0.2],
	[4.8, 3.4, 1.6, 0.2],
	[4.8, 3.0, 1.4, 0.1],
	[4.3, 3.0, 1.1, 0.1],
	[5.8, 4.0, 1.2, 0.2],
	[5.7, 4.4, 1.5, 0.4],
	[5.4, 3.9, 1.3, 0.4],
	[5.1, 3.5, 1.4, 0.3],
	[5.7, 3.8, 1.7, 0.3],
	[5.1, 3.8, 1.5, 0.3],
	[5.4, 3.4, 1.7, 0.2],
	[5.1, 3.7, 1.5, 0.4],
	[4.6, 3.6, 1.0, 0.2],
	[5.1, 3.3, 1.7, 0.5],
	[4.8, 3.4, 1.9, 0.2],
	[5.0, 3.0, 1.6, 0.2],
	[5.0, 3.4, 1.6, 0.4],
	[5.2, 3.5, 1.5, 0.2],
	[5.2, 3.4, 1.4, 0.2],
	[4.7, 3.2, 1.6, 0.2],
	[4.8, 3.1, 1.6, 0.2],
	[5.4, 3.4, 1.5, 0.4],
	[5.2, 4.1, 1.5, 0.1],
	[5.5, 4.2, 1.4, 0.2],
	[4.9, 3.1, 1.5, 0.2],
	[5.0, 3.2, 1.2, 0.2],
	[5.5, 3.5, 1.3, 0.2],
	[4.9, 3.6, 1.4, 0.1],
	[4.4, 3.0, 1.3, 0.2],
	[5.1, 3.4, 1.5, 0.2],
	[5.0, 3.5, 1.3, 0.3],
	[4.5, 2.3, 1.3, 0.3],
	[4.4, 3.2, 1.3, 0.2],
	[5.0, 3.5, 1.6, 0.6],
	[5.1, 3.8, 1.9, 0.4],
	[4.8, 3.0, 1.4, 0.3],
	[5.1, 3.8, 1.6, 0.2],
	[4.6, 3.2, 1.4, 0.2],
	[5.3, 3.7, 1.5, 0.2],
	[5.0, 3.3, 1.4, 0.2],
	// --- versicolor (50-99) ---
	[7.0, 3.2, 4.7, 1.4],
	[6.4, 3.2, 4.5, 1.5],
	[6.9, 3.1, 4.9, 1.5],
	[5.5, 2.3, 4.0, 1.3],
	[6.5, 2.8, 4.6, 1.5],
	[5.7, 2.8, 4.5, 1.3],
	[6.3, 3.3, 4.7, 1.6],
	[4.9, 2.4, 3.3, 1.0],
	[6.6, 2.9, 4.6, 1.3],
	[5.2, 2.7, 3.9, 1.4],
	[5.0, 2.0, 3.5, 1.0],
	[5.9, 3.0, 4.2, 1.5],
	[6.0, 2.2, 4.0, 1.0],
	[6.1, 2.9, 4.7, 1.4],
	[5.6, 2.9, 3.6, 1.3],
	[6.7, 3.1, 4.4, 1.4],
	[5.6, 3.0, 4.5, 1.5],
	[5.8, 2.7, 4.1, 1.0],
	[6.2, 2.2, 4.5, 1.5],
	[5.6, 2.5, 3.9, 1.1],
	[5.9, 3.2, 4.8, 1.8],
	[6.1, 2.8, 4.0, 1.3],
	[6.3, 2.5, 4.9, 1.5],
	[6.1, 2.8, 4.7, 1.2],
	[6.4, 2.9, 4.3, 1.3],
	[6.6, 3.0, 4.4, 1.4],
	[6.8, 2.8, 4.8, 1.4],
	[6.7, 3.0, 5.0, 1.7],
	[6.0, 2.9, 4.5, 1.5],
	[5.7, 2.6, 3.5, 1.0],
	[5.5, 2.4, 3.8, 1.1],
	[5.5, 2.4, 3.7, 1.0],
	[5.8, 2.7, 3.9, 1.2],
	[6.0, 2.7, 5.1, 1.6],
	[5.4, 3.0, 4.5, 1.5],
	[6.0, 3.4, 4.5, 1.6],
	[6.7, 3.1, 4.7, 1.5],
	[6.3, 2.3, 4.4, 1.3],
	[5.6, 3.0, 4.1, 1.3],
	[5.5, 2.5, 4.0, 1.3],
	[5.5, 2.6, 4.4, 1.2],
	[6.1, 3.0, 4.6, 1.4],
	[5.8, 2.6, 4.0, 1.2],
	[5.0, 2.3, 3.3, 1.0],
	[5.6, 2.7, 4.2, 1.3],
	[5.7, 3.0, 4.2, 1.2],
	[5.7, 2.9, 4.2, 1.3],
	[6.2, 2.9, 4.3, 1.3],
	[5.1, 2.5, 3.0, 1.1],
	[5.7, 2.8, 4.1, 1.3],
	// --- virginica (100-149) ---
	[6.3, 3.3, 6.0, 2.5],
	[5.8, 2.7, 5.1, 1.9],
	[7.1, 3.0, 5.9, 2.1],
	[6.3, 2.9, 5.6, 1.8],
	[6.5, 3.0, 5.8, 2.2],
	[7.6, 3.0, 6.6, 2.1],
	[4.9, 2.5, 4.5, 1.7],
	[7.3, 2.9, 6.3, 1.8],
	[6.7, 2.5, 5.8, 1.8],
	[7.2, 3.6, 6.1, 2.5],
	[6.5, 3.2, 5.1, 2.0],
	[6.4, 2.7, 5.3, 1.9],
	[6.8, 3.0, 5.5, 2.1],
	[5.7, 2.5, 5.0, 2.0],
	[5.8, 2.8, 5.1, 2.4],
	[6.4, 3.2, 5.3, 2.3],
	[6.5, 3.0, 5.5, 1.8],
	[7.7, 3.8, 6.7, 2.2],
	[7.7, 2.6, 6.9, 2.3],
	[6.0, 2.2, 5.0, 1.5],
	[6.9, 3.2, 5.7, 2.3],
	[5.6, 2.8, 4.9, 2.0],
	[7.7, 2.8, 6.7, 2.0],
	[6.3, 2.7, 4.9, 1.8],
	[6.7, 3.3, 5.7, 2.1],
	[7.2, 3.2, 6.0, 1.8],
	[6.2, 2.8, 4.8, 1.8],
	[6.1, 3.0, 4.9, 1.8],
	[6.4, 2.8, 5.6, 2.1],
	[7.2, 3.0, 5.8, 1.6],
	[7.4, 2.8, 6.1, 1.9],
	[7.9, 3.8, 6.4, 2.0],
	[6.4, 2.8, 5.6, 2.2],
	[6.3, 2.8, 5.1, 1.5],
	[6.1, 2.6, 5.6, 1.4],
	[7.7, 3.0, 6.1, 2.3],
	[6.3, 3.4, 5.6, 2.4],
	[6.4, 3.1, 5.5, 1.8],
	[6.0, 3.0, 4.8, 1.8],
	[6.9, 3.1, 5.4, 2.1],
	[6.7, 3.1, 5.6, 2.4],
	[6.9, 3.1, 5.1, 2.3],
	[5.8, 2.7, 5.1, 1.9],
	[6.8, 3.2, 5.9, 2.3],
	[6.7, 3.3, 5.7, 2.5],
	[6.7, 3.0, 5.2, 2.3],
	[6.3, 2.5, 5.0, 1.9],
	[6.5, 3.0, 5.2, 2.0],
	[6.2, 3.4, 5.4, 2.3],
	[5.9, 3.0, 5.1, 1.8],
]

// 类标签: 0=setosa ×50, 1=versicolor ×50, 2=virginica ×50(数据按类连续排列,
// 标签由行号 i/50 派生, 见 main())。

fn main() {
	println('== Iris 鸢尾花分类: MixtureSPN 离散场景因子 ≡ 条件后验分类 ==')
	println('数据集: Fisher Iris (UCI), 150 样本 × 4 特征(cm), 三类各 50 条')

	// 确定性分层划分: 每类(连续 50 条)内每 5 个取第 1 个(类内索引 i%5==0)
	// 作测试, 其余训练 → 每类训练 40 / 测试 10, 合计训练 120 / 测试 30。
	mut train_f := []f64{cap: 120 * 4}
	mut train_y := []int{cap: 120}
	mut test_f := []f64{cap: 30 * 4}
	mut test_y := []int{cap: 30}
	for i in 0 .. 150 {
		label := i / 50 // 每类 50 条连续排列
		row := iris_data[i]
		if i % 5 == 0 {
			test_f << row
			test_y << label
		} else {
			train_f << row
			train_y << label
		}
	}
	println('划分: 每类内索引 i%5==0 → 测试(分层, 各类 40 训练 / 10 测试), ' +
		'训练 ${train_y.len} / 测试 ${test_y.len}')

	// 训练: fit_mixture_spn — 主管线 A(训练)步。
	f_train := mlx.arr32(train_f, [train_y.len, 4])
	t_train := mlx.zeros([train_y.len, 1], .float32) // 连续目标占位, 本例不用
	mut sc_i32 := []i32{cap: train_y.len}
	for y in train_y {
		sc_i32 << i32(y)
	}
	scene := mlx.array_i32(sc_i32, [train_y.len, 1]) // 类别列 (n,1)
	stratum := mlx.col(scene, 0) // 类别索引作 stratum: 逐类 tied 方差
	m := conger.fit_mixture_spn(f_train, t_train, stratum, 1e-3, scene, [3], 0)
	println('fit_mixture_spn: K=${m.f_mu.dim(0)} 个实例级分量, 白化特征维 D=${m.f_mu.dim(1)}')

	// 推理: predict — 主管线 B(推理)步, 第二项即 P(class|x)。
	f_test := mlx.arr32(test_f, [test_y.len, 4])
	_, cat_p, _ := m.predict(f_test)
	cp := cat_p.data_f32() // (30,3) 行主序
	mut preds := []int{cap: test_y.len}
	for i in 0 .. test_y.len {
		mut best := 0
		for c in 1 .. 3 {
			if cp[i * 3 + c] > cp[i * 3 + best] {
				best = c
			}
		}
		preds << best
	}

	// 评估: 总体 + 逐类准确率 + 错分明细。
	mut correct := 0
	mut per_class_total := [3]int{}
	mut per_class_correct := [3]int{}
	for i, y in test_y {
		per_class_total[y]++
		if preds[i] == y {
			correct++
			per_class_correct[y]++
		}
	}
	println('')
	println('== 测试结果 ==')
	println('总体准确率: ${correct}/${test_y.len} = ${f64(correct) / f64(test_y.len) * 100.0:.1f}%')
	for c in 0 .. 3 {
		println('  ${class_names[c]:10s}: ${per_class_correct[c]}/${per_class_total[c]}' +
			' = ${f64(per_class_correct[c]) / f64(per_class_total[c]) * 100.0:.1f}%')
	}
	if correct < test_y.len {
		println('错分样本明细:')
		for i, y in test_y {
			if preds[i] != y {
				p0, p1, p2 := cp[i * 3], cp[i * 3 + 1], cp[i * 3 + 2]
				println(
					'  样本 ${i:2}: 真类=${class_names[y]} 预测=${class_names[preds[i]]}' +
					' 后验=[${p0:.3f}, ${p1:.3f}, ${p2:.3f}]')
			}
		}
	} else {
		println('无错分样本。')
	}
	if f64(correct) / f64(test_y.len) < 0.85 {
		panic('准确率低于 85%, 请检查 fit/predict 接线而非硬凑')
	}

	// == 模型内部结构 DAG(mermaid) ==
	// MixtureSPN 是浅层混合, 天然是 DAG: 根 Sum(均匀混合 K 个分量) → 每分量
	// 一个 Product → 白化特征高斯叶 × D + 目标高斯叶 + 类别因子叶。K=120
	// 全画不可读, 这里取探测样本(优先错分样本)责任度最高的 3 个分量展开,
	// 叶子上标注训练出的真实参数, 其余分量折叠。
	_, _, r_all := m.predict(f_test)
	r := r_all.data_f32() // (30, K) 行主序
	k := m.f_mu.dim(0)
	d := m.f_mu.dim(1)
	w := m.log_w.data_f32()
	fmu := m.f_mu.data_f32()
	fvar := m.f_var.data_f32()
	tmu := m.t_mu.data_f32()
	clp := m.cat_logp.data_f32()
	mut probe := 0
	for i, y in test_y {
		if preds[i] != y {
			probe = i
			break
		}
	}
	mut top := []int{cap: 3}
	for top.len < 3 {
		mut best := -1
		mut bestv := f64(-1e30)
		for j in 0 .. k {
			if j in top {
				continue
			}
			v := f64(r[probe * k + j])
			if v > bestv {
				bestv = v
				best = j
			}
		}
		top << best
	}
	mut lines := ['flowchart TD']
	lines << '    S["Σ 混合根节点<br/>K=${k} · 均匀 log_w=${f64(w[0]):.3f}"]'
	for rank, j in top {
		rj := f64(r[probe * k + j])
		lines << '    S -->|"r=${rj:.3f}"| P${rank}["Π 分量 #${j}"]'
		mut chain := []string{cap: d + 2}
		for i in 0 .. d {
			mu := f64(fmu[j * d + i])
			va := f64(fvar[j * d + i])
			lines << '    P${rank} --> G${rank}_${i}["z${i + 1} ~ N(μ=${mu:.3f}, σ²=${va:.3f})"]'
			chain << 'G${rank}_${i}'
		}
		lines << '    P${rank} --> T${rank}["t ~ N(μ=${f64(tmu[j]):.3f}, tied σ²)<br/>(占位目标, 本例不用)"]'
		chain << 'T${rank}'
		p0 := math.exp(f64(clp[j * 3]))
		p1 := math.exp(f64(clp[j * 3 + 1]))
		p2 := math.exp(f64(clp[j * 3 + 2]))
		lines << '    P${rank} --> C${rank}["class ~ Cat<br/>setosa ${p0:.3f} · versicolor ${p1:.3f} · virginica ${p2:.3f}"]'
		chain << 'C${rank}'
		// 隐形链把同一分量的叶子竖向堆叠, 避免 6 个叶子横向排开
		lines << '    ' + chain.join(' ~~~ ')
	}
	lines << '    S --> REST["… 其余 ${k - 3} 个分量(结构相同)"]'
	os.write_file('examples/iris_model_dag.mmd', lines.join('\n') + '\n') or { panic(err) }
	println('')
	println('模型结构 DAG 已写出: examples/iris_model_dag.mmd' +
		'(探测样本=#${probe}, 展开责任度 top-3 分量)')

	// 注: 本例对应 docs/architecture.md 主管线的两步 —— A. 训练
	// fit_mixture_spn(确定性装配实例级对角高斯混合, 无 EM 迭代)与 B. 推理
	// predict(核回归责任度 r 对 cat_logp 加权得到离散场景因子后验
	// P(class|x))。特征无需手工标准化: fit 内部 whiten(f) 用 PCA Gram 特征
	// 分解把原始特征投影到零均值、单位方差的白化坐标(basis_dim=0 保留全部
	// 方向), 各维度量纲差异(花萼 ~5-8cm vs 花瓣宽 ~0.1-2.5cm)被白化吸收,
	// predict 时用同一 (f_mean, basis) 变换测试样本, 保证训练/推理一致。
}
