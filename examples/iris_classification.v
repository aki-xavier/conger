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
import json2

const class_names = ['setosa', 'versicolor', 'virginica']

// --- 自包含 plotly.js 可视化(替代 vsl.plot, 无第三方依赖) -------------------

// scatter_trace 生成一组等大 marker 散点的 plotly trace JSON。
fn scatter_trace(xs []f64, ys []f64, name string, size f64) string {
	sizes := []f64{len: xs.len, init: size}
	return '{"type":"scatter","mode":"markers","name":' + json2.encode(name) + ',"x":' +
		json2.encode(xs) + ',"y":' + json2.encode(ys) + ',"marker":{"size":' + json2.encode(sizes) +
		'}}'
}

// contour_trace 生成逐类似然等高线 trace JSON(showscale:false 直接写入,
// 避免多条等高线各带一个 colorbar 互相重叠并压住图例)。
fn contour_trace(xs []f64, ys []f64, z [][]f64, name string, colorscale string) string {
	return '{"type":"contour","showscale":false,"name":' + json2.encode(name) + ',"colorscale":' +
		json2.encode(colorscale) +
		',"ncontours":6,"contours":{"coloring":"lines","showlabels":true},"x":' + json2.encode(xs) +
		',"y":' + json2.encode(ys) + ',"z":' + json2.encode(z) + '}'
}

// heatmap_trace 生成后验热力图 trace JSON。
fn heatmap_trace(z [][]f64, x []int, y []string) string {
	return '{"type":"heatmap","z":' + json2.encode(z) + ',"x":' + json2.encode(x) + ',"y":' +
		json2.encode(y) + '}'
}

// plot_layout 生成布局 JSON;xtitle/ytitle 为空则省略对应轴。
fn plot_layout(title string, width int, height int, xtitle string, ytitle string) string {
	mut s := '{"title":' + json2.encode(title) + ',"width":${width},"height":${height}'
	if xtitle != '' {
		s += ',"xaxis":{"title":' + json2.encode(xtitle) + '}'
	}
	if ytitle != '' {
		s += ',"yaxis":{"title":' + json2.encode(ytitle) + '}'
	}
	return s + '}'
}

// write_plot_html 把 trace/layout JSON 写成自包含 HTML(plotly.js 走 CDN,
// 不启动服务器也不打开浏览器)。
fn write_plot_html(path string, traces []string, layout string) {
	head := r'<!DOCTYPE html>
<html>
  <head><meta charset="utf-8"><title>conger plot</title></head>
  <body>
    <div id="gd"></div>
    <script type="module">
import "https://cdn.plot.ly/plotly-2.26.2.min.js";
const layout = '
	mid := ';
const data = '
	tail := ';
Plotly.newPlot("gd", { data, layout });
    </script>
  </body>
</html>
'
	os.write_file(path, head + layout + mid + '[' + traces.join(',') + ']' + tail) or { panic(err) }
}

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
	os.write_file('docs/iris_model_dag.mmd', lines.join('\n') + '\n') or { panic(err) }
	println('')
	println('模型结构 DAG 已写出: docs/iris_model_dag.mmd' +
		'(探测样本=#${probe}, 展开责任度 top-3 分量)')

	// == 可视化(自包含 plotly.js JSON → HTML, 写盘不弹浏览器) ==
	// V1: 测试样本在最有判别力的两个原始特征(花瓣长/宽)上的散点,
	// 按真类着色, 错分样本大点高亮。
	mut sp_traces := []string{}
	// V1-likelihood: 逐类似然等高线 p(x|class)。白化是线性可逆变换(本例
	// D=4 全保留), 故每个分量在原始特征空间是全协方差高斯:
	// μx = f_mean + f_mu·Bᵀ, Σx = B·diag(f_var)·Bᵀ; 高斯的边缘分布 =
	// 子向量/子矩阵, 对花瓣平面(特征 2,3)取 2×2 块即可精确求值。
	fm := (m.f_mean or { panic('missing f_mean') }).data_f32()
	bmat := (m.basis or { panic('missing basis') }).data_f32() // (4, D) 行主序
	dd := m.f_mu.dim(1)
	// 分量所属类别(cat_logp 行 argmax)
	mut comp_class := []int{len: k}
	for j in 0 .. k {
		mut bc := 0
		for c in 1 .. 3 {
			if clp[j * 3 + c] > clp[j * 3 + bc] {
				bc = c
			}
		}
		comp_class[j] = bc
	}
	// 花瓣平面网格
	nx, ny := 60, 40
	gx0, gx1 := 0.5, 7.5
	gy0, gy1 := -0.2, 2.9
	mut xsg := []f64{len: nx}
	mut ysg := []f64{len: ny}
	for i in 0 .. nx {
		xsg[i] = gx0 + (gx1 - gx0) * f64(i) / f64(nx - 1)
	}
	for i in 0 .. ny {
		ysg[i] = gy0 + (gy1 - gy0) * f64(i) / f64(ny - 1)
	}
	for c in 0 .. 3 {
		mut zc := [][]f64{len: ny, init: []f64{len: nx}}
		mut kc := 0
		for j in 0 .. k {
			if comp_class[j] != c {
				continue
			}
			kc++
			// μx 的边缘(特征 2,3)
			mut mu2 := [2]f64{}
			for a in 0 .. 2 {
				mut s := f64(fm[2 + a])
				for q in 0 .. dd {
					s += f64(fmu[j * dd + q]) * f64(bmat[(2 + a) * dd + q])
				}
				mu2[a] = s
			}
			// Σx 的边缘 2×2
			mut sig := [4]f64{}
			for a in 0 .. 2 {
				for b in 0 .. 2 {
					mut s := 0.0
					for q in 0 .. dd {
						s += f64(bmat[(2 + a) * dd + q]) * f64(fvar[j * dd + q]) * f64(bmat[
							(2 + b) * dd + q])
					}
					sig[a * 2 + b] = s
				}
			}
			det := sig[0] * sig[3] - sig[1] * sig[2]
			i00, i01, i11 := sig[3] / det, -sig[1] / det, sig[0] / det
			norm := 1.0 / (2.0 * math.pi * math.sqrt(det))
			for iy in 0 .. ny {
				for ix in 0 .. nx {
					dx := xsg[ix] - mu2[0]
					dy := ysg[iy] - mu2[1]
					e := dx * dx * i00 + 2.0 * dx * dy * i01 + dy * dy * i11
					zc[iy][ix] += norm * math.exp(-0.5 * e)
				}
			}
		}
		inv := 1.0 / f64(kc)
		for iy in 0 .. ny {
			for ix in 0 .. nx {
				zc[iy][ix] *= inv
			}
		}
		sp_traces << contour_trace(xsg, ysg, zc, 'p(x|' + class_names[c] + ')', [
			'Blues',
			'Oranges',
			'Greens',
		][c])
	}
	for c in 0 .. 3 {
		mut xs := []f64{}
		mut ys := []f64{}
		for i, y in test_y {
			if y == c {
				xs << test_f[i * 4 + 2]
				ys << test_f[i * 4 + 3]
			}
		}
		sp_traces << scatter_trace(xs, ys, class_names[c], 8.0)
	}
	mut mxs := []f64{}
	mut mys := []f64{}
	for i, y in test_y {
		if preds[i] != y {
			mxs << test_f[i * 4 + 2]
			mys << test_f[i * 4 + 3]
		}
	}
	if mxs.len > 0 {
		sp_traces << scatter_trace(mxs, mys, '错分样本', 16.0)
	}
	write_plot_html('docs/iris_scatter.html', sp_traces, plot_layout('Iris 测试集: 花瓣长 × 花瓣宽(按真类着色)',
		720, 560, '花瓣长 (cm)', '花瓣宽 (cm)'))

	// V2: 后验 P(class|x) 热力图(3 类 × 30 测试样本)。
	mut z := [][]f64{len: 3, init: []f64{len: test_y.len}}
	for c in 0 .. 3 {
		for i in 0 .. test_y.len {
			z[c][i] = cp[i * 3 + c]
		}
	}
	mut xidx := []int{len: test_y.len, init: index}
	write_plot_html('docs/iris_posterior_heatmap.html', [
		heatmap_trace(z, xidx, class_names),
	],
		plot_layout('predict 后验 P(class|x)(逐测试样本)', 900, 420, '测试样本编号',
		''))
	println('可视化已写出: docs/iris_scatter.html · docs/iris_posterior_heatmap.html')

	// 注: 本例对应 docs/architecture.md 主管线的两步 —— A. 训练
	// fit_mixture_spn(确定性装配实例级对角高斯混合, 无 EM 迭代)与 B. 推理
	// predict(核回归责任度 r 对 cat_logp 加权得到离散场景因子后验
	// P(class|x))。特征无需手工标准化: fit 内部 whiten(f) 用 PCA Gram 特征
	// 分解把原始特征投影到零均值、单位方差的白化坐标(basis_dim=0 保留全部
	// 方向), 各维度量纲差异(花萼 ~5-8cm vs 花瓣宽 ~0.1-2.5cm)被白化吸收,
	// predict 时用同一 (f_mean, basis) 变换测试样本, 保证训练/推理一致。
}
