module main

// main_pipeline.v — conger 内核主管线(docs/architecture.md §主管线流程)的最小
// 端到端示例:A. 训练 fit_mixture_spn → B. 推理 predict → C. 结构门控
// GenericStructureGate + 出生 StructureBirthController → D. 模板学习
// TemplateDeltaLearner.tdl_learn → E. 持久化 save/split_save + 加载往返 →
// F. 模型内存(按需加载 load_transform/load_components + 动态遗忘
// forget_components + 基截断 truncate_basis)→ G. 似然核网络 KernelGraph
// (自定义 LikelihoodKernel + 前馈依赖 + 反馈迭代)。
//
// 运行: v -gc boehm -no-memory-limit run examples/main_pipeline.v
//
// 玩具域直接复用 conger 导出的验证域符号: new_toy_series_family /
// train_toy_expert / toy_x(toy_series_family.v · toy_series_expert.v),
// 接法与 generic_framework_test.v 一致。
import math
import os
import mlx
import conger

// ToyProposer 从出生证据生成「在 linear 之上 attach 一个二次项」的候选模板提案。
struct ToyProposer {}

fn (s ToyProposer) propose(cases []conger.StructureCase) []conger.TemplateProposal {
	mut out := []conger.TemplateProposal{}
	for c in cases {
		rmin := conger.min_of(c.residuals) or { 0.0 }
		out << conger.TemplateProposal{
			family:        'toy_quadratic'
			operation:     'attach'
			params:        c.params
			residual:      rmin
			complexity:    3.0
			score:         rmin + 3.0
			parent_family: 'linear'
			delta:         conger.TemplateDelta{
				ratio: 1.0 + rmin
			}
			metadata:      conger.TemplateMetadata{
				n_cases: cases.len
			}
		}
	}
	return out
}

// --- G 阶段的自定义似然核 --------------------------------------------------------

// DiffSlopeKernel 用相邻差分从观测流估计斜率 a_raw = Δy/Δx(自定义
// LikelihoodKernel #1)。观测路由为 [x_t, y_t, x_{t-1}, y_{t-1}](上一步观测由
// 驱动方随步提供), 核本身保持无状态。
struct DiffSlopeKernel {}

fn (k DiffSlopeKernel) out_dim() int {
	return 1
}

fn (k DiffSlopeKernel) step(ctx conger.KernelContext) []f64 {
	x, y := ctx.feed[0], ctx.feed[1] // 前馈父核 'obs'(SourceKernel)的当前步输出
	if ctx.t == 0 {
		return [0.0]
	}
	xp, yp := ctx.feed[2], ctx.feed[3]
	return [(y - yp) / (x - xp)]
}

// EmaKernel 指数滑动平均: out_t = out_{t-1} + α·(raw_t − out_{t-1})。raw 来自
// 前馈父核(似然估计之间的依赖), out_{t-1} 来自自反馈边(一步时滞的迭代)。
// ctx.t == 0 时直接采用 raw(初始化约定)。
struct EmaKernel {
	alpha f64
}

fn (k EmaKernel) out_dim() int {
	return 1
}

fn (k EmaKernel) step(ctx conger.KernelContext) []f64 {
	raw := ctx.feed[0]
	if ctx.t == 0 {
		return [raw]
	}
	prev := ctx.back[0]
	return [prev + k.alpha * (raw - prev)]
}

// InterceptKernel 由平滑后的斜率估计计算截距 raw = y − a·x, 再做 EMA 平滑
// (自定义 LikelihoodKernel #2; 前馈依赖 slope_ema, 自反馈迭代)。
struct InterceptKernel {
	alpha f64
}

fn (k InterceptKernel) out_dim() int {
	return 1
}

fn (k InterceptKernel) step(ctx conger.KernelContext) []f64 {
	x, y := ctx.obs[0], ctx.obs[1] // 路由到本节点的原始观测
	a := ctx.feed[0] // slope_ema 的当前步输出(前馈依赖)
	raw := y - a * x
	if ctx.t == 0 {
		return [raw]
	}
	prev := ctx.back[0] // 自身上一步输出(反馈迭代)
	return [prev + k.alpha * (raw - prev)]
}

// --- 主管线 A → B → C → D → E ---------------------------------------------------

fn main() {
	println('== A. 训练: fit_mixture_spn(确定性, 无 EM) ==')
	n := 96 // 每个机制 96 个样本(核回归需要足够密度, 见 generic_framework_test.v 的 192)
	linear_expert := conger.train_toy_expert('linear', n, 1)
	sine_expert := conger.train_toy_expert('sine', n, 2)
	println('linear 专家: K=${linear_expert.net.f_mu.dim(0)} 个实例级分量, 白化特征维 D=${linear_expert.net.f_mu.dim(1)}')
	println('sine   专家: K=${sine_expert.net.f_mu.dim(0)} 个实例级分量, 白化特征维 D=${sine_expert.net.f_mu.dim(1)}')

	println('')
	println('== B. 推理: predict → (E[t|x], P(scene|x), 责任度 r) ==')
	x := conger.toy_x()
	// 观测一条真实线性序列 y = 1.2x - 0.3
	y_lin := x.multiply(mlx.f32_scalar(1.2)).add(mlx.f32_scalar(-0.3))
	f_lin := linear_expert.family.encode(y_lin).expand_dims(0)
	tm, _, r := linear_expert.net.predict(f_lin)
	et := tm.data_f32()
	println('E[params|x] = [${et[0]:.4f}, ${et[1]:.4f}]  (真值 [1.2, -0.3])')
	println('最大责任度 max(r) = ${f64(r.max().item_f32()):.4f}  (K=${r.dim(1)})')

	println('')
	println('== C. 结构门控 GenericStructureGate + 出生 StructureBirthController ==')
	mut experts := map[string]conger.GenericExpert[voidptr]{}
	experts['linear'] = linear_expert
	experts['sine'] = sine_expert
	gate := conger.GenericStructureGate[voidptr]{
		birth_residual: 0.30
	}
	mut birth := &conger.StructureBirthController{
		min_cases: 2
		proposer:  ToyProposer{}
	}
	mut registry := conger.GenericExpertRegistry[voidptr]{
		experts:          experts
		gate:             gate
		birth_controller: birth
	}
	// C1: 已知结构 → MAP 专家
	d_lin := registry.decide(y_lin)
	println('已知观测(线性): MAP 专家 = ${d_lin.estimate.structure_id}, ' +
		'后验 linear=${d_lin.posterior['linear']:.4f} sine=${d_lin.posterior['sine']:.4f}, ' +
		'needs_new_structure=${d_lin.needs_new_structure}')
	// C2: 未知结构(二次序列)→ 残差超阈, 累积出生证据
	y_unknown := x.multiply(x).multiply(mlx.f32_scalar(1.5)).add(mlx.f32_scalar(-0.2))
	d_u1 := registry.decide(y_unknown)
	println(
		'未知观测(二次) 第 1 次: 最优残差=${conger.min_of(d_u1.residuals) or { 0.0 }:.4f} > birth_residual=0.30, ' +
		'needs_new_structure=${d_u1.needs_new_structure}, 出生请求=${registry.last_birth_request != none}')
	registry.decide(y_unknown)
	req := registry.last_birth_request or { panic('第 2 次未知观测应触发出生请求') }
	println('未知观测(二次) 第 2 次: 达到 min_cases=2 → StructureBirthRequest' +
		' (cases=${req.cases.len}, residual_mean=${req.residual_mean:.4f}, 提案=${req.proposals.len} 个)')

	println('')
	println('== D. 模板学习: TemplateDeltaLearner.tdl_learn ==')
	tdl := conger.TemplateDeltaLearner{
		min_evidence: 2
	}
	lineages := {
		'linear': conger.TemplateLineage{
			family: 'linear'
		}
	}
	specs := tdl.tdl_learn([req], lineages)
	if specs.len == 0 {
		panic('期望至少学到 1 个 ChildTemplateSpec')
	}
	spec := specs[0]
	println('ChildTemplateSpec: name=${spec.name}')
	println('  parent=${spec.parent_family} op=${spec.operation} gen=${spec.generation}' +
		' evidence=${spec.evidence_count} residual_mean=${spec.residual_mean:.4f}' +
		' scale_ratio=[${spec.constraints.scale_ratio[0]:.4f}, ${spec.constraints.scale_ratio[1]:.4f}]')
	println('  lineage: ${spec.lineage().signature()}')

	println('')
	println('== E. 持久化: save/load_mixture_spn + split_save/assemble_model 往返 ==')
	tmp := os.temp_dir()
	full_path := os.join_path(tmp, 'conger_example_model.safetensors')
	split_path := os.join_path(tmp, 'conger_example_model')
	defer {
		os.rm(full_path) or {}
		os.rm(split_path + '.transform.safetensors') or {}
		os.rm(split_path + '.components.safetensors') or {}
	}
	// E1: 单文件 save → load_mixture_spn
	linear_expert.net.save(full_path)
	loaded := conger.load_mixture_spn(full_path) or { panic(err) }
	tm2, _, _ := loaded.predict(f_lin)
	err1 := math.abs(f64(tm.subtract(tm2).abs().max().item_f32()))
	println('save → load_mixture_spn 往返: 预测最大绝对误差 = ${err1:.6f}')
	// E2: 分级 split_save → assemble_model
	conger.split_save(linear_expert.net, split_path)
	assembled := conger.assemble_model(split_path)
	tm3, _, _ := assembled.predict(f_lin)
	err2 := math.abs(f64(tm.subtract(tm3).abs().max().item_f32()))
	println('split_save → assemble_model 往返: 预测最大绝对误差 = ${err2:.6f}')
	println('(临时文件已清理: ${tmp}/conger_example_model*)')

	println('')
	println('== F. 模型内存: 按需加载 + 动态遗忘 + 基截断 ==')
	// F1: 按需加载 — 分量表常驻、白化基按需(复用 E 阶段写出的 split 文件)。
	f_mean2, basis2 := conger.load_transform(split_path)
	comp2, meta2 := conger.load_components(split_path)
	k2 := (comp2['f_mu'] or { mlx.zeros([1, 1], .float32) }).dim(0)
	println('load_transform: 仅加载白化变换 f_mean(${f_mean2.dim(0)},) + basis ${basis2.dim(0)}x${basis2.dim(1)}')
	println('load_components: 仅加载分量表 K=${k2}(不含 basis), meta n_stratum=${meta2.n_stratum}')
	// F2: 动态遗忘 — coreset 最远距离采样, 按 stratum 比例驱逐分量。
	full := conger.assemble_model(split_path)
	println('assemble_model 全量装配: K=${full.f_mu.dim(0)}, 大小=${conger.model_size_mb(full):.6f} MB')
	forgotten := conger.forget_components(full, 24, 'coreset', 7)
	println('forget_components(k_max=24, coreset): K=${forgotten.f_mu.dim(0)}, 大小=${conger.model_size_mb(forgotten):.6f} MB')
	tm4, _, _ := forgotten.predict(f_lin)
	err3 := math.abs(f64(tm.subtract(tm4).abs().max().item_f32()))
	et4 := tm4.data_f32()
	println('遗忘后预测 E[params|x] = [${et4[0]:.4f}, ${et4[1]:.4f}], 与完整模型误差 = ${err3:.4f}')
	// F3: 基截断 — 只保留最高方差的 d_max 个白化方向。
	truncated := conger.truncate_basis(full, 2)
	println('truncate_basis(d_max=2): 白化特征维 D=${full.f_mu.dim(1)} → ${truncated.f_mu.dim(1)}, 大小=${conger.model_size_mb(truncated):.6f} MB')

	println('')
	println('== G. 似然核网络: 自定义 LikelihoodKernel · 前馈依赖 · 反馈迭代 ==')
	// 图结构: obs(SourceKernel) → slope_raw(差分) → slope_ema(EMA, 自反馈)
	//                                      └──────→ b_ema(截距, 自反馈)
	// 前馈边 = 似然估计之间的依赖(截距依赖斜率的当前步估计, 必须无环);
	// 反馈边 = 迭代关系(EMA 注入自身上一步输出, 允许成环, 一步时滞打破代数环)。
	graph := conger.KernelGraph{
		nodes: {
			'obs':       conger.KernelNode{
				kernel: conger.SourceKernel{
					dim: 4 // [x_t, y_t, x_{t-1}, y_{t-1}]
				}
			}
			'slope_raw': conger.KernelNode{
				kernel:  DiffSlopeKernel{}
				parents: ['obs']
			}
			'slope_ema': conger.KernelNode{
				kernel:   EmaKernel{
					alpha: 0.3
				}
				parents:  ['slope_raw']
				feedback: ['slope_ema']
			}
			'b_ema':     conger.KernelNode{
				kernel:   InterceptKernel{
					alpha: 0.3
				}
				parents:  ['slope_ema']
				feedback: ['b_ema']
			}
		}
	}
	order := conger.topo_order(graph.nodes) or { panic(err) }
	println('拓扑序(仅前馈边): ${order}; 反馈边: slope_ema→slope_ema, b_ema→b_ema(自反馈)')
	// 沿 32 点网格逐步送入同一条线性观测 y = 1.2x − 0.3, 迭代估计收敛到真值。
	xs := x.data_f32()
	ys := y_lin.data_f32()
	mut steps := []map[string][]f64{cap: xs.len}
	for t in 0 .. xs.len {
		tprev := if t > 0 { t - 1 } else { 0 }
		steps << {
			'obs':   [f64(xs[t]), f64(ys[t]), f64(xs[tprev]), f64(ys[tprev])]
			'b_ema': [f64(xs[t]), f64(ys[t])]
		}
	}
	trace := conger.run_recurrent(graph, steps) or { panic(err) }
	for t in [0, 8, 16, 31] {
		a := trace.output(t, 'slope_ema')[0]
		b := trace.output(t, 'b_ema')[0]
		println('  t=${t:2}: 斜率 a=${a:.4f}  截距 b=${b:.4f}')
	}
	a_last := trace.output(31, 'slope_ema')[0]
	b_last := trace.output(31, 'b_ema')[0]
	println('迭代收敛: a=${a_last:.4f} b=${b_last:.4f}  (真值 a=1.2, b=-0.3)')

	println('')
	println('主管线 A→B→C→D→E + F(模型内存) + G(似然核网络) 全部完成。')
}
