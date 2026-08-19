module main

// mrf_lattice.v — MRF 似然核(mrf_kernels.v)的领域中性演示:在二维格点上用
// 似然核网络做空间上下文协同估计。
//
//   A. GMRF 去噪:8×8 连续场(左半 -1 / 右半 +1,加观测噪声),每个格点一个
//      GMRFKernel,4-邻域反馈边,阻尼不动点迭代收敛到联合后验均值 E[x|y];
//      对照 run_residual 残差驱动异步调度的更新次数。
//   B. Potts 平滑:同布局的二类离散场,每点只有带噪的局部逐类证据,Potts
//      核经邻域投票恢复整片标签。
//   C. 参数学习:EMLoop 驱动 GMRF 核网络(E 步 = 阻尼松弛取后验均值,M 步 =
//      伪似然最小二乘),从带噪观测自动学得 μ/β/σ² 并用学得参数重新去噪。
//
// 运行: v -gc boehm -no-memory-limit run examples/mrf_lattice.v
import math
import conger

const rows = 8
const cols = 8

fn main() {
	gmrf_demo()
	potts_demo()
	learning_demo()
}

// 真值场:左半 -1,右半 +1
fn truth(_r int, c int) f64 {
	return if c < cols / 2 { -1.0 } else { 1.0 }
}

fn gmrf_demo() {
	println('== A. GMRF 连续场去噪(${rows}×${cols} 格点) ==')
	beta := 0.35
	sig2 := 0.5 // 空间条件方差 σ²
	tau2 := 0.64 // 观测噪声 τ²
	nodes := conger.grid4_nodes(rows, cols, fn [beta, sig2, tau2] (r int, c int, n_neighbors int) conger.LikelihoodKernel {
		return conger.new_gmrf_kernel(0.0, math.log(sig2), math.log(tau2),
			[beta].repeat(n_neighbors))
	})
	g := conger.KernelGraph{
		nodes: nodes
	}
	println('反馈环节点数: ${conger.feedback_cycle_nodes(g).len}/${nodes.len}(全格点成环)')

	mut rng := conger.new_rng(42)
	mut obs0 := map[string][]f64{}
	mut mse_noisy := 0.0
	for r in 0 .. rows {
		for c in 0 .. cols {
			y := truth(r, c) + rng.normal(0.0, math.sqrt(tau2))
			obs0[conger.grid4_name(r, c)] = [y]
			mse_noisy += (y - truth(r, c)) * (y - truth(r, c))
		}
	}
	n := f64(rows * cols)
	println('观测 RMSE: ${math.sqrt(mse_noisy / n):.4f}')

	mut obs := []map[string][]f64{cap: 400}
	for _ in 0 .. 400 {
		obs << obs0
	}
	trace := conger.run_recurrent_opts(g, obs, conger.RecurrentOptions{
		damping: 0.4
		tol:     1e-10
	}) or { panic(err) }
	last := trace.steps.len - 1
	mut mse_est := 0.0
	mut pll_first := 0.0
	mut pll_last := 0.0
	for r in 0 .. rows {
		for c in 0 .. cols {
			name := conger.grid4_name(r, c)
			est := trace.output(last, name)
			mse_est += (est[0] - truth(r, c)) * (est[0] - truth(r, c))
			pll_first += trace.output(0, name)[1]
			pll_last += est[1]
		}
	}
	println('同步阻尼迭代: ${trace.steps.len} 步, converged=${trace.converged}')
	println('估计 RMSE: ${math.sqrt(mse_est / n):.4f}')
	println('伪似然(全格点求和): 第 0 步 ${pll_first:.2f} → 收敛后 ${pll_last:.2f}')

	rt := conger.run_residual(g, obs0, conger.RecurrentOptions{
		damping: 0.4
		tol:     1e-10
	}, 100000) or { panic(err) }
	println('残差驱动异步调度: ${rt.steps.len - 1} 次单点更新, converged=${rt.converged}(同步扫描约 ${(trace.steps.len - 1) * rows * cols} 次)')
	println('')
}

fn potts_demo() {
	println('== B. Potts 二类场平滑(${rows}×${cols} 格点) ==')
	j_coupling := 1.5
	nodes := conger.grid4_nodes(rows, cols, fn [j_coupling] (r int, c int, n_neighbors int) conger.LikelihoodKernel {
		return conger.new_potts_kernel([0.0, 0.0], [j_coupling].repeat(n_neighbors))
	})
	g := conger.KernelGraph{
		nodes: nodes
	}
	// 局部证据:观测 y = 真值标签均值(-1/+1) + 噪声 → 逐类对数似然
	mut rng := conger.new_rng(7)
	tau2 := 0.64
	mut obs0 := map[string][]f64{}
	mut raw_err := 0
	for r in 0 .. rows {
		for c in 0 .. cols {
			y := truth(r, c) + rng.normal(0.0, math.sqrt(tau2))
			d0, d1 := y + 1.0, y - 1.0
			obs0[conger.grid4_name(r, c)] = [
				-0.5 * d0 * d0 / tau2,
				-0.5 * d1 * d1 / tau2,
			]
			if (y > 0) != (truth(r, c) > 0) {
				raw_err++
			}
		}
	}
	println('局部证据直接 argmax 错误数: ${raw_err}/${rows * cols}')

	mut obs := []map[string][]f64{cap: 300}
	for _ in 0 .. 300 {
		obs << obs0
	}
	trace := conger.run_recurrent_opts(g, obs, conger.RecurrentOptions{
		damping: 0.3
		tol:     1e-10
	}) or { panic(err) }
	last := trace.steps.len - 1
	mut err := 0
	for r in 0 .. rows {
		mut line := ''
		for c in 0 .. cols {
			out := trace.output(last, conger.grid4_name(r, c))
			pred := if out[1] > out[0] { 1 } else { 0 }
			if f64(pred) != (truth(r, c) + 1.0) / 2.0 {
				err++
			}
			line += if pred == 1 { '#' } else { '.' }
		}
		println('  ${line}')
	}
	println('MRF 平滑后错误数: ${err}/${rows * cols}(${trace.steps.len} 步收敛, converged=${trace.converged})')
}

fn learning_demo() {
	println('== C. GMRF 参数学习(EMLoop × 核网络) ==')
	tau2 := 0.64 // 观测噪声(假设已知,学习时不拟合)
	mut rng := conger.new_rng(42)
	mut y := []f64{cap: rows * cols}
	for r in 0 .. rows {
		for c in 0 .. cols {
			y << truth(r, c) + rng.normal(0.0, math.sqrt(tau2))
		}
	}
	l := conger.new_gmrf_learner(rows, cols, math.log(tau2))
	mut loop := conger.EMLoop[conger.GMRFLearner, []f64, conger.GMRFMeans]{
		max_iters: 40
		tol:       1e-4
		damping:   0.2
		model:     l
	}
	init := [0.0, 0.05, math.log(2.0)]
	println('初始: mu=0 β=0.05 σ²=2,PL(y)=${l.log_likelihood(init, y):.2f}')
	res := loop.run(y, init)
	mu_l, beta_l, sig2_l := res.params[0], res.params[1], math.exp(res.params[2])
	println('学得: mu=${mu_l:.3f} β=${beta_l:.3f} σ²=${sig2_l:.3f}(${res.iterations} 轮,PL(y)=${res.log_likelihood:.2f})')

	// 用学得参数重新去噪(不再手工调参)
	nodes := conger.grid4_nodes(rows, cols, fn [mu_l, beta_l, sig2_l, tau2] (r int, c int, n_neighbors int) conger.LikelihoodKernel {
		return conger.new_gmrf_kernel(mu_l, math.log(sig2_l), math.log(tau2),
			[beta_l].repeat(n_neighbors))
	})
	mut obs0 := map[string][]f64{}
	for r in 0 .. rows {
		for c in 0 .. cols {
			obs0[conger.grid4_name(r, c)] = [y[r * cols + c]]
		}
	}
	mut obs := []map[string][]f64{cap: 400}
	for _ in 0 .. 400 {
		obs << obs0
	}
	trace := conger.run_recurrent_opts(conger.KernelGraph{
		nodes: nodes
	}, obs, conger.RecurrentOptions{
		damping: 0.4
		tol:     1e-10
	}) or { panic(err) }
	last := trace.steps.len - 1
	mut mse := 0.0
	for r in 0 .. rows {
		for c in 0 .. cols {
			est := trace.output(last, conger.grid4_name(r, c))[0]
			mse += (est - truth(r, c)) * (est - truth(r, c))
		}
	}
	println('学得参数去噪 RMSE: ${math.sqrt(mse / f64(rows * cols)):.4f}(A 部分手工调参为 0.5082)')
}
