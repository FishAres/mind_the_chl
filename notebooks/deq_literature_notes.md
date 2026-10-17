# Literature notes: deep equilibrium models (DEQs) for CHL gap filling

Compiled 2026-10-11 from two delegated web searches (DEQ methodology; fixed-point and learned-solver methods in ocean gap filling), then cross-checked against our own code. Companion to `pinn_literature_notes.md`.
Verification key: [V] abstract or paper fetched and read by the search agent; [S] search snippets only; [U] own arithmetic or reading of our code.

## 0. Short version

1. **No published DEQ or implicit-layer model exists for satellite gap filling, cloud removal, data assimilation or ocean colour.** Four extended searches found none. The closest references are FNO-DEQ (steady-state PDEs on a 2D grid) and the imaging-inverse-problem DEQs (Gilton et al. 2021; ELDER; i-DEQ), which include inpainting. A DEQ gap filler would be new.
2. **Our spatial nODE already is a DEQ, a linear one.** `chl_cnn/spatial_node.py` solves `A(θ,x) C = b` by batched CG and backpropagates with the implicit function theorem (one more CG solve). That is exactly the DEQ recipe (fixed point + IFT backward) for the affine map `f(C) = C - η(A C - b)`. The literature on DEQ vs neural ODE (Pal et al. 2022; Ding et al. 2023) says: when only the steady state matters, solve for the fixed point; integrating the trajectory is 2-4x more expensive and adds nothing. We made that choice on 2026-10-06 without the vocabulary. [U, V]
3. So the question is not "DEQ vs nODE" but **"linear fixed-point map vs learned nonlinear fixed-point map"**. The nonlinear version is what the imaging DEQs do: one step of gradient descent on a data term, plus a learned CNN prior/source, iterated to convergence. FINDINGS.md says the CNN beats the linear-Gaussian model by 0.02-0.03 at every distance including distance 1, which is the signal a learned nonlinear step could pick up.
4. **Cheap baseline first.** FNO-DEQ found that the same block weight-tied and unrolled a fixed 6-12 times matched the true DEQ on Darcy flow; 4DVarNet saw no gain beyond 5 unrolled steps on SSH. A weight-tied unrolled CNN with input injection is the baseline a root-finding DEQ must beat.
5. **Every ocean gap-filling method that beats a single-pass CNN is iterative**: DINEOF (fixed-point EOF completion), SPDE/GMRF optimal interpolation (a sparse elliptic solve), 4DVarNet (unrolled learned gradient solver). 4DVarNet reports the best CHL numbers in the literature (RMSLE 0.058 log10 vs DINEOF 0.112 vs UNet 0.095 at ~1 km, Gulf of Lion).

## 1. How the existing code maps onto a DEQ  [U]

| DEQ ingredient | `spatial_node.py` today | Nonlinear DEQ generalisation |
|---|---|---|
| state z | ln CHL field C on free (unobserved ocean) pixels | same, or C plus a few latent channels (MDEQ-style) |
| fixed-point map f(z, x) | affine: `z - η(A(θ,x) z - b)`, A = Gᵀ(aD)G + aβ SPD | `z - η ∇_z E(z) - η g_θ(z, covariates, context)` or `CNN_θ(z, mask, covariates)` with observed pixels overwritten |
| input injection x | D(x) from the D-net, prior b from climatology and context days | same covariates fed to the cell at every iteration (FNO-DEQ style) |
| forward solver | CG, tol 1e-6, ≤500 iterations | Anderson acceleration, 20-30 iterations, tol 1e-2 to 1e-3 |
| backward | exact IFT via CG on the same SPD operator | IFT via Anderson, or phantom gradient / Jacobian-free (one step through f at z*) |
| uniqueness | guaranteed (SPD) | only if f is contractive; enforced via spectral norm, monotone parameterisation, or Jacobian regularisation |
| warm start | x0 = prior | day t from day t-1's fixed point (StreamDEQ, DEQ-Flow) |

The linear model's limitation is structural: the fill at a pixel is a fixed linear combination of observed pixels and the prior, with weights set by D and β. A learned source `g_θ(z, c)` that depends on the current fill z makes the map nonlinear, and the fixed point is then no longer a Gaussian posterior mean. IDEAS.md §2.5 proposed a source `g_θ(x)` of covariates only; the DEQ form lets it depend on the state too.

## 2. Core DEQ machinery (what to use, what breaks)

**Bai, Kolter, Koltun 2019, "Deep Equilibrium Models", NeurIPS.** [V] https://arxiv.org/abs/1909.01377
Replace a stack of weight-tied layers by the fixed point z* = f(z*, x), found by root finding (Broyden). Backward uses the IFT: solve one linear system in J = ∂f/∂z at z*. Memory O(1) in depth. No convergence guarantee in general.

**Bai, Koltun, Kolter 2020, "Multiscale DEQ" (MDEQ), NeurIPS.** [V] https://arxiv.org/abs/2006.08656
State is a set of feature maps at several resolutions driven to equilibrium together. First implicit model at ImageNet/Cityscapes scale. Relevant here as the way to propagate information across long spatial ranges in one solve (our gaps are up to hundreds of km).

**Kolter, Duvenaud, Johnson 2020, "Deep implicit layers" tutorial.** [V] https://implicit-layers-tutorial.org/
Reference PyTorch DEQ in ~50 lines: Anderson acceleration (memory 5, damping β, max_iter 25-50, tol 1e-2) forward; the backward solves `g = Jᵀg + y` with the same solver via autograd VJPs. Plain fixed-point iteration needed ~2x the iterations of Anderson. Stability: GroupNorm inside the cell, small-variance init. Stability is "largely an empirical issue".

**Bai, Koltun, Kolter 2021, "Stabilizing equilibrium models by Jacobian regularization", ICML.** [V] https://arxiv.org/abs/2106.14342
Without regularisation the spectral radius of J grows during training, the solver needs >3x more iterations, and training can diverge. Fix: add γ‖εᵀJ‖²/d (one Hutchinson sample) to the loss with probability p = 0.05-0.4. Cost +25-30 % memory, ~one extra backward. Solver evaluations fell 27-30 → 14-15 (ImageNet).

**Winston & Kolter 2020, "Monotone operator equilibrium networks", NeurIPS.** [V] https://arxiv.org/abs/2006.08591
Parameterise W = (1-m)I - AᵀA + B - Bᵀ so the operator is strongly monotone: fixed point exists, is unique, and operator-splitting converges linearly. Multiscale conv version solves the linear part by FFT. Trails unconstrained DEQs in accuracy. This is the choice if we want the guarantee we have today (SPD) to survive the move to a nonlinear map.

**Fung et al. 2022, "Jacobian-free backpropagation" (JFB), AAAI.** [V] https://arxiv.org/abs/2103.12803
Drop (I-J)⁻¹ from the IFT gradient: backprop through a single application of f at z*. Provably a descent direction when f is contractive. Epoch time 531 s → 147 s on CIFAR-10, no accuracy loss. Bolte et al. 2023 [S] show the error of this one-step estimator scales with the inner solver's convergence rate.

**Geng et al. 2021, "On training implicit models" (phantom gradient), NeurIPS.** [V] https://arxiv.org/abs/2111.05177
Inexact gradient from k damped unrolled steps after the solve, or a truncated Neumann series; JFB is k = 1. Backward ~1.7x faster than exact IFT, better ImageNet accuracy. Default in TorchDEQ.

**Geng & Kolter 2023, TorchDEQ.** [V] https://arxiv.org/abs/2310.18605 , `pip install torchdeq`
Solvers: fixed-point, Anderson, Broyden. Gradients: IFT, phantom, BPTT. Normalisation: weight norm, spectral norm with learnable scale, norm clipping. Example settings: 20 forward iterations tol 1e-2, 30 backward tol 1e-3, Anderson damping 0.8, phantom damping 0.6. Recommended stability toolbox: normalise weights so plain iteration converges, Jacobian regularisation, fixed-point correction, random iteration counts.

**Bai, Koltun, Kolter 2022, "Neural deep equilibrium solvers", ICLR.** [S] https://iclr.cc/virtual/2022/poster/5919
Learn the initial guess and a learnable Anderson update: ~2x faster inference at equal accuracy for ~1-3 % extra parameters.

**McCallum, Arora, Foster 2025, "Reversible DEQs".** [V] https://arxiv.org/abs/2509.12917
Argues the usual DEQ gradients are approximations and that this causes instability; a reversible formulation gives exact gradients with fewer function evaluations. Recent, not replicated.

**DEQ vs neural ODE.** Pal, Edelman, Rackauckas 2022, "Continuous DEQs" [V] https://arxiv.org/abs/2201.12240 : a DEQ is a neural ODE integrated to t → ∞; solving the steady state directly with a linear-system backward cut training cost 2-4x; their Skip-DEQ adds an explicit predictor then an implicit correction. Ding et al. 2023, "Two sides of the same coin" (HomoODE), NeurIPS [V] https://arxiv.org/abs/2310.09583 : both are equilibrium finding via homotopy continuation. Takeaway: integrate a trajectory only if intermediate pseudo-time carries meaning. Ours does not.

### Reported costs and failure modes

| item | literature | source |
|---|---|---|
| forward iterations | 16-32 Anderson (FNO-DEQ), 20 (TorchDEQ), 25-50 (tutorial), up to 100 in training (Gilton) | [V] |
| backward | same solver 30 iterations tol 1e-3, or phantom/JFB; JFB 3.6x faster epochs | [V] |
| memory | O(1) in depth; 4-6x less than recurrent nets of equal effective depth | [V] |
| wall clock vs explicit | training 1.5-3x slower (DEQ-Transformer, FNO-DEQ); inference slower since f is evaluated 20-30 times | [V] |
| instability | Jacobian spectral radius grows, solver stops converging, loss diverges; i-DEQ reports PSNR collapse at 200 iterations and ~1 dB loss without a grid-searched init; training failed on inpainting at some noise levels | [V] Bai 2021, i-DEQ |
| uniqueness | guaranteed only for monotone or contractive parameterisations | [V] |
| warm start across time | StreamDEQ (ECCV 2022, https://arxiv.org/abs/2204.13492 ) and DEQ-Flow (CVPR 2022, https://arxiv.org/abs/2204.08442 ) start frame t from frame t-1's fixed point: a few iterations per frame, >3x faster at equal accuracy | [V] |
| test-time compute | more iterations at inference reduce error (DEQ) whereas unrolled nets degrade if run past their training depth | [V] Gilton, FNO-DEQ |
| input masks | no paper studies sensitivity of a DEQ to the observation mask | — |

## 3. DEQs for PDEs and neural operators

**Marwah et al. 2023, "Deep equilibrium based neural operators for steady-state PDEs" (FNO-DEQ), NeurIPS.** [V] https://arxiv.org/abs/2312.00234
Steady-state PDE solutions are fixed points, so one FNO block is weight-tied with input injection (the coefficient field projected and added at every application) and solved to equilibrium. Anderson 16-32 steps forward, phantom gradient backward. Darcy relative L2 0.0055 (2.4 M params) vs 0.0064 for a 4-block FNO (7.7 M); Navier-Stokes 0.128 vs 0.151. More robust to noisy training data. Training up to 2.5x slower. **A weight-tied FNO unrolled a fixed number of times matched FNO-DEQ on Darcy.** Failure mode: a shallow cell with noisy data became unstable, attributed to the cell not reaching the fixed point so implicit-gradient errors accumulate.

**Pacheco & Camponogara 2024, "Physics-informed DEQ" (PIDEQ), CASE.** [V] https://arxiv.org/abs/2406.03472
DEQ output + PINN residual loss, but only for ODE initial-value problems (Van der Pol). No PDE result. No other physics-informed DEQ for PDEs was found. "PINN-style DEQ" on a 2D field is open ground.

**He, Liu, Xu 2025, "Self-composing neural operators".** [V] https://arxiv.org/abs/2508.20650
Repeated application of one multigrid-inspired block, depth increased during training; large gains over FNO on Helmholtz. Explicit weight-tied unrolling, same idea.

**Liu, Ding, Osher, Yin 2025, "Expressive power of implicit models".** [V] https://arxiv.org/abs/2510.03638
Expressive power grows with test-time iterations then saturates (image reconstruction and scientific computing experiments).

**Implicit graph networks.** Gu et al. 2020, IGNN, NeurIPS [V] https://arxiv.org/abs/2009.06211 : node states are the fixed point X = φ(W X A + b(U)); a Perron-Frobenius bound on ‖W‖·λ(A) gives well-posedness, enforced by projected gradient. Liu et al. 2021, EIGNN, NeurIPS [V]: the linear case has a closed form via eigendecomposition of A. On a regular grid, IGNN is a learned diffusion; a convolutional DEQ is the same object with the grid adjacency baked into the stencil. Our SPD operator is the special case where φ is the identity and W is the diffusion stencil.

Not found: a learned preconditioner or learned linear solver for elliptic problems framed as a DEQ.

## 4. DEQs for inverse problems: the template for gap filling

**Gilton, Ongie, Willett 2021, "Deep equilibrium architectures for inverse problems in imaging", IEEE TCI.** [V] https://arxiv.org/abs/2102.07944
Three fixed-point maps with forward operator A, data y, step η, learned CNN R_θ (spectrally normalised DnCNN):
- DE-Grad: `f(x) = x + η Aᵀ(y - Ax) - η R_θ(x)`
- DE-Prox: `f(x) = R_θ(x + η Aᵀ(y - Ax))`
- DE-ADMM: fixed point over (x, u) with an (I + αAᵀA)⁻¹ step.
Anderson (m = 5), stop at relative change 1e-3, cap 100 in training; backward IFT with Anderson capped at 50. Broyden slightly worse and slower. 4x MRI PSNR 33.7 vs 32.6 for the best unrolled net; deblurring tied. Unrolled nets degrade when run past training depth; the DEQ keeps quality past 100 iterations. Contraction: DE-Grad needs η < 1/(L+1) and (R_θ - I) ε-Lipschitz; DE-Prox's guarantee needs a trivial nullspace, which fails for undersampled A, yet it still converged empirically. Init from denoisers pretrained at several noise levels.

For gap filling A = M, the 0/1 cloud mask, so L = 1 and μ = 0: DE-Grad's condition applies (η < 1/2, contractive residual), DE-Prox's does not.

**Zou, Liu, Wohlberg, Kamilov 2023, ELDER, IEEE OJSP.** [V] https://arxiv.org/abs/2303.05386
Learns an explicit CNN regulariser by DEQ training. Includes inpainting: the data step simply overwrites observed pixels with the data and keeps the network output elsewhere [S]. This is the simplest possible cell for us.

**Clerc et al. 2026, i-DEQ, NeurIPS.** [V] https://arxiv.org/abs/2605.19705
Inertial extrapolation + gradient step on data fidelity + learned gradient-step regulariser; JFB backward; adaptive restart; pretrained-denoiser init. Tasks include inpainting with 50 % pixels removed. Reports where the classical DEQ failed (unstable at 100 iterations, collapse at 200, training failure on inpainting at low noise); i-DEQ trained in all cases, 2-5x faster.

**Pramanik, Zimmerman, Jacob 2022, MOL.** [V] https://arxiv.org/abs/2206.04797
Learned monotone CNN step alternating with a CG data-consistency step inside a DEQ; unique fixed point, robustness to perturbations; O(1) memory enabled 3D and 2D+time MRI. Note [S]: for ill-conditioned A the convergence condition forces the CNN's Lipschitz constant toward zero. The "CG data step + learned step" alternation is structurally our solver plus a learned correction.

**Riccio, Ehrhardt, Benning 2022/24.** [V] https://arxiv.org/abs/2206.13193
Bilevel learning of variational regularisers is a special case of DEQ training; comparison on denoising, inpainting, deconvolution.

**Ryu et al. 2019, "Plug-and-play methods provably converge with properly trained denoisers", ICML.** [S] https://arxiv.org/abs/1905.05406
Real spectral normalisation for Lipschitz-bounded denoisers; the condition Gilton reuses.

Remote-sensing adjacent: Gkillas et al. 2023, DEQ sparse coding for hyperspectral denoising, IEEE TIP [V] https://arxiv.org/abs/2203.15901 ; Li & Yaghoobi 2025, PnP-ADMM + deep image prior with a 1-Lipschitz constraint and fixed-point proof for hyperspectral inpainting (EMIT), Remote Sensing [S] https://arxiv.org/abs/2501.08195 .

## 5. Ocean gap-filling methods, read as fixed-point methods

| method | iteration structure | learned part | memory in depth |
|---|---|---|---|
| DINEOF (Beckers & Rixen 2003) | fill → rank-k SVD → refill, to convergence: `x* = P_Ω(y) + (I-P_Ω) SVD_k(x*)` | none (k by cross-validation) | n/a |
| SPDE / GMRF OI (Lindgren, Rue, Lindström 2011) | sparse elliptic solve = fixed point of CG/Jacobi | none; Deep GMRF (Sidén & Lindsten 2020) learns a CNN precision | n/a |
| our spatial nODE | CG on SPD operator, IFT backward | D(x), β, context weights | O(1) |
| 4DVarNet (Fablet 2021; Beauchamp 2023) | 5-15 unrolled ConvLSTM gradient steps on `‖Hx-y‖² + λ‖x-Φ(x)‖²`, BPTT | prior Φ and the solver | ∝ depth |
| neural SPDE 4DVarNet (Beauchamp 2023/25) | same unroll; prior is a learned fractional advection-diffusion SPDE | SPDE coefficients κ, m, H, τ | ∝ depth |
| DINCAE 2.0 refinement | 2 autoencoders in sequence, unshared weights | both | ∝ depth |
| DINCAE 1.0, UNet, OCNET, SIREN, XGBoost | single pass | all | — |
| diffusion ensembles (Barth 2024) | long stochastic chain, not an equilibrium | denoiser | — |

Key references, with numbers:

**Beckers & Rixen 2003, JAOT; Alvera-Azcárate et al. 2005, Ocean Modelling.** [V] https://doi.org/10.1016/j.ocemod.2004.08.001
DINEOF convergence is observed, not proven. Adriatic AVHRR SST, 61 % cloud: CV error 0.6 °C, ~30x faster than OI. Multivariate variant (2007, [S]) stacks SST + CHL + wind. DINEOF with error maps: Beckers et al. 2006, Ocean Science [V] https://doi.org/10.5194/os-2-183-2006 .

**Barth et al. 2020, 2022, DINCAE 1.0 / 2.0, GMD.** [V] https://doi.org/10.5194/gmd-13-1609-2020 , https://doi.org/10.5194/gmd-15-2183-2022
Conv autoencoder with inverse-error-variance input channel, Gaussian NLL output, t±1 context, extra random masks. Provençal SST CV RMSE: DINEOF 0.463 → DINCAE 0.384 °C. 2.0: sum skip connections, multivariate (CHL and wind "improved results only marginally" for SST), and a refinement step (second autoencoder); the code supports an arbitrary number of sequential refinement steps, only one tested.

**Mehdipour et al. 2026, GMD 19, 1619.** [V] https://doi.org/10.5194/gmd-19-1619-2026
DINEOF vs DINCAE, Sentinel-3 OLCI CHL, Atlantic, 4 km. CV RMSLE (log10): DINEOF 0.05-0.16, DINCAE 0.03-0.12. In-situ regression favoured DINEOF; conclusion: DINEOF more reliable and cheaper, DINCAE better on transient features.

**Barth et al. 2024, Ocean Science 20, 1567: diffusion ensemble, Black Sea CHL at 300 m.** [V] https://doi.org/10.5194/os-20-1567-2024
U-Net DDPM trained on incomplete OLCI L3, 64-member ensembles; ensemble-mean RMSE "slightly smaller" than DINCAE; single members 0.20-0.21 log10; inference 30 min vs 2.7 s. Directly relevant to our BLK store.

**Fablet et al. 2021, JAMES; Beauchamp et al. 2023, GMD 16, 2119 (4DVarNet-SSH).** [V] https://arxiv.org/abs/2007.12941 , https://doi.org/10.5194/gmd-16-2119-2023
Fixed 5 unrolled iterations: "more iterations increased training time without a clear accuracy gain". Gulf Stream OSSE resolved scale 1.42° (OI) → 0.83°. No implicit or equilibrium 4DVarNet variant exists.

**Beauchamp, Fablet, Georgenthum 2023/25, neural SPDE prior.** [V] https://arxiv.org/abs/2311.01783
Learned fractional advection-diffusion SPDE as the 4DVarNet prior, sparse precision from backward-Euler transition matrices; SSH resolved scale 0.90° vs 1.17° for a UNet prior. The direct neural descendant of the Lindgren-Rue-Lindström link, and the closest published relative of our D-tensor model.

**Nguyen, Dorffer, Jourdin, Fablet 2025, IEEE JSTARS.** [V] https://arxiv.org/abs/2503.11588
CMEMS L3 CHL, Gulf of Lion (~45 % missing) and Wadden Sea (~60 %), ~1 km. RMSLE (log10):

| variable / region | DINEOF | UNet | 4DVarNet |
|---|---|---|---|
| CHL, Gulf of Lion | 0.112 | 0.095 | 0.058 |
| bbp443, Gulf of Lion | 0.080 | 0.075 | 0.050 |
| bbp443, Wadden Sea | 0.17 | – | 0.12 |

4DVarNet trained on Mediterranean bbp443 transferred to CHL (0.061) and to the North Sea (0.13) without fine-tuning. Dorffer et al. 2025, TGRS [V] https://arxiv.org/abs/2503.11532 train 4DVarNet with observations only (masking rectangular patches), beating DINEOF and UNet.

**Sidén & Lindsten 2020, Deep GMRF, ICML.** [V] https://proceedings.mlr.press/v119/siden20a.html
A GMRF's data → latent map is a one-layer linear CNN; stacking gives higher-order GMRFs; variational inference; beat SOTA on a satellite land-surface-temperature gap dataset.

**Stock et al. 2020, Remote Sensing 12, 3313.** [V] https://www.mdpi.com/2072-4292/12/20/3313
Ten-algorithm CHL cloud-filling benchmark, MODIS 4-9 km: ordinary kriging lowest RMSE in 3 of 4 regions; differences among the best small; DINEOF "worked well across study areas".

**Hong et al. 2023, ESSD 15, 5281 (OCNET).** [V] https://doi.org/10.5194/essd-15-5281-2023
Global 0.25° daily gap-free CHL from 405 regional U-Nets + SST/salinity/PAR; log10 RMSE "<0.2, mostly <0.1". Same resolution as our IO store.

Not found: any western Indian Ocean / Arabian Sea deep-learning CHL gap-filling paper with log errors (only a 2018 DINEOF Arabian Sea study [S]); any GeoSMART "mind the chl gap" publication.

**Unit note [U]:** literature RMSLE is in log10; our MAE is in ln. Divide ln by 2.303 for log10. Our dilated CNN val MAE 0.1365 ln ≈ 0.059 log10 MAE at 0.25°; RMSE would be higher. Not comparable across regions or resolutions; listed only for scale.

## 6. What this suggests for the project (proposals, not results)

Ordered by cost. Each is one trainer option on the existing pipeline unless stated.

**6a. Weight-tied unrolled CNN, fixed depth (the baseline).** [S-class cost]
Cell `z ← CNN_θ(z, mask, visible CHL, covariates, context)`, observed pixels overwritten after each step (ELDER's inpainting prox), warm start from climatology or the linear fill, K = 6-12 steps, plain BPTT. Same parameter count as the cell, memory ∝ K. If this does not move the 0.1365 the DEQ machinery will not either (FNO-DEQ and 4DVarNet evidence). Also gives the first look at whether iterates converge, i.e. whether a fixed point exists to be found.

**6b. Same cell as a true DEQ.** [S-M]
TorchDEQ: Anderson forward 20-30 iterations, phantom gradient backward, spectral norm on the cell from the start, Jacobian regularisation if solver iterations grow. Score vs 6a at equal wall clock, and the test-time curve of MAE vs iterations (the DEQ-specific property).

**6c. Linear solver + learned state-dependent source (the smallest nonlinear step from today's code).** [M]
`f(z) = z - η[ A(θ,x) z - b ] - η g_θ(z, c)`, i.e. DE-Grad with our SPD energy as the data term and a bounded local CNN source as the prior. This is IDEAS.md §2.5 with the source allowed to see the current fill. Can be solved MOL-style: alternate one CG solve with a learned step. The SPD part keeps the map well-conditioned; the Lipschitz bound on g_θ (spectral norm, bounded output) gives contraction. Directly tests the distance-1 gap hypothesis in FINDINGS.md.

**6d. Daily warm start.** [S on top of 6b/6c]
Start day t from day t-1's equilibrium (StreamDEQ): fewer iterations and a causal fill that doubles as a forecast; the state-space idea in IDEAS.md §2.1 in DEQ clothing.

**6e. Physics as structure inside the cell, not as a penalty.** [M-L, Black Sea]
The only "PINN + DEQ" paper is on ODEs. For us the physics is already an operator (anisotropic diffusion, and semi-Lagrangian transport on the 2 km grid); a soft transport residual on the CNN changed val MAE by ≤ 0.001, so the literature-consistent route is 6c with transport inside A, not a PDE loss on a DEQ output.

**Risks reported in the literature that apply here:** growing solver iteration counts during training (monitor NFE and ‖J‖ per epoch); collapse at large iteration budgets; dependence on init (i-DEQ: ~1 dB); 1.5-3x training wall clock; no paper on mask sensitivity, and our masks are 70 % missing with real cloud structure, well beyond the 50 % random inpainting in i-DEQ.

## 7. Open questions this search could not answer

- Does a nonlinear fixed point beat a weight-tied unroll on a 70 %-missing, structured-mask problem? No published data point.
- How many iterations does a convolutional cell need to propagate across a 500 km gap at 0.25°, and does MDEQ-style multiscale state remove that dependence?
- Whether 4DVarNet's "no gain past 5 steps" holds for CHL or is specific to SSH.
