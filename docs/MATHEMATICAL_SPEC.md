# ERGT-Phi v1 — mathematical-to-code migration specification

Status: PROPOSED RESEARCH SPECIFICATION; not a validated trained architecture.
Scope: migrate an existing, pinned ERGT implementation. This document does not
replace the baseline's typed operators, hard measurements, or two native solvers.
The accompanying module implements and tests the NEW sparse inner problem only.

## 1. Provenance and exact baseline boundary

User-provided primary manuscript: `ergt_native_geodesic_transport_paper.pdf`,
Section 3, equations 1–27. User design report: `ERGT-Phi-Design-Report.docx`.
The variational energy, lagged scheduling, bounds, and solvers below are NEW
specifications proposed in this conversation, not equations already validated in
either manuscript. The physical article is inspiration, not a proof of this model.

The public ERGT repository inspected for interface reconciliation is
`github.com/jalaljafari2009/ERGT-paper`. Its MANIFEST reports:

- `ergt_four_seed/_locked_runtime/ergt_reviewer/physics_core.py`:
  `02943066d9b46b6ada5602398ff688cb20ec228f0c9411010ad1e9b2e6bcb3ac`
- `ergt_four_seed/_locked_runtime/ergt_reviewer/native_solver.py`:
  `0891d59c8a8bf3a1e3e1016d874a9abc2a1e4588f584f86834a3926e5642a2a7`
- `ergt_four_seed/_locked_runtime/ergt_reviewer/product_geodesic.py`:
  `53f5f814bd144fb8ecbe1894a1d0764980ce2f79f35c2d1fb05ffa1b196f0c5e`

These are hashes REPORTED BY THE PUBLIC MANIFEST, not independently verified
local downloads. Pin and verify the actual source/config/checkpoint before migration.
Do not overwrite the locked reproduction release; create a separate research tree.

Baseline modules to preserve:
- `RawTokenInputAdapter`: raw input to semantic IDs and internal identity fibre.
- `PhysicsNativeSubstrate`: seed, observables, world potentials, transport, memory.
- `ProductManifoldFieldInducer` and `IdentityFibreMetricBinder`: local typed proposals.
- `padding_independent_multiscale_support`: original label-free mesh.
- `LocalPhysicalConditionField`: original local conditions.
- `_selected_event_fields`, `product_manifold_geodesic_closure`,
  `registered_hard_measurements`, `hard_min_plus_typed_boundary_solve`:
  original native execution and decision.

The only initial governing insertion is between `_phi_gate`/`_world_potentials`
and `candidate_weight`. Keep the legacy `phi` output at [B,N,N]; expose NEW
`phase_adhesion` at [B,W,N,N]. Do not silently change existing loss tensor shapes.

Exact compatibility:
```
if phase_mode == 'disabled' or coupling == 0:
    return baseline.forward(raw_token_ids, attention_mask)
```
This must happen before new RNG use or input/state mutation. Preserving the old
formula in floating point is not the same as preserving its operation sequence.
A changed baseline checkpoint is NOT reverted simply by zeroing the coupling.

## 2. Public input, output, dimensions

Input: raw_token_ids int64 [B,N]; valid/padding mask bool [B,N]. Query, source and
candidate content are serialized INSIDE those raw tokens. No external edge list,
entity pointers, oracle horizon, relation labels, or candidate path is an input.
N <= registered context capacity. Empty/invalid examples follow the registered
input validation policy. Identity collision audit and batch/padding invariance remain.

Output: baseline answer IDs {0=A,1=B,2=abstain}, native validity, action, payload,
deficit and failure fields; add extension diagnostics without reinterpreting native
confidence as a probability of truth. Numerical/extension-contract failures cannot
force a candidate answer. Default experimental handling: abort training step; abstain
with a separate extension error at inference. Explicit baseline fallback is a distinct,
logged, cost-accounted policy and reruns from RAW input, never a partially mutated state.

Indices: b sample, i/j token, w world, u predicted event slot, r relation type,
s discrete register, t outer field step, k inner solve step, h closure depth.
The three clocks t,k,h MUST remain distinct.

Core shapes (batch index omitted in formulas):
Psi, Pi: [B,N,d]; identity: [B,N,d_id]; theta/anchor: [B,W,N];
base adhesion: [B,N,N]; corrected adhesion/edge/action: [B,W,N,N];
phase sparse patch: edge_index [2,E_phi] PER EXAMPLE AND WORLD;
phase base, weight, logit: [E_phi]; relation mixture: [E_phi,R]; offsets: [W,R].
Do not materialize [B,W,N,N,R] merely to implement the phase equations.

## 3. Baseline field and observables

Use the original input adapter and initial field:
Psi0_i = f_psi(x_i + position_i + gamma f_id(identity_i)).
Pi0=0; bulk_deficit0=0; edge_memory initially None; conserved typed charge comes
from Psi0 and is not overwritten by the evolving geometry.

Native observables C,R,stability,G,O,B and eight potentials Omega_w are unchanged.
Base b_ij = exp(mean_k(log(z_k))) with the actual BASELINE epsilon convention.
Do not replace the implementation's clamp-before-log with a paper-level `+epsilon`.

The actual reference initializes edge memory to the FIRST candidate weight:
E_new = a*Omega when E_old is None;
E_new = beta E_old + (1-beta) a*Omega otherwise.
The first step is not `(1-beta)*a*Omega` from a zero state.

Native nonnegative lengths, finite-speed/top-k masks, action, identity hold edges,
row-normalized transport, value maps, momentum, field LayerNorm and bulk decoder
remain unchanged. Identity hold in field aggregation is NOT an executable relation.
The gather convention M_i=sum_j T_ij V(Psi_j) is not transposed during migration.

## 4. Native proposal probe and causal scheduling

Define `ProbeNative(snapshot)` as the existing local field inducer, label-free
support assembly and identity binder, STOPPING BEFORE candidate answer solving.
It returns selected event indices, endpoint logits and full relation logits.
Run it after a completed outer step. This is a NEW scheduling change, not a new
supervised inference channel. Preserve the conserved initial typed-charge input.

For each predicted slot u:
Psrc_ui = masked_softmax(source_logits_ui)
Ptgt_uj = masked_softmax(target_logits_uj)
Prel_ur = softmax(full_relation_logits_u)[r], r excludes the null class.
Thus sum_r Prel_ur is event-presence mass, not necessarily 1. Do NOT multiply
that presence mass again.
Q_ijr = sum_u valid_slot_u * Psrc_ui * Prel_ur * Ptgt_uj.

Scheduling: Q^0=0. At the end of outer step t, form Q^(t+1) from that completed
snapshot. The inner solve at step t uses ONLY stop_gradient(Q^t), never Q^(t+1).
Consequently the first outer step is native; later steps acquire phase feedback.
The last fresh proposal output is reused by the final native execution.
Final native proposals are NOT detached from their own native training losses.
The phase branch's consumption of them is detached in v1.

This removes the circular definition `geometry -> endpoints -> phase -> geometry`
within a single solve. Extra probe cost must be measured.

## 5. Phase anchors and offsets

anchor_iw = pi * tanh([W2 SiLU(W1 LN(Psi0_i))]_w).
Use per-token LN, not batch normalization. Initialize new weights using a separate,
recorded RNG so the baseline RNG stream is unchanged. Anchor states are fixed
inside each solve and attached to the same raw input across outer steps.

Initial offset code: beta_wr = 2*pi*r/R (r zero-indexed within the non-null alphabet),
shared across worlds initially. This is a representation code, NOT an operator
algebra. Calibrate anchors/offsets using TRAINING-ONLY structured type losses;
freeze offsets before activating geometry feedback. If offsets are trainable later,
constrain their displacement from a calibrated code, and test against collapse.
Scalar phases never replace `applyOperator(s,r)` for noncommutative programs.

Allowed local phase chart: |theta_iw-anchor_iw| <= phase_radius.
Do not wrap/reduce angles modulo 2*pi within iterations; this introduces chart
jumps. All interaction observables use sine/cosine and remain periodic.

## 6. Sparse patch, relation mixtures and weights

Compute a PREVIEW edge state using current Psi and OLD edge memory but native b.
Select phase candidate pairs by the native top-k on that preview; exclude padding,
self pairs, invalid domains, and pairs with Q mass <= epsilon_Q. No gold edges.
An edge with b outside [epsilon_a,1-epsilon_a] is left unchanged, NOT clamped into
a new baseline. Outside the patch corrected adhesion equals the original b exactly.

For a patch edge e=(i,j,w):
q_e = sum_r stop_gradient(Q_ijr)
pi_er = stop_gradient(Q_ijr/q_e)
raw_weight_e = q_e/(1+q_e)
d_i = sum_{e incident to i} raw_weight_e (both source and target occurrences)
weight_e = raw_weight_e / max(1,max_i d_i).
Normalize independently per example/world. Thus max total incident weight <=1.
If there are no supported patch edges, return theta=anchor and a=b exactly.
Relations with zero mass never receive arbitrary positive support from normalization.

The structural and semantic weight inputs remain fixed throughout all k iterations.

## 7. Energy, response and constraints (NEW)

Let delta_er=theta_j-theta_i-beta_wr;
kappa_er=(1+cos(delta_er))/2;
xi_er=a_e*kappa_er.

h(x)=x^n/(x^n+c^n), n integer >=2, c>0.
G(x)=x*(1+nu*h(x)).
chi(x)=nu*n*x^n*c^n/(x^n+c^n)^2.
G'(x)=1+nu*h(x)+chi(x).
G''(x)=nu*n*c^n*x^(n-1)*((n+1)*c^n-(n-1)*x^n)/(x^n+c^n)^3.
These expressions are finite at x=0; do not implement x^-1 shortcuts.

R(theta,a)=sum_e weight_e*[sum_r pi_er*G(a_e*kappa_er)-c0*a_e].
DB(a||b)=a*log(a/b)+(1-a)*log((1-a)/(1-b)).
F(theta,a)=mu/2*||theta-anchor||^2 + tau_a*sum_e DB(a_e||b_e) - lambda*R.

This is an information-inspired COMPUTATIONAL energy, not a physical energy,
probability of truth, or proof of information/momentum conservation.

Let v_e=logit(a_e), v0_e=logit(b_e). The feasible set is:
|theta-anchor| <= rho_theta,
vlo=max(logit(epsilon_a),v0-rho_a),
vhi=min(logit(1-epsilon_a),v0+rho_a).
These v bounds correspond to a convex box for a. They keep 0<a<1 and imply
|log(a)-log(b)|<=rho_a. Conditional on the SAME old E and Omega, the resulting
edge-length shift versus preview is at most rho_a (with the native lower clamp).
This is a LOCAL bound, not a bound on the whole recurrent model versus the old model.

## 8. Explicit gradients (NEW)

Let D be the directed incidence matrix: -1 at source, +1 at target.
No dense D is needed: use gather and two scatter/index_add operations.

u_e = -weight_e*a_e/2 * sum_r pi_er*G'(xi_er)*sin(delta_er)
g_theta = grad_theta R = D^T u
g_a_e = grad_a R = weight_e*[sum_r pi_er*kappa_er*G'(xi_er)-c0].

Consequently:
grad_theta F = mu*(theta-anchor)-lambda*g_theta,
grad_a F = tau_a*(logit(a)-logit(b))-lambda*g_a.
All edge contributions to both endpoints must be included; updating destinations
alone would not differentiate the stated energy.

## 9. Primary closed-form solver (NEW)

Freeze context; initialize v=v0. Initialize theta to the projected previous phase,
or to anchor for a cold start. Use fixed K during training.

Each simultaneous (Jacobi) step:
a = sigmoid(v)
(g_theta,g_a) = gradients of R at CURRENT (theta,a)
theta_next = clip((theta+eta*mu*anchor+eta*lambda*g_theta)/(1+eta*mu),
                  anchor-rho_theta,anchor+rho_theta)
v_next = clip((v+eta*tau_a*v0+eta*lambda*g_a)/(1+eta*tau_a),vlo,vhi).
Commit both next states together; return a=sigmoid(v) after exactly K iterations.

This is an angle-proximal/Bernoulli-mirror update, not ordinary Euclidean gradient
on logits and not the Euclidean proximal map for DB. Its fixed-point KKT conditions
are exactly those of F over the stated box. No scalar root solve is needed here.
The separate audit solver in phase_core.py uses an actual Euclidean prox/root solve
to check agreement at convergence; it is NOT the proposed production loop.

A diagnostic extra step may compute the residual without committing that step to
forward output. If used for adaptive stopping later, define the clock explicitly.

## 10. Computable stability certificate (NEW derivation)

All bounds concern a FIXED context and fixed patch/weights/pi. Set
M1=1+nu*(1+n/4),
M2=nu*n*(n+1)/c,
dstar=max(max_i sum_{e incident i} weight_e, max_e weight_e),
CR=dstar*(1.5*M2+2*M1).
Then ||Hessian R||_2<=CR. Explanation:
||grad(a*kappa)||^2<=3/2; ||Hessian(a*kappa)||<=2; mixture weights sum to one;
summing local bounds gives the incident-weight factor dstar.

F0's curvature is >=m0=min(mu,4*tau_a). Thus lambda*CR<m0 gives a strongly
convex inner problem and a unique constrained optimum.
For the explicit mirror iteration use the STRONGER sufficient condition:
mB=min(mu,tau_a), (5/4)*lambda*CR <= sigma*mB, 0<sigma<1.
Choose eta=1/mB. In norm ||(dtheta,dv)||block=max(||dtheta||2,||dv||2):
q=(1+eta*(5/4)*lambda*CR)/(1+eta*mB) <=(1+sigma)/2<1.
The 1/4 term comes from sigmoid's global Lipschitz constant.
The two block rows each satisfy the corresponding contraction inequality; clipping
cannot increase these differences. Therefore the explicit map contracts and converges
to the unique minimizer of F. Parameter settings violating the bound are rejected,
not silently reinterpreted as stable.

For an exact-map residual rK=||T(yK)-yK||block:
||yK-y*||block<=rK/(1-q).
With numerical map error eK, use (rK+eK)/(1-q). Ordinary float arithmetic alone is
not a verified interval certificate. The module reports an analytic bound ignoring
roundoff, expressly labeled as such.

No claim of optimizer convergence or contraction of the full field/geometry feedback
loop follows from this inner theorem.

## 11. Rejoin the native field path

Scatter a only into eligible sparse patch cells. On all other pairs use literal b.
Form E with the original first-step/EMA convention, then call the native geometry,
admissibility, action, field transport, momentum, LayerNorm and bulk-deficit update.
Do NOT recompute top-k, pi, anchors, weights or native evidence during k iterations.
Top-k can be recomputed at the next OUTER step.

During early guarded migration, compare all relevant native masks with the preview.
If the experiment promises frozen masks, changed top-k/cone validity is a failed
contract, not a license to keep an edge that now violates the cone. Reject that
migration attempt or rerun a complete logged fallback. Later dynamic-mask mode is
a separately qualified schedule stage.

## 12. Final native execution and boundary

Use the final snapshot and freshly produced native proposals; phase states do not
feed an answer head. Keep both layers of native computation:
(1) token-world product closure, with native cross-world transitions, mesh, costs,
    feasibility and max-min support;
(2) typed event/register execution, including original measured local conditions,
    action-cell quantization, terminal mass and boundary deficit.
Raw world S is not silently substituted for the registered final event action.

Paper-level recurrences, retaining originals and their implementation details:
D_next(j,w')=min(D(j,w'), min_(i,w legal) D(i,w)+cost((i,w),(j,w'))).
Widest support next=max(old, max_legal min(origin_support,edge_capacity)).
s'=applyOperator(s,r_event).
Typed payload next=max(old, max_event payload_origin*transmission*effective_terminal).
Boundary deficit next=min(old, min_event deficit_origin+event_deficit).
These support channels have DIFFERENT semantics; do not multiply every probabilistic
proposal down the chain or replace a max-min channel with the payload product.

Read the required native terminal register per candidate:
valid_c = finite(A_c) and mass_c>=mass_min and deficit_c<=deficit_max,
plus inherited role/cone/overflow checks where the baseline actually implements them.
No valid candidate -> abstain. Exactly one valid -> choose it. Two valid and action
margin below the registered threshold -> abstain; otherwise choose lower action.
Do not evaluate inf-inf when both are unreachable.

Important inherited limitation: independent minima/maxima in a generic multipath
instance do not, by themselves, certify that ONE witness attains all extrema.
Keep baseline compatibility in v1; add adversarial multipath audit before promotion.
If a single-witness guarantee is required, introduce a SEPARATE versioned solver
upgrade retaining nondominated path labels (action,-log mass,deficit,register,history).
Never claim that merely preserving the terminal formula proves a new global guarantee.

## 13. Whole forward pseudocode

```
forward(raw, mask, config):
  if disabled or lambda==0: return original_native_forward(raw,mask)
  state = refactored_native_seed(raw,mask)  # parity-tested, same params/dtypes/order
  anchor = anchor_network(state.Psi0)
  theta_prev = anchor
  Q_prev = empty
  for t in range(native_field_steps):
      obs, b, Omega = native_observe(state)
      preview = native_edge_update(state.E,b[:,None]*Omega)
      patch = select_phase_patch(preview, b, Q_prev, mask)
      a = broadcast_base_without_changing_legacy_phi(b)
      for each example and world (independent blocks):
          pi,omega = detached_lagged_relation_context(Q_prev,patch)
          result = solve_phase_mirror(anchor, gather(b), edges, pi, beta, omega,
                                      config, theta_prev)
          scatter_only_patch(a,result.adhesion)
          theta_prev = result.theta
      E_new = native_edge_update(state.E,a*Omega)
      geometry = native_geometry(E_new,obs,state)
      enforce_stage_specific_mask_policy(preview,geometry)
      state = native_field_and_bulk_step(state,geometry)
      proposal = ProbeNative(state)
      Q_prev = stop_gradient(build_Q(proposal))
  return NativeFinish(state,proposal), extension_diagnostics
```

The phase loop is conceptual block batching; vectorization must preserve per-example,
per-world normalization and must not introduce any cross-example interaction.
The original native inference must not perform extra probes in disabled mode.

## 14. Differentiation and learning

Differentiate the K explicit mirror steps by normal unrolling. No differentiation
through argmin, boolean indices, certification accept/reject or newly sampled topology.
For native hard operators, retain the native training surrogate/straight-through
policy; do not invent an unregistered gradient. Teacher-free inference remains hard.
Proposals and degree-normalized weights are detached only on entry to the phase solve.
Anchors and allowed continuous base variables can receive gradients in qualified stages.

Do not train anchors merely to minimize F itself: that can manufacture agreement.
For gold relation pairs USED ONLY IN THE LOSS, define
kappa_bar_ijr=mean_w(kappa_wijr), logits_rel=kappa_bar/T_rel,
L_phase=CE(logits_rel,gold_relation) over valid supervised events.
Gold endpoints are indexing for the training loss only, NEVER an input edge set.
If the phase parametrization cannot fit a task's algebra, retain that limitation;
do not rewrite the typed register to make the auxiliary loss small.

Total loss = original_native_loss + alpha L_phase + beta L_retain
             + gamma L_parameter_constraints.
L_retain compares selected validated native continuous states with a frozen baseline
computed from the same raw training input; it must not freeze every baseline error.
Use the native loss exact weights/schedule and checkpoint qualification unless a
separate experiment explicitly changes them. Final held-out horizons never tune this.

In shadow stage the answer gradient cannot train a disconnected phase branch;
L_phase provides the specified signal. During coupled stages native losses also train
it through corrected geometry. Inner gradients are explicit formulas, so no nested
`autograd.grad` is needed in the primary solver.

## 15. Migration state machine

M0: pin source, config, data, optimizer, RNG and baseline checkpoints; exact refactor parity.
M1: isolated kernel tests, analytic derivatives, domain/zero-limit/stability tests.
M2: shadow-only phase calibration; base frozen; answer unchanged; record phase collapse.
M3: geometry-only snapshot experiment, weak lambda, nu=0, frozen masks/field; no outer feedback.
M4: full causally lagged forward loop; initially freeze base parameters, then unlock ONE group.
M5: enable nu incrementally; recalculate CR and coupling ceiling; no simultaneous n increase.
M6: optional coherent transport subspace, with its own zero-effect gate and separate tests.
M7: optional computational reduction, individually qualified and error/cost accounted.
M8: freeze; independent zero-teacher windows and unopened horizon/multipath/failure panels.

Increment a scalar only when preregistered readiness windows pass. On failure restore
model + optimizer + scheduler + RNG + data cursor, not just parameters. Keep phase_mode,
mask_mode, feedback_mode, lambda, nu, K and active parameter groups in the run manifest.

Example research defaults (not empirically optimal): mu=1, tau_a=.25, n=2, c=.5,
nu=0 initially, lambda=0 when disabled and .02 in first isolated active test,
rho_theta=.35 rad, rho_a=.10 log-odds, epsilon_a=1e-4, K=24, sigma=.5.
Native field depth, worlds, sparse support, horizon and thresholds are imported from
the actual locked run, not inferred from class defaults.

## 16. Optional extensions — not part of v1 activation

Coherent subspace: z_i^(k+1)=(1-rho)b_i+rho sum_j T_ij U_ij z_j^k, rho<1,
||U_ij||<=1 on the fixed native mask. U can be a convex mixture of relation-specific
orthogonal block rotations. Fixed T gives an infinity-of-node-2-norm contraction.
Inject only via M_eff=M+gamma_z Pz, with gamma_z=0 reproducing the old field update.
Scalar phases still do not implement noncommutative typed operators.

Efficiency: inner residual stopping uses a derived bound, not an answer probability.
Decision stability additionally requires Lipschitz/error bounds through action and
NO mask/register threshold changes. On a frozen graph, edge action error <=epsilon
and maximum H steps give path error <=H*epsilon; winner margin >2H*epsilon is a
sufficient ranking condition, not a complete terminal-validity proof.
World compression is exact only if the necessary operators/factors are losslessly
shared; approximate compression needs measured reconstruction and decision error.
Do not select a rank, truncation or stopping confidence merely to report a speedup.

## 17. Required tests and reporting

- lambda zero: original forward AND training-gradient parity; no new RNG effects.
- disabled module insertion: same field states, masks, event proposals and boundary outputs.
- analytic reward gradients and closed-form-unroll finite-difference checks.
- zero evidence/empty patch: exact neutrality; no division-by-zero on inactive cells.
- gauge shift: anchor and theta shifted together leaves adhesion invariant.
- edge permutation and batch/padding invariance.
- numerical Hessian observations vs analytic bound on toy fixtures (not a global proof by sampling).
- failed analytic stability guard prevents execution under the certified label.
- logit-domain trust region and nonnegative native action.
- native wrong-register, missing-evidence, low-payload, high-deficit, ambiguity and multipath audits.
- compare phase+linear, phase+nonlinear and matched-capacity non-phase controls.
- complete rollback/resume equivalence; no updates after final freeze.
- report answer coverage, unsupported acceptance, accuracy, memory, latency, probe/diagnostic/fallback cost.

The supplied test suite validates only the independent kernel. It does NOT establish
full model convergence, trained accuracy, native integration parity, scientific novelty,
or any inference efficiency advantage.
