# Module → Paper Mapping

This table maps every code artefact to the corresponding section / construct in
the proposal *"Adaptive Calibrated Differentiable Agentic Networks (ACDAN)"*
(AAAI-2027). Vietnamese section titles from the proposal are kept for traceability.

| Paper concept (proposal) | Where in the proposal | Code |
|---|---|---|
| **ACDAN** overall architecture | "Kiến trúc Đề xuất: ACDAN" | [`agent.py`](../src/acdan/agent.py) |
| Latent-space reasoning (recurrent block unroll) | "Lập luận ẩn ... recurrent block" | [`latent_reasoning.py`](../src/acdan/latent_reasoning.py) `LatentReasoner._unroll` |
| In-Place Test-Time Training (In-Place TTT) | "Huấn luyện thời gian kiểm thử tại chỗ" | `LatentReasoner._ttt_adapt` |
| Differentiable Textual Optimization (DTO) | "Tối ưu hóa Văn bản Vi phân" | [`dto.py`](../src/acdan/dto.py) `DifferentiableTextOptimizer.optimize` |
| Autoregressive Lattice DTO | multi-step extension of DTO | [`sequence_dto.py`](../src/acdan/sequence_dto.py) `AutoregressiveLatticeOptimizer` |
| First-order logit update `L ← L − η ∇J` | "Tối ưu hóa bậc một qua DTO" | `DifferentiableTextOptimizer.optimize` loop |
| Process Reward Model (TIM-PRM / Athena-PRM) | "Multimodal PRM", "TIM-PRM" | [`rewards.py`](../src/acdan/rewards.py) `ProcessRewardModel`, `MockProcessReward` |
| PRM gradient back-prop into logits | "Đạo hàm từ PRM được truyền ngược" | `MockProcessReward.grad_wrt_probs` + DTO softmax VJP |
| Monte-Carlo **Net Information Gain** (O(N)) | "Monte Carlo Net Information Gain" | `rewards.net_information_gain` |
| Two-layer **Agentic Computation Graph** | "Đồ thị Động Hai Lớp" | [`graph.py`](../src/acdan/graph.py) `AgenticComputationGraph` |
| Execution layer (EX) | "Lớp Thực thi" | `AgenticComputationGraph._build_execution_layer` |
| Dependency layer (ED) + strictness levels | "Lớp Phụ thuộc ... quan sát/khai báo/suy diễn" | `_build_dependency_layer`, `Strictness` |
| von Neumann entropy (anti state-collapse) | objective term + "entropy von Neumann" | `AgenticComputationGraph.von_neumann_entropy`, `make_entropy_hook` |
| DECS dead-step pruning (>50% token cut) | "DECS ... Cắt tỉa nhánh chết" | `AgenticComputationGraph.dead_steps` / `prune` |
| **Tool Usage Inertia** / Inertial Sensing | "Quán tính Sử dụng Công cụ", AutoTool | [`inertia.py`](../src/acdan/inertia.py) `InertialSensor` |
| Probabilistic tool-transition graph | "đồ thị chuyển trạng thái xác suất" | `InertialSensor.fit`, `transition_matrix` |
| Self-verification / **Independent Question Asking** | "Đặt Câu hỏi Độc lập" | [`verification.py`](../src/acdan/verification.py) `IndependentVerifier`, `MockIndependentVerifier` |
| **RLCM** confidence probe | "đầu dò độ tin cậy siêu nhẹ" | `ConfidenceProbe` |
| Margin-based confidence | "Biên độ Tin cậy" | `verification.margin_score`, `SelfVerifier` |
| Confidence calibration metric (ECE) | "Hiệu chuẩn Lòng tin" | `verification.expected_calibration_error` |
| Adaptive coefficients by task difficulty | "hệ số điều phối thích ứng tùy độ khó" | `agent._plan` `difficulty_scale` |
| Ablation switches per module | "Ablation-ready" | [`config.py`](../src/acdan/config.py) `AblationFlags` |
| Pluggable real backends (no core changes) | "triển khai nhanh", OpenHands/PRM | [`registry.py`](../src/acdan/registry.py) |
| Synthetic benchmark families (math/code/tool) | GAIA / GSM8K / LiveCodeBench / ToolBench groups | [`tasks/synthetic.py`](../src/acdan/tasks/synthetic.py) |
| Reproducible metrics & summary | "Đo lường ... Benchmarks" | [`metrics.py`](../src/acdan/metrics.py), [`evaluate.py`](../src/acdan/evaluate.py) |

## What is faithfully implemented vs. stubbed

**Implemented as real, runnable algorithms (numpy):**
- DTO first-order optimisation with analytic gradients (softmax VJP), verified
  against finite differences in tests.
- von Neumann entropy via eigendecomposition of a PSD density matrix.
- O(N) Net Information Gain.
- Dead-step detection on the dependency graph + pruning.
- First-order Markov inertial sensing with Laplace smoothing.
- RLCM confidence probe (2-layer MLP trained by BCE gradient descent) + ECE.
- In-place TTT (auxiliary reconstruction objective, analytic gradient).

**Mocked / pluggable (offline stand-ins behind real interfaces):**
- The **core model** (`MockCoreModel`) — replace with an LLM action head.
- The **PRM** (`MockProcessReward`) — replace with TIM-PRM / Athena-PRM.
- The **independent verifier** (`MockIndependentVerifier`) — replace with a real
  tool/sandbox evidence query.
- **Datasets** — synthetic task families stand in for GAIA/GSM8K/LiveCodeBench.

The boundary is deliberately the `registry.py` seam: swapping any mock for a real
backend requires implementing a small Protocol and registering it — **no change
to `agent.py`, `dto.py`, `graph.py`, `inertia.py`, or `verification.py`.**
