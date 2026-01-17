# A Minimal Control-Theoretic Framework for the Triadic Phase Operator ΔΦ

**Marcel Krüger¹ and Don Feeney²**

¹ *Independent Researcher, Meiningen, Germany*  
marcelkrueger092@gmail.com  
ORCID: [0009-0002-5709-9729](https://orcid.org/0009-0002-5709-9729)

² *Independent Researcher, Pennsylvania, USA*  
dfeen87@gmail.com  
ORCID: [0009-0003-1350-4160](https://orcid.org/0009-0003-1350-4160)

**January 17, 2026**

---

## Abstract

We present a minimal, control-theoretic formulation of the **triadic phase operator ΔΦ** as a deterministic diagnostic tool for stability monitoring, regime detection, and reproducible analysis in nonstationary and path-dependent systems. The framework introduces a triadic feature decomposition into state, memory, and constraint components, yielding an explicit instability vector whose norm defines an interpretable stability score. 

Importantly, **ΔΦ is formulated as an observer-style diagnostic** and does not constitute a control law or prediction mechanism. We further clarify how learning-based methods may be incorporated strictly as feature and constraint estimators without compromising auditability or Lyapunov-compatibility. 

The resulting framework is domain-agnostic, falsifiable, and suitable for safety-critical, human-in-the-loop environments where deterministic replay and conceptual transparency are required.

---

## 1. Introduction

Many complex dynamical systems exhibit path dependence, hysteresis, delayed response, and abrupt regime transitions that are not adequately captured by instantaneous state variables alone. Examples range from physiological regulation and neural dynamics to human-in-the-loop engineering systems and nonstationary industrial processes. In such settings, classical control and learning-based approaches often face difficulties related to interpretability, auditability, and formal stability certification.

This work introduces a **minimal diagnostic framework** centered on the **triadic phase operator ΔΦ**, designed to monitor coherence and stability without intervening in system dynamics. Rather than proposing a new control law or predictive model, the framework focuses on the structured extraction of three feature classes:

1. **Instantaneous state configuration**
2. **Accumulated memory**
3. **Admissible constraint margins**

Their joint evolution is encoded in a deterministic instability vector whose norm provides an interpretable stability score.

### Key Motivation: Decoupling Diagnosis from Control

The ΔΦ operator is intentionally formulated as an **observer-like object**: it evaluates how a system evolves relative to its own history and constraints, but **does not issue actions or optimize objectives**. This separation is essential for:

- ✓ Reproducibility
- ✓ Falsifiability  
- ✓ Regulatory compatibility in safety-critical contexts

While the framework is compatible with learning-based feature construction, all stability logic remains **explicit, deterministic, and replayable**. The present note therefore serves as a clean control-theoretic scaffold, independent of any specific physical ontology or application domain.

---

### Purpose

This note provides a minimal, control-theoretic interpretation of the triadic phase operator ΔΦ for stability monitoring, regime detection, and reproducible implementation. It is **intentionally not** a re-derivation of HLV spacetime physics; instead it is a structural scaffold compatible with standard nonlinear systems and control language.

---

### System Setting

Consider a (possibly nonstationary) system

```
ẋ(t) = f(x(t), u(t), t),    y(t) = h(x(t), t)                    (1)
```

with **x** ∈ ℝⁿ, input **u** ∈ ℝᵐ, and measured signal **y** ∈ ℝᵖ. The system may be path-dependent and exhibit hysteresis or delayed response.

---

### Triadic Feature Map (State / Memory / Constraint)

We define a **triadic feature state**

```
Φ(t) = {φₛ(t), φᵢ(t), φ_c(t)} ∈ ℝ³                              (2)
```

where each component is a feature extracted from **y(t)** and/or a reconstructed state:

- **φₛ(t)** *(State feature)*: instantaneous structural configuration (e.g., synchrony, curvature proxy, order parameter)
- **φᵢ(t)** *(Integral / memory feature)*: an accumulated or filtered history (e.g., EWMA / integral state)
- **φ_c(t)** *(Constraint feature)*: an admissible-regime boundary or stability envelope (e.g., safe set, feasibility margin)

> **Implementation note:** A typical choice is φᵢ(t) = ∫₀ᵗ k(τ) φₛ(τ) dτ or an EWMA filter, and φ_c(t) can be a data-driven estimate of a stability boundary (e.g., percentile envelope) or a control barrier bound.

---

### Definition of the ΔΦ Operator

Define the **triadic phase operator** as

```
         ⎡ φ̇ₛ(t)        ⎤       ⎡ φ̇ₛ(t)          ⎤
ΔΦ(t) := ⎢ eᵢ(t)         ⎥   =   ⎢ φₛ(t) − φᵢ(t)  ⎥         (3)
         ⎣ e_c(t)        ⎦       ⎣ φ_c(t) − φₛ(t) ⎦
```

**Interpretation:**
1. **φ̇ₛ(t)** — Local variation rate
2. **φₛ(t) − φᵢ(t)** — Deviation from historical coherence
3. **φ_c(t) − φₛ(t)** — Distance to constraint boundary

---

### Control-Theoretic Reading

**ΔΦ is a diagnostic operator (observer-like), not a controller.** It is structurally analogous to:

- Error-state vectors (e = x̂ − x)
- Lyapunov derivative indicators (tracking drift toward instability)
- Regime-shift detectors in hysteretic systems

---

## Summary Table

**Table 1:** Minimal semantics of the triadic features and the induced ΔΦ components

| Object | Definition (minimal) | Control meaning / examples |
|--------|---------------------|---------------------------|
| **φₛ(t)** | Instantaneous feature from y(t) or x̂(t) | Order parameter, synchrony, curvature proxy, phase estimate |
| **φᵢ(t)** | Memory feature (EWMA / integral / hysteresis state) | Captures path dependence; "where the system came from" |
| **φ_c(t)** | Constraint / admissible envelope feature | Safe-set margin, feasibility boundary, stability envelope |
| **φ̇ₛ(t)** | Time derivative / slope | Local volatility, fast reconfiguration, incipient transition |
| **eᵢ(t) = φₛ − φᵢ** | Coherence-to-history mismatch | Hysteresis gap; drift from learned baseline |
| **e_c(t) = φ_c − φₛ** | Boundary proximity margin | Approaching instability / constraint violation |
| **‖ΔΦ‖** | Norm on Eq. (3) | Global instability / regime-shift score |

---

### Minimal Stability Score

Fix weights **wₛ, wᵢ, w_c > 0** and define

```
J(t) = ‖ΔΦ(t)‖²_W = wₛ·φ̇ₛ² + wᵢ·(φₛ − φᵢ)² + w_c·(φ_c − φₛ)²      (4)
```

A system is **coherently stable** on a horizon [t, t + T] if

```
sup_{τ∈[t,t+T]} J(τ) < ε²                                        (5)
```

---

### Lyapunov-Compatible Formulation (Minimal)

Let **e(t) = (eᵢ(t), e_c(t))ᵀ**. A candidate Lyapunov function is

```
V(e) = ½ eᵀPe,    P ≻ 0                                          (6)
```

If the feature dynamics imply an inequality of the form

```
V̇(t) ≤ −α‖e(t)‖² + β|φ̇ₛ(t)|·‖e(t)‖                              (7)
```

then for sufficiently small |φ̇ₛ| (or bounded input) the error state is **ultimately bounded**, i.e., regime coherence persists; spikes in |φ̇ₛ| can transiently break the bound and flag transitions.

---

## Algorithmic Pipeline (Diagnostic Observer)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  y(t) ──────────────────────────────────────────┐               │
│  measured                                       │               │
│  signal                                         ↓               │
│                                          ┌─────────────┐        │
│                                          │ Feature map │        │
│                                          └──────┬──────┘        │
│                                                 │               │
│                                            φₛ(t)│               │
│                          ┌──────────────────────┼──────┐        │
│                          ↓                      ↓      ↓        │
│                    ┌─────────┐          ┌──────────┐  │        │
│                    │ Memory  │          │Constraint│  │        │
│                    │         │          │          │  │        │
│                    └────┬────┘          └────┬─────┘  │        │
│                         │                    │        │        │
│                    φᵢ(t)│               φ_c(t)│        │        │
│                         │                    │        │        │
│                         └─────────┬──────────┘        │        │
│                                   ↓                   │        │
│                              ┌─────────┐              │        │
│                              │  ΔΦ(t)  │←─────────────┘        │
│                              └────┬────┘                       │
│                                   │                            │
│                                   ↓                            │
│                              ┌─────────┐                       │
│                              │  J(t)   │                       │
│                              │‖ΔΦ(t)‖² │                       │
│                              └────┬────┘                       │
│                                   │                            │
│                                   ↓                            │
│                          ┌────────────────┐                    │
│                          │ Regime flag /  │                    │
│                          │     alert      │                    │
│                          └────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Figure:** Diagnostic observer pipeline showing the flow from measured signal y(t) through feature extraction, memory and constraint processing, to the ΔΦ operator and final stability score J(t).

---

## 2. AI-Assisted Diagnostics vs. Classical Control

The ΔΦ framework is intentionally positioned as a **diagnostic observer**, not as a control law. To avoid conceptual ambiguity, it is useful to contrast classical control architectures with AI-assisted ΔΦ diagnostics.

### Comparison Table

**Table 2:** Classical control architectures versus AI-assisted ΔΦ diagnostics

| Aspect | Classical Control / AI Control | AI-Assisted ΔΦ Diagnostics |
|--------|-------------------------------|---------------------------|
| **Primary role** | Actively modifies system input u(t) | Observes and evaluates system behavior only |
| **Decision authority** | Controller or policy issues actions | Human or external controller remains authoritative |
| **Use of AI** | Policy learning, optimization, prediction | Feature construction and constraint estimation only |
| **Closed-loop enforcement** | Yes (feedback acts on dynamics) | No (diagnostic layer only) |
| **Failure mode** | Unstable control, policy drift, reward hacking | False positive / false negative diagnostics |
| **Interpretability** | Often opaque (black-box policies) | Explicit stability terms φ̇ₛ, φₛ − φᵢ, φ_c − φₛ |
| **Safety guarantees** | Difficult to certify formally | Lyapunov-compatible, observer-style bounds |
| **Regulatory suitability** | Challenging in safety-critical domains | Compatible with audit, replay, and human oversight |

---

### System Architecture Diagram

**Figure 1:** Strict separation between control and AI-assisted ΔΦ diagnostics

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                         System                                   │
│                     ┌──────────┐                                 │
│                     │  x(t)    │                                 │
│                     │  y(t)    │                                 │
│                     └────┬─────┘                                 │
│                          │                                       │
│                          │ observes                              │
│            ┌─────────────┴──────────────┐                        │
│            │                            │                        │
│            ↓                            ↓                        │
│    ┌──────────────┐              ┌────────────┐                 │
│    │ Controller / │              │  φₛ, φᵢ,   │                 │
│    │     AI       │              │   φ_c      │                 │
│    └──────┬───────┘              └──────┬─────┘                 │
│           │                             │                        │
│     u(t)  │ acts                        ↓                        │
│           │                        ┌─────────┐                  │
│           │                        │   ΔΦ    │                  │
│           │                        └────┬────┘                  │
│           │                             │                        │
│           │                             ↓                        │
│           │                        ┌─────────┐                  │
│           │                        │  J(t)   │                  │
│           │                        └────┬────┘                  │
│           │                             │                        │
│           │                             │ informs                │
│           │                             ↓                        │
│           │                      ┌─────────────┐                │
│           └──────────────────────│   Human     │                │
│                                  │  decision   │                │
│                                  └─────────────┘                │
│                                                                  │
│                    ╔════════════════════════════╗                │
│                    ║  NO FEEDBACK LOOP FROM     ║                │
│                    ║  DIAGNOSTICS TO CONTROL    ║                │
│                    ╚════════════════════════════╝                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

The diagnostic pipeline **observes** system behavior and evaluates stability **without modifying** system inputs. Human decision-makers remain in the loop, with the ΔΦ diagnostic providing information but not issuing control commands.

---

## Editor's Note: Why This Is Not AI Control

Although learning-based tools may be used to construct φₛ(t), φᵢ(t), or φ_c(t), the ΔΦ framework **does not implement** control, optimization, or prediction.

### No AI component:

- ✗ Issues commands to the system
- ✗ Modifies the control input u(t)
- ✗ Enforces constraints autonomously
- ✗ Replaces human or supervisory decision-making

All AI usage is strictly confined to **representation and estimation**. The stability score

```
J(t) = ‖ΔΦ(t)‖²
```

remains **deterministic, reproducible, and explicitly specified**. As a result, any claimed performance arises from the disclosed pipeline, not from adaptive or opaque decision logic.

> **This separation is deliberate and essential** for falsifiability, auditability, and applicability in safety-critical environments.

---

## Where the Framework Intentionally Ends

This note **does not claim**:

- An ontological statement about spacetime or fundamental physics
- That φₛ, φᵢ, φ_c are unique or "correct" choices
- That ΔΦ enforces control; it only diagnoses structure

The framework requires **explicit disclosure** of the chosen feature map, memory filter, and constraint model. This is deliberate: **reproducibility and falsifiability** are properties of the implemented pipeline, not of rhetoric.

---

### Optional Context (HLV link, clearly non-required)

In HLV, a triadic time structure ψ(t) = t + iφ(t) + jχ(t) is used as an operator-level organizer of coherence and memory channels; the present note uses a purely control-theoretic triad without requiring the operator ontology. (The separation is intentional.) [1]

---

## 3. Discussion

The primary contribution of the ΔΦ framework lies in its **structural clarity** rather than in algorithmic novelty. By reducing stability assessment to an explicit triadic feature mismatch, the framework provides a common language for analyzing regime shifts across heterogeneous systems.

### Separation of Diagnosis and Control

A central design choice is the **strict separation between diagnostic observation and control**. Unlike adaptive or reinforcement-learning-based controllers, ΔΦ does not:

- Modify system inputs
- Enforce constraints
- Optimize performance metrics

All learning-based components, when used, are confined to representation and estimation layers. This avoids common failure modes such as policy drift, reward hacking, or opaque decision boundaries.

### Control-Theoretic Perspective

From a control-theoretic perspective, the formulation admits **Lyapunov-compatible interpretations**. The stability score can be viewed as a weighted quadratic form over an explicitly defined error state, allowing boundedness and persistence arguments without requiring model inversion or full state observability.

At the same time, the framework remains **agnostic** to the specific choice of features, memory filters, or constraint estimators, which must be disclosed explicitly in any concrete implementation.

### Limitations

Limitations are equally important. The framework:

- Does **not** guarantee causal explanations for regime transitions
- Does **not** prescribe optimal responses
- Validity depends on the quality of the chosen feature maps
- Requires sufficient observability of the underlying system

> **These boundaries are intentional** and serve to preserve falsifiability and methodological honesty.

---

## 4. Conclusion

We have presented a minimal, control-theoretic formulation of the **triadic phase operator ΔΦ** as a deterministic diagnostic framework for monitoring stability, memory effects, and constraint proximity in complex dynamical systems. The approach is **explicitly non-interventional** and separates diagnosis from control, prediction, and optimization.

By expressing instability as a transparent function of state variation, historical deviation, and constraint margins, the framework supports:

- ✓ Auditability
- ✓ Deterministic replay
- ✓ Lyapunov-compatible reasoning

Learning-based methods may be incorporated without compromising these properties, provided they are restricted to feature and constraint estimation.

### Final Statement

**The ΔΦ framework is not a theory of dynamics, intelligence, or spacetime.** It is a falsifiable diagnostic scaffold whose utility depends entirely on explicit implementation choices. This intentional minimalism is its strength: it enables cross-domain application while preserving conceptual and regulatory clarity.

---

## 5. Reference Implementation and Reproducible Pipeline

To support reproducibility, auditability, and independent verification, a reference implementation of the diagnostic pipeline described in Section 2 is provided as an open, modular repository:

### **[AI-Stabilized-Spacetime-Dynamics](https://github.com/dfeen87/AI-Stabilized-Spacetime-Dynamics)**

The purpose of this implementation is not deployment or optimization, but to demonstrate how the theoretical ΔΦ framework can be instantiated as a fully deterministic, non-interventional diagnostic observer.

---

### Scope of the Implementation

The repository provides a modular realization of the pipeline

```
y(t) → (φₛ, φᵢ, φ_c) → ΔΦ(t) → J(t)
```

with explicit separation between feature construction, diagnostic logic, and external decision-making.

Specifically, the implementation includes:

- ✓ A **feature-construction layer** for φₛ(t), φᵢ(t), and φ_c(t), allowing multiple concrete realizations to be evaluated under identical diagnostic logic
- ✓ An **explicit implementation** of the ΔΦ operator and the stability score J(t) exactly as defined in Eq. (3) and Eq. (4)
- ✓ **Deterministic execution paths** enabling replay, parameter locking, and auditability for validation studies
- ✓ **Unit tests and reproducibility checks** ensuring consistency of the diagnostic output under repeated execution

---

### Non-Interventional Design

No component in the repository:

- ✗ Issues control actions
- ✗ Modifies system inputs u(t)
- ✗ Performs closed-loop enforcement
- ✗ Optimizes objectives or predictions

Any learning-based modules, when present, are strictly confined to representation or constraint estimation and do not alter the diagnostic structure of ΔΦ or the computation of J(t).

---

### Purpose and Limitations

The repository is **not intended** as a controller, predictive engine, or production system. Instead, it serves as:

1. A reproducible companion to the theoretical framework
2. A concrete reference for reviewers and readers
3. A testbed for falsification, comparison, and extension

All conclusions drawn from the ΔΦ framework remain **contingent on the explicitly disclosed choices** of feature maps, memory filters, and constraint estimators in a given experiment. The reference implementation enforces this disclosure by design, reinforcing the framework's emphasis on transparency and methodological honesty.

---

## Appendix A: Conceptual Origin and Lineage of the Triadic Operator

The triadic phase operator ΔΦ employed in this work has its conceptual origin in earlier theoretical developments within the **Helix–Light–Vortex (HLV) framework**, where triadic decompositions were introduced as an organizing principle for coherence, memory, and admissible structure in nonstationary systems.

In the original HLV context, the triadic structure was motivated by geometric and information-theoretic considerations related to spiral-time dynamics. The present work, however, **does not rely on these physical or ontological assumptions**. Instead, it extracts the purely mathematical core of the triadic construction and reformulates it in standard control-theoretic and diagnostic terms.

**This appendix is included solely to document priority and conceptual lineage.** All definitions, results, and claims in the main text are fully independent of the HLV framework and remain valid even if the HLV interpretation is set aside entirely. The triadic operator is treated here as a general diagnostic structure applicable to a broad class of nonstationary, path-dependent systems.

The conceptual origin of the triadic operator can be traced to earlier work within the Helix–Light–Vortex framework [1].

---

## Appendix B: External EEG-Based Demonstration of the ΔΦ Diagnostic Operator

To illustrate the practical applicability of the **ΔΦ diagnostic operator** to real-world neurophysiological data, we reference an independent, publicly available EEG analysis pipeline implemented as a Kaggle notebook and mirrored on GitHub.

### Scope and Purpose

The referenced notebook applies a **triadic deviation framework** to multichannel EEG recordings from epilepsy patients and healthy controls. Its purpose is demonstrative and methodological: to show that the ΔΦ structure can be operationalized on empirical data to detect regime transitions, instability, and collapse-like dynamics.

> **Important**: No claim of theoretical novelty is made within the notebook itself. The present manuscript remains the primary and authoritative source for the formal definition, interpretation, and control-theoretic framing of the ΔΦ operator.

---

### Data and Feature Construction

The EEG analysis uses publicly available datasets (**CHB–MIT Scalp EEG Database** via PhysioNet), processed into windowed features corresponding to three conceptual axes:

- **Structural (S)**: proxy measures related to signal topology, variability, and local organization
- **Informational (I)**: entropy-based and complexity-related measures capturing historical deviation
- **Coherence (C)**: synchronization and correlation metrics reflecting admissible collective structure

Deviations from baseline are computed as **ΔS**, **ΔI**, and **ΔC**, which are combined into a scalar instability index:

```
ΔΦ = α|ΔS| + β|ΔI| + γ|ΔC|,    where α + β + γ = 1
```

This implementation is consistent with the abstract diagnostic definition used throughout this manuscript, while acknowledging that specific feature choices, normalizations, and weights are implementation-dependent.

---

### Qualitative Results

The notebook demonstrates that:

✓ **EEG data from healthy subjects** cluster in low-ΔΦ (isostatic) regimes

✓ **Epileptic EEG recordings** exhibit elevated ΔΦ values, including transitions into allostatic, high-allostatic, and collapse-like regimes

✓ **Peaks in ΔΦ** align temporally with seizure-related activity

These observations support the interpretation of **ΔΦ as a regime-instability diagnostic** rather than as a predictive or control mechanism.

---

### Reproducibility and Attribution

The implementation, data processing steps, and visualization code are fully documented and publicly accessible at:

### 🔗 **[EEG Analysis Notebook](https://github.com/nwycomp/NeuroDynamics-Collapse-Validation-/blob/main/eeg-part-four.ipynb)**

```
https://github.com/nwycomp/NeuroDynamics-Collapse-Validation-/blob/main/eeg-part-four.ipynb
```

The notebook serves as an **external demonstration example** and does **not** define the theoretical framework itself. All formal definitions, scope limitations, and claims remain governed by the present manuscript.

---

## References

**[1]** M. Krüger. "A Mathematical Unification of the Helix–Light–Vortex (HLV) Framework: Discrete Geometry, Spiral Time, Unified Lagrangians, and an Effective Field Theory Approach to Geometric Unification," Zenodo (2026). doi:10.5281/zenodo.18261685.

---

## Quick Reference

### Mathematical Symbols

| Symbol | Definition |
|--------|-----------|
| **x** ∈ ℝⁿ | State vector |
| **u** ∈ ℝᵐ | Control input |
| **y** ∈ ℝᵖ | Measured signal |
| **Φ(t)** | Triadic feature state |
| **φₛ(t)** | State feature (instantaneous) |
| **φᵢ(t)** | Integral/memory feature |
| **φ_c(t)** | Constraint feature |
| **ΔΦ(t)** | Triadic phase operator |
| **eᵢ(t)** | Memory error: φₛ − φᵢ |
| **e_c(t)** | Constraint margin: φ_c − φₛ |
| **J(t)** | Stability score: ‖ΔΦ(t)‖²_W |
| **V(e)** | Lyapunov candidate function |
| **wₛ, wᵢ, w_c** | Feature weights |

### Key Equations

```
ΔΦ(t) = [φ̇ₛ(t), φₛ(t)−φᵢ(t), φ_c(t)−φₛ(t)]ᵀ                    (3)

J(t) = wₛ·φ̇ₛ² + wᵢ·(φₛ−φᵢ)² + w_c·(φ_c−φₛ)²                    (4)

V̇(t) ≤ −α‖e(t)‖² + β|φ̇ₛ(t)|·‖e(t)‖                            (7)
```

---

*A control-theoretic diagnostic framework for stability monitoring in complex, nonstationary systems.*
