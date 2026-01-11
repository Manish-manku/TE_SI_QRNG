# Trust-Enhanced Quantum Randomness (TE-SI-QRNG)

## A Self-Testing Source-Independent Approach to Random Number Generation

This repository contains the complete implementation for **Trust-Enhanced Source-Independent Quantum Random Number Generator (TE-SI-QRNG)**, a novel approach that bridges practical quantum random number generation with measurable trust guarantees.

---

## Overview

**TE-SI-QRNG** replaces static trust assumptions in quantum random number generation with dynamic, self-tested trust metrics. This enables:

- **Measurable Trust**: Quantified through trust vector T = {ε_bias, ε_drift, ε_corr, ε_leak}
- **Adaptive Certification**: Entropy bounds adjust in real-time based on measured trust
- **Practical Deployment**: Comparable complexity to standard SI-QRNG
- **Attack Resilience**: Detects bias, drift, correlation, and adversarial manipulation

---

## Key Innovation

Unlike Device-Independent QRNG (requires impractical Bell tests) and standard Source-Independent QRNG (assumes unconditional trust), TE-SI-QRNG:

1. Continuously monitors source and measurement quality
2. Quantifies trust using operational self-tests (no Bell inequalities required)
3. Adjusts entropy certification: H_min^trusted(X|E) = H_min(X|E) - f(T)
4. Adapts extraction rate to maintain security guarantees

---

## Repository Structure

```
Trust Enhanced QRNG/
├── te_si_qrng.py          # Core TE-SI-QRNG implementation
├── simulator.py            # Quantum source simulator
├── experiment.py           # Comprehensive experimental validation
├── README.md              # This file
└── results/               # Experimental results (generated)
    ├── figures/           # Generated plots
    └── data/              # Experimental data (JSON)
```

---

## Installation

### Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

### Setup

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Clone/download this repository
cd "Trust Enhanced QRNG"
```

---

## Usage

### Quick Start

```python
from te_si_qrng import TrustEnhancedQRNG
from simulator import QuantumSourceSimulator, SourceParameters, SourceType

# Create quantum source
params = SourceParameters(source_type=SourceType.IDEAL)
source = QuantumSourceSimulator(params, seed=42)

# Create TE-SI-QRNG
qrng = TrustEnhancedQRNG(block_size=10000) #block_size can be alter for fast excuetion.

# Generate certified random bits
random_bits, metadata = qrng.generate_certified_random_bits(
    n_bits=50000, #n_bits can alter for fast excuetion.
    source_simulator=source
)

print(f"Generated {len(random_bits)} certified random bits")
print(f"Trust score: {metadata[0]['trust_score']:.4f}")
```

### Running Experiments

```bash
# Run all experiments
python experiment.py
```

This generates:
- 5 comprehensive experiments validating TE-SI-QRNG
- Figures saved to `results/figures/`
- Data saved to `results/data/`


---

## Core Components

### 1. Trust Vector

Quantifies device reliability:

```python
@dataclass
class TrustVector:
    epsilon_bias: float    # Deviation from uniformity
    epsilon_drift: float   # Temporal instability
    epsilon_corr: float    # Memory/correlation effects
    epsilon_leak: float    # Side-channel leakage
```

### 2. Self-Testing Framework

**Statistical Tests**:
- Santha-Vazirani violation test
- Runs test for independence
- Autocorrelation analysis
- Frequency monobit test

**Quantum Witness Tests**:
- Dimension witness
- Energy constraint test
- POVM consistency check

**Physical Drift Monitoring**:
- Detector efficiency tracking
- Dark count rate analysis

### 3. Entropy Certification

Trust-adjusted min-entropy:

```
H_min^trusted(X|E) = H_min(X|E) - f(T)
```

where f(T) is the trust penalty that increases as trust degrades.

### 4. Randomness Extraction

**Toeplitz Hashing**:
- Quantum-proof extractor
- Adaptive extraction rate based on trust
- Efficient GF(2) matrix multiplication

---

## Experimental Validation

### Experiment 1: Trust Quantification
Validates that trust metrics correctly distinguish source types.

### Experiment 2: Entropy Certification
Compares certified min-entropy with empirical entropy.

### Experiment 3: Attack Detection
Tests response to adversarial manipulation.

### Experiment 4: Comparison with SI-QRNG
Compares TE-SI-QRNG against standard Source-Independent QRNG.

### Experiment 5: Temporal Adaptation
Tests adaptation to time-varying source quality.

---

## Implementation Details

### Trust Score Computation

```python
def trust_score(self) -> float:
    return 1.0 - np.sqrt(
        self.epsilon_bias**2 +
        self.epsilon_drift**2 +
        self.epsilon_corr**2 +
        self.epsilon_leak**2
    ) / 2.0
```

### Trust Penalty

```python
def trust_penalty(self) -> float:
    return -np.log2(max(self.trust_score(), 0.01))
```

### Adaptive Extraction

```python
extraction_rate = trust_score * efficiency * h_min_trusted
output_length = int(input_length * extraction_rate)
```

---

## Threat Model

### Adversarial Capabilities
- Full control of quantum source
- Classical side-channel access
- Temporal adaptation
- Knowledge of self-testing algorithms

### Trust Assumptions
- Semi-trusted measurement devices (monitored)
- Trusted classical post-processing
- Bounded adversary (cannot arbitrarily replace detectors)

### Security Goal
Ensure output satisfies:
```
H_min^ε(X^n | E) ≥ ℓ - leak_EC
```
even under adversarial conditions.

---

## Research Article

The complete IEEE-format research article (`paper.tex`) includes:

1. **Introduction**: Problem motivation and contribution
2. **Related Work**: Positioning vs. DI-QRNG, SI-QRNG, Semi-DI-QRNG
3. **Threat Model**: Formal adversary model and assumptions
4. **Architecture**: System design and components
5. **Self-Testing Framework**: Mathematical formalization
6. **Entropy Certification**: Trust-adjusted bounds and proofs
7. **Randomness Extraction**: Quantum-proof extractors
8. **Experimental Results**: Comprehensive validation
9. **Discussion**: Deployment, limitations, future work

---

## Applications

### High-Security Cryptography
- Cryptographic key generation
- One-time pad creation
- Secure seed generation

### Gambling and Gaming
- Provably fair random number generation
- Regulatory compliance (trust metrics)

### Scientific Computing
- Monte Carlo simulations requiring certified randomness
- Quantum algorithm testing

### Research
- Quantum foundations experiments
- Randomness beacon services

---


## Future Work

1. **Experimental Validation**: Test with real quantum hardware (photon-based, phase-noise)
2. **Formal Security Proof**: Composable security framework proof
3. **Optimized Extraction**: Explore alternative quantum-proof extractors
4. **ML-Enhanced Self-Testing**: Machine learning for advanced attack detection
5. **Standardization**: Submit to NIST, ISO standardization bodies

---


## Acknowledgments

This work advances the state-of-the-art in practical quantum random number generation by introducing dynamic trust quantification. I thank the quantum cryptography community for foundational work on device-independent and source-independent approaches.

---

## References

Key references include:
- Pironio et al., "Random numbers certified by Bell's theorem," Nature 2010
- Li et al., "Source-independent quantum random number generation," Phys. Rev. X 2011
- Dupuis et al., "Entropy accumulation," Commun. Math. Phys. 2020
- DeVetak & Winter, "Distillation of secret key and entanglement," Proc. R. Soc. A 2005


