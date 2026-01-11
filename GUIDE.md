# Complete Guide to TE-SI-QRNG Implementation

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the System](#understanding-the-system)
3. [Implementation Details](#implementation-details)
4. [Running Experiments](#running-experiments)
5. [Interpreting Results](#interpreting-results)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quick demo
python demo.py

# 3. Run all experiments 
python run_all.py
```

### First Usage 

```python
from te_si_qrng import TrustEnhancedQRNG
from simulator import QuantumSourceSimulator, SourceParameters, SourceType

# Create ideal quantum source
params = SourceParameters(source_type=SourceType.IDEAL)
source = QuantumSourceSimulator(params, seed=42)

# Create TE-SI-QRNG
qrng = TrustEnhancedQRNG(block_size=10000) #block_sie can be alter for fast excuetion.

# Generate certified random bits
bits, metadata = qrng.generate_certified_random_bits(
    n_bits=10000, #n_bits can be alter for fast excuetion.
    source_simulator=source
)

print(f"Generated {len(bits)} bits with trust {metadata[0]['trust_score']:.3f}")
```

---

## Understanding the System

### Core Concept

**Problem**: Traditional QRNG either:
- Assumes perfect trust (SI-QRNG) → vulnerable to device imperfections
- Requires impractical Bell tests (DI-QRNG) → not deployable

**Solution**: TE-SI-QRNG dynamically measures trust and adjusts entropy certification

### Key Components

#### 1. Trust Vector T = {ε_bias, ε_drift, ε_corr, ε_leak}

- **ε_bias**: Deviation from 50/50 distribution
- **ε_drift**: Temporal instability (detector aging, temperature)
- **ε_corr**: Memory effects between bits
- **ε_leak**: Side-channel leakage indicators

#### 2. Self-Testing Layer

**Statistical Tests**:
- Santha-Vazirani test → detects biased sources
- Runs test → detects non-independence
- Autocorrelation → detects memory effects
- Frequency test → detects distribution bias

**Quantum Witness Tests**:
- Dimension witness → verifies quantum dimensionality
- Energy constraint → checks physical consistency
- POVM consistency → validates quantum measurements

**Physical Monitoring**:
- Detector efficiency tracking
- Dark count rate analysis

#### 3. Trust-Adjusted Entropy

```
H_min^trusted(X|E) = H_min(X|E) - f(T)
```

where `f(T) = -log2(Trust(T))` is the penalty

#### 4. Adaptive Extraction

Extraction rate adjusts based on trust:
```
rate = Trust(T) × efficiency × H_min^trusted
```

---

## Implementation Details

### Architecture Flow

```
┌─────────────────────────┐
│  Quantum Source         │ ← Untrusted
└────────────┬────────────┘
             │
             v
┌─────────────────────────┐
│  Measurement Module     │ ← Semi-trusted
└────────────┬────────────┘
             │
             v
┌─────────────────────────┐
│  Self-Testing Layer     │ ← Core innovation
│  - Statistical tests    │
│  - Quantum witnesses    │
│  - Physical monitoring  │
└────────────┬────────────┘
             │
             v
┌─────────────────────────┐
│  Entropy Estimator      │ ← Trust-adjusted
└────────────┬────────────┘
             │
             v
┌─────────────────────────┐
│  Randomness Extractor   │ ← Adaptive Toeplitz
└────────────┬────────────┘
             │
             v
┌─────────────────────────┐
│  Certified Random Bits  │
└─────────────────────────┘
```

### Key Classes

#### `TrustVector`
Stores trust parameters:
```python
trust_score = 1 - ||T||_2 / 2
trust_penalty = -log2(trust_score)
```

#### `StatisticalSelfTester`
Implements NIST-style statistical tests adapted for self-testing

#### `QuantumWitnessTester`
Tests quantum properties without full Bell inequality

#### `PhysicalDriftMonitor`
Tracks temporal changes in hardware

#### `EntropyEstimator`
Computes trust-adjusted min-entropy with finite-size corrections

#### `RandomnessExtractor`
Toeplitz hashing with adaptive output length

#### `TrustEnhancedQRNG`
Main system integrating all components

---

## Running Experiments

### Individual Experiments

#### Experiment 1: Trust Quantification
```python
from experiment import ExperimentRunner

runner = ExperimentRunner(output_dir="results")
results = runner.experiment_1_trust_quantification(n_bits=100000) #n_bits can be alter for fast excuetion.
```

**What it tests**: Whether trust metrics correctly identify different source types

**Expected outcome**: Ideal sources get ~0.99 trust, attacked sources get <0.7 trust

#### Experiment 2: Entropy Certification
```python
results = runner.experiment_2_entropy_certification(n_bits=50000) #n_bits can be alter for fast excuetion.
```

**What it tests**: Certified entropy vs. empirical entropy

**Expected outcome**: Certified entropy is always conservative (below empirical)

#### Experiment 3: Attack Detection
```python
results = runner.experiment_3_attack_detection(n_bits=50000) #n_bits can be alter for fast excuetion.
```

**What it tests**: Response to increasing attack strength

**Expected outcome**: Trust degrades, extraction rate reduces

#### Experiment 4: Comparison with SI-QRNG
```python
results = runner.experiment_4_comparison_with_si_qrng(n_bits=50000) #n_bits can be alter for fast excuetion.
```

**What it tests**: TE-SI-QRNG vs. standard approach

**Expected outcome**: Better quality-security trade-off

#### Experiment 5: Temporal Adaptation
```python
results = runner.experiment_5_temporal_adaptation(n_blocks=20) 
```

**What it tests**: Adaptation to time-varying source quality

**Expected outcome**: Trust tracks source quality changes

### All Experiments at Once

```bash
python run_all.py
```

Runs all 5 experiments and generates:
- 5 figures in `results/figures/`
- 5 JSON data files in `results/data/`

---

## Interpreting Results

### Trust Scores

| Trust Score | Interpretation | Action |
|-------------|----------------|--------|
| 0.95 - 1.00 | Excellent | Full extraction rate |
| 0.85 - 0.95 | Good | Normal operation |
| 0.70 - 0.85 | Fair | Reduced extraction |
| 0.50 - 0.70 | Poor | Conservative mode |
| < 0.50 | Critical | Halt & investigate |

### Extraction Rates

| Rate | Quality | Use Case |
|------|---------|----------|
| 0.85-0.90 | Ideal | High-security crypto |
| 0.70-0.85 | Good | Standard crypto |
| 0.50-0.70 | Acceptable | Non-critical apps |
| 0.30-0.50 | Poor | Testing/debug only |
| < 0.30 | Unacceptable | Do not use |

### Figures Interpretation

#### Figure 1: Trust Comparison
- **X-axis**: Different source scenarios
- **Y-axis**: Trust score
- **Look for**: Clear separation between ideal and attacked sources

#### Figure 2: Entropy Comparison
- **Left plot**: Empirical vs. certified entropy
  - Certified should be below empirical (conservative)
- **Right plot**: Extraction rates
  - Should decrease for low-quality sources

#### Figure 3: Attack Response
- **Multiple subplots**: Different attack strengths
- **Look for**: Decreasing trust and extraction rate as attacks increase

#### Figure 4: Comparison with SI-QRNG
- **Output bits**: TE-SI-QRNG may produce less for bad sources (safer)
- **Quality**: TE-SI-QRNG should maintain higher quality

#### Figure 5: Temporal Adaptation
- **Top plot**: Trust tracking source quality
  - Should follow sinusoidal pattern
- **Bottom plot**: Performance metrics adapting

---

## Customization

### Custom Quantum Source

Create your own source by subclassing:

```python
class MyQuantumSource:
    def generate_block(self, n_bits):
        # Your quantum source implementation
        bits = ...  # Generate quantum random bits
        bases = ...  # Measurement basis choices (optional)
        signal = ...  # Raw physical signal (optional)
        return bits, bases, signal

# Use with TE-SI-QRNG
qrng = TrustEnhancedQRNG(block_size=10000)
random_bits, metadata = qrng.generate_certified_random_bits(
    n_bits=10000,
    source_simulator=my_source
)
```

### Adjust Parameters

```python
# More conservative security
qrng = TrustEnhancedQRNG(
    block_size=10000,
    security_parameter=1e-9,  # Tighter security (default: 1e-6)
    extractor_efficiency=0.85  # More conservative (default: 0.9)
)

# Faster processing (less conservative)
qrng = TrustEnhancedQRNG(
    block_size=5000,  # Smaller blocks
    security_parameter=1e-4,  # Looser security
    extractor_efficiency=0.95  # More aggressive extraction
)
```

### Custom Self-Tests

Add your own tests:

```python
class MyTester:
    def my_custom_test(self, bits):
        # Your test implementation
        passes = ...
        metric = ...
        return passes, metric

# Integrate into TE-SI-QRNG
qrng = TrustEnhancedQRNG(block_size=10000)
# Add custom tester to qrng...
```

---

## Troubleshooting

### Issue: "Import Error"

**Cause**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue: "Trust score always low"

**Possible causes**:
1. Source is actually biased → check source parameters
2. Block size too small → increase to 10000+
3. Security parameter too tight → relax to 1e-6

**Debug**:
```python
bits, bases, signal = source.generate_block(10000)
trust = qrng.run_self_tests(bits, bases, signal)
print(f"ε_bias: {trust.epsilon_bias}")
print(f"ε_corr: {trust.epsilon_corr}")
# Identify which component is high
```

### Issue: "Very low extraction rate"

**Cause**: Low trust or low entropy

**Solution**:
1. Check trust score: `metadata['trust_score']`
2. Check min-entropy: `metadata['h_min_trusted']`
3. If source quality is actually poor, this is correct behavior

### Issue: "Experiments take too long"

**Solution**: Reduce sample sizes:
```python
runner.experiment_1_trust_quantification(n_bits=10000)  # Instead of 100000
```

### Issue: "Figures not generated"

**Cause**: matplotlib backend issues

**Solution**:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

---

## Advanced Topics

### Integrating with Real Hardware

1. **Photon-based QRNG**:
   - Replace `generate_block()` with actual detector readout
   - Map photon counts to bits
   - Pass detector efficiency and dark count rates

2. **Phase noise QRNG**:
   - Use quadrature measurements as `raw_signal`
   - Extract bits from sign of measurements
   - Monitor laser stability

3. **Other quantum sources**:
   - Ensure binary output (or digitize)
   - Provide physical monitoring data if available
   - Calibrate trust parameters on known-good source

### Production Deployment

**Recommended configuration**:
```python
qrng = TrustEnhancedQRNG(
    block_size=100000,  # Large blocks for statistics
    security_parameter=1e-9,  # High security
    extractor_efficiency=0.85  # Conservative
)

# Continuous monitoring
while True:
    bits, metadata = qrng.generate_certified_random_bits(...)

    if metadata[0]['trust_score'] < 0.7:
        alert_operator()  # Send alert

    if metadata[0]['trust_score'] < 0.5:
        halt_system()  # Stop output

    log_metrics(metadata)  # Record for analysis
```

### Performance Optimization

1. **Parallel processing**: Process multiple blocks in parallel
2. **Caching**: Cache extractor matrices for repeated use
3. **Hardware acceleration**: Use NumPy with MKL or GPU libraries

---



---

## Further Reading


1. **Code**: Browse source code with extensive documentation
2. **Demo**: Run `python demo.py` for interactive examples
3. **Experiments**: See `experiment.py` for validation methodology

---

