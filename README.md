# Diagnostic Analysis: Why SPAI Underperforms on Medical Images

This document presents quantitative analyses comparing the spectral and statistical properties of natural and medical images, with the goal of understanding why SPAI (Spectral AI-Generated Image Detector) — a model that detects AI-generated images using spectral learning — performs well on natural-scene images but poorly on medical-domain images.

Each analysis section begins with a plain-language explanation of the metrics used, followed by the results and an objective interpretation.

---

## Analysis 1: Power Spectral Density (PSD) & Spectral Slope (α)

### What This Analysis Measures

Every image can be decomposed into a sum of patterns at different spatial *frequencies* using the **2D Fourier Transform (FFT)**. Low frequencies represent smooth, large-scale structures (e.g., overall brightness gradients, large shapes), while high frequencies represent fine details (e.g., sharp edges, textures, noise).

The **Power Spectral Density (PSD)** tells us *how much energy* the image contains at each frequency. By radially averaging the 2D power spectrum, we get a single curve: power as a function of frequency.

### Metrics Explained

#### α (Alpha) — The Spectral Slope

Natural photographs of the real world tend to follow a well-known statistical regularity:

$$P(f) = C \cdot f^{-\alpha}$$

where $P(f)$ is the power at spatial frequency $f$, $C$ is a constant, and $\alpha$ is the **spectral slope exponent**. On a log-log plot, this becomes a straight line with slope $-\alpha$.

| α value | What it means |
|---------|---------------|
| Higher α (e.g., 3.0+) | Power drops off **steeply** with frequency — the image is **very smooth**, dominated by low-frequency content, with relatively little high-frequency detail |
| Lower α (e.g., 2.0) | Power drops off **more gradually** — the image has a **richer mix** of low and high-frequency content (more edges, textures, fine details) |

**Why α matters for SPAI:** SPAI works by splitting each image into a low-frequency component and a high-frequency component using a fixed circular mask in the frequency domain (radius = 16 pixels in a 224×224 FFT). It then measures how well a ViT backbone can *reconstruct* the relationship between these components. If the spectral slope α is very different between natural and medical images, the same fixed frequency cutoff captures fundamentally different proportions of the image's information — potentially rendering the method ineffective.

#### R² — Goodness of Fit

R² measures how well the $P(f) = C \cdot f^{-\alpha}$ power-law model actually fits the image's radial PSD. An R² of 1.0 means a perfect fit (the image's spectrum follows the power law exactly). High R² means the image obeys the expected 1/f statistical regularity; low R² means it doesn't.

#### |Δα| — The Alpha Gap

|Δα| is the absolute difference in mean α between two groups (e.g., real vs. synthetic images within a domain). A larger gap means the two groups are more spectrally distinguishable — making it easier for a spectral-based detector like SPAI to tell them apart.

#### KS Statistic (Kolmogorov-Smirnov Test)

The KS test is a non-parametric statistical test that measures whether two sets of numbers come from the same underlying distribution. It reports:

- **KS statistic** (range 0 to 1): The maximum vertical distance between the cumulative distribution functions of the two groups. A value near 1 means the distributions are almost completely separated; near 0 means they overlap almost entirely.
- **p-value**: The probability of seeing this large a KS statistic if the two groups were actually drawn from the same distribution. A very small p-value (e.g., < 0.001, marked ***) means the difference is statistically significant — it is extremely unlikely to have occurred by chance.

> **Note:** Statistical significance (small p-value) does NOT mean the difference is practically large. A tiny but real difference can be "statistically significant" with enough data points. That is why we also report effect size (Cohen's d).

#### Cohen's d — Effect Size

Cohen's d measures the *practical magnitude* of the difference between two groups, expressed in units of standard deviation:

$$d = \frac{|\mu_A - \mu_B|}{\sigma_{\text{pooled}}}$$

| Cohen's d | Interpretation |
|-----------|----------------|
| < 0.2 | Negligible difference |
| 0.2 – 0.5 | Small effect |
| 0.5 – 0.8 | Medium effect |
| > 0.8 | Large effect |

A large Cohen's d means the distributions are well-separated and a classifier could reliably distinguish the two groups based on this metric alone. A small Cohen's d means the distributions overlap heavily, making classification unreliable.

---

### Results

**Data:** 150 images per group, all resized to 256×256 and converted to grayscale before FFT.

| Source | Dataset |
|--------|---------|
| Real Natural | COCO 2017 val (natural photographs) |
| Synthetic Natural | CIFAKE (Stable Diffusion 1.4 generated images) |
| Real Medical | Indiana University Chest X-rays (frontal + lateral views) |
| Synthetic Medical | AI-generated chest X-rays (paired with real, generated via LLM-guided generation) |

#### Table 1: Spectral Slope (α) Summary

| Group | N | α mean | α std | α 95% CI | R² mean |
|-------|---|--------|-------|----------|---------|
| Real Natural | 150 | 2.5562 | 0.2882 | [2.5101, 2.6023] | 0.9880 |
| Synthetic Natural | 150 | 3.3313 | 0.2437 | [3.2923, 3.3703] | 0.9561 |
| Real Medical | 150 | 2.8680 | 0.1826 | [2.8388, 2.8973] | 0.9941 |
| Synthetic Medical | 150 | 3.0384 | 0.1326 | [3.0172, 3.0596] | 0.9958 |

#### Table 2: Statistical Separability (Real vs. Synthetic)

| Comparison | |Δα| | KS stat | p-value | Cohen's d | Effect |
|------------|------|---------|---------|-----------|--------|
| Real Natural vs Synthetic Natural | **0.7751** | 0.8667 | 1.60e-58 *** | **2.91** | Large |
| Real Medical vs Synthetic Medical | **0.1704** | 0.5067 | 6.93e-18 *** | **1.07** | Large |

#### Table 3: Cross-Domain Differences

| Comparison | |Δα| | KS stat | p-value | Cohen's d |
|------------|------|---------|---------|-----------|
| Real Natural vs Real Medical | 0.3119 | 0.5933 | 7.59e-25 *** | 1.30 |
| Synthetic Natural vs Synthetic Medical | 0.2929 | 0.5867 | 2.93e-24 *** | 1.50 |

---

### Interpretation

#### Finding 1: The spectral separation gap is 4.5× smaller for medical images

The most critical finding is the difference in **|Δα|** between domains:

- **Natural domain**: |Δα| = 0.7751 (real α = 2.56, synthetic α = 3.33)
- **Medical domain**: |Δα| = 0.1704 (real α = 2.87, synthetic α = 3.04)

The spectral slope gap in the medical domain is only **22% of the natural domain gap** (0.1704 / 0.7751 = 0.22). This means that spectral slope — the primary signal that frequency-based detectors exploit — provides **4.5× less separation** between real and synthetic medical images than it does for natural images.

#### Finding 2: Both domains do show statistically significant differences

It would be inaccurate to claim there is *no* spectral difference in the medical domain. The KS test yields p = 6.93e-18 for medical images, which is highly significant (***), and Cohen's d = 1.07 still exceeds the 0.8 threshold for a "large" effect.

However, this must be interpreted carefully:

- **Cohen's d of 1.07 for medical vs. 2.91 for natural** means the natural domain has **2.7× stronger practical separability**. While 1.07 is technically "large" by convention, it represents far more distributional overlap than d = 2.91.
- **KS stat of 0.51 for medical vs. 0.87 for natural** tells the same story: the cumulative distributions overlap much more in the medical domain.

#### Finding 3: Medical images are inherently more spectrally homogeneous

The standard deviations of α tell an important story:

| Group | α std |
|-------|-------|
| Real Natural | 0.2882 |
| Synthetic Natural | 0.2437 |
| Real Medical | **0.1826** |
| Synthetic Medical | **0.1326** |

Medical images have **much lower variance** in their spectral slope. Real medical X-rays have 37% less variance than real natural images, and synthetic medical X-rays have 46% less variance than synthetic natural images.

This means medical images — both real and AI-generated — occupy a **narrower, more compressed region** of spectral space. Because both real and synthetic medical images are tightly clustered with only a thin gap between them (Δα = 0.17), a spectral-based detector has very little room to place a decision boundary.

#### Finding 4: AI generators produce steeper spectral slopes in both domains

In both natural and medical domains, AI-generated images have **higher α** (steeper spectral falloff) than real images:

- Natural: 2.56 (real) → 3.33 (synthetic), +0.78
- Medical: 2.87 (real) → 3.04 (synthetic), +0.17

This suggests that generative models tend to produce images with **relatively less high-frequency content** (smoother, fewer fine details) compared to real images. This is consistent with findings in the literature (e.g., the Edge-Enhanced ViT paper) that AI-generated images exhibit higher smoothness and reduced noise.

However, the key insight is: **medical X-rays are already inherently smooth** (α = 2.87 for real medical vs. 2.56 for real natural), so the additional smoothing introduced by AI generation produces a much smaller *relative* shift.

#### Finding 5: The 1/f power-law fits better for medical images

All R² values are high (> 0.95), but medical images actually show a **better** fit to the 1/f power law than natural images (R² = 0.994–0.996 vs. 0.956–0.988). This is likely because medical X-ray images have simpler structural composition (largely homogeneous tissue regions with sparse anatomical landmarks) that more closely approximates the ideal 1/f model.

---

### Summary for Analysis 1

| Aspect | Natural Domain | Medical Domain | Implication for SPAI |
|--------|---------------|----------------|---------------------|
| |Δα| (real vs synthetic) | 0.775 | 0.170 | 4.5× less spectral signal available |
| Cohen's d | 2.91 | 1.07 | 2.7× less practical separability |
| KS statistic | 0.87 | 0.51 | More distributional overlap |
| α variance (real) | 0.288 | 0.183 | Tighter clustering → harder to separate |
| α variance (synthetic) | 0.244 | 0.133 | Even tighter → nearly overlapping |

> **Conclusion for Analysis 1:** The spectral slope gap between real and synthetic images is dramatically smaller in the medical domain (|Δα|=0.17) compared to the natural domain (|Δα|=0.78). While the difference is still statistically significant, the practical separability is much weaker, and the inherently low spectral variance of medical images further compresses the available signal. This provides quantitative evidence for why SPAI's spectral learning approach — which relies on detecting deviations in spectral structure — has reduced discriminative power on medical images.

---

## Analysis 2: Frequency Band Energy Distribution

### What This Analysis Measures

SPAI doesn't use the entire frequency spectrum at once. It splits every image into two parts using a **circular mask** in the frequency domain:

- **Low-frequency band** (inside the circle): smooth, large-scale structures
- **High-frequency band** (outside the circle): fine details, edges, textures, noise

The mask has a fixed **radius of 16** on a **224×224** FFT grid (from SPAI's config). This means only **1.6% of all frequency bins** fall in the low-frequency circle — yet as we'll see, that tiny region captures the overwhelming majority of image energy.

SPAI then feeds the original image, the low-freq filtered image, and the high-freq filtered image through a ViT backbone separately, and compares their feature representations using cosine similarity. **If the high-frequency band contains almost no energy, the high-freq filtered image is essentially blank noise**, and the ViT features extracted from it are meaningless — breaking SPAI's core mechanism.

### Metrics Explained

#### E_low / E_high (Energy Fractions)

After computing the 2D FFT of an image, we measure what percentage of total energy falls inside vs. outside the circular mask.

- **E_low = 95%** means 95% of the image's information is in smooth, low-frequency patterns.
- **E_high = 5%** means only 5% is in fine details and edges.

#### E_ratio (Low-to-High Ratio)

Simply E_low ÷ E_high. If E_ratio = 40×, the low-frequency band has 40 times more energy than the high-frequency band. Higher values mean the high-freq band is increasingly empty and uninformative.

#### Spectral Centroid

The "center of mass" of the power spectrum. Think of it like a balance point — if most energy is at low frequencies, the centroid is near the center (low value). A higher centroid means energy is more spread across frequencies.

---

### Results

#### Table 1: Energy Distribution Summary

| Group | N | E_low (%) | E_high (%) | E_ratio (L/H) | Spectral Centroid |
|-------|---|-----------|------------|----------------|-------------------|
| Real Natural | 150 | 95.72% | **4.28%** | 44× | 2.58 |
| Synthetic Natural | 150 | 99.71% | **0.29%** | 682× | 0.79 |
| Real Medical | 150 | 99.47% | **0.53%** | 264× | 0.62 |
| Synthetic Medical | 150 | 99.44% | **0.56%** | 209× | 0.74 |

#### Table 2: Separability of Real vs. Synthetic (High-Freq Energy)

| Comparison | Δ E_high | KS stat | p-value | Cohen's d | Effect |
|------------|----------|---------|---------|-----------|--------|
| Real Natural vs Synthetic Natural | **3.988%** | 0.9267 | 7.86e-70 *** | **1.90** | Large |
| Real Medical vs Synthetic Medical | **0.027%** | 0.2733 | 2.45e-05 *** | **0.09** | Negligible |

#### Table 3: Separability on Other Energy Metrics

| Metric | Domain | Cohen's d | Effect |
|--------|--------|-----------|--------|
| Energy Ratio | Natural | 0.97 | Large |
| Energy Ratio | Medical | 0.40 | Small |
| Spectral Centroid | Natural | 1.55 | Large |
| Spectral Centroid | Medical | 0.45 | Small |

---

### Interpretation

#### Finding 1: Medical images have almost no high-frequency energy

Real natural images put **4.28%** of their energy in the high-frequency band. That may sound small, but it's enough — it contains edges, textures, and fine details that differ between real and AI-generated images.

Real medical X-rays put only **0.53%** in the high-frequency band — roughly **8× less** than natural images. Medical images are dominated by smooth tissue gradients and large anatomical structures, with very little fine-grained texture.

#### Finding 2: The high-frequency gap between real and fake is essentially zero for medical images

This is the most critical finding of Analysis 2:

- **Natural domain**: Real images have 4.28% high-freq energy, synthetic have 0.29% → a gap of **3.99 percentage points**.
- **Medical domain**: Real images have 0.53%, synthetic have 0.56% → a gap of **0.03 percentage points**.

The gap is **148× smaller** in the medical domain. Cohen's d drops from **1.90** (large, easily separable) to **0.09** (negligible — the distributions are virtually identical).

In practical terms: when SPAI applies its frequency mask to a medical X-ray, the resulting high-frequency image is nearly empty for *both* real and synthetic images. The ViT backbone receives near-identical (near-zero) inputs in both cases, making the cosine similarity scores meaningless.

#### Finding 3: Real medical images are already as "smooth" as synthetic natural images

A striking observation from the data:

| Group | E_high |
|-------|--------|
| Real Medical | 0.53% |
| Synthetic Natural | 0.29% |

Real medical X-rays have comparable high-frequency content to *AI-generated* natural images. The smoothness that SPAI interprets as a sign of AI generation in natural images is simply the **normal baseline** for medical images. This creates a fundamental confusion: the model associates "smooth spectrum" with "fake," but medical images are inherently smooth regardless of whether they're real or AI-generated.

#### Finding 4: The spectral centroid confirms the same pattern

The spectral centroid (center of mass of energy across frequencies) reinforces all the above:

- Natural: centroid shifts from 2.58 (real) to 0.79 (synthetic) — a large, detectable shift.
- Medical: centroid is at 0.62 (real) vs 0.74 (synthetic) — barely any difference, and both are already very low.

---

### Summary for Analysis 2

| Aspect | Natural Domain | Medical Domain | Ratio |
|--------|---------------|----------------|-------|
| E_high gap (real vs synthetic) | 3.99% | 0.03% | 148× smaller |
| Cohen's d (E_high) | 1.90 | 0.09 | 21× weaker |
| KS statistic (E_high) | 0.93 | 0.27 | 3.4× smaller |
| Cohen's d (Spectral Centroid) | 1.55 | 0.45 | 3.4× weaker |

> **Conclusion for Analysis 2:** Using SPAI's exact frequency mask (radius=16), the high-frequency energy gap between real and synthetic images is **148× smaller** in the medical domain compared to natural images, with a Cohen's d of just 0.09 (negligible effect). This means the high-frequency filtered image — a central input to SPAI's detection mechanism — carries virtually no discriminative signal for medical images. The model cannot distinguish real from fake if both produce near-identical (near-empty) high-frequency representations.

---

*Analysis 3, 4, and 5 will be appended as they are completed.*
