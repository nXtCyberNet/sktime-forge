# Mathematical Formulations and Statistical Logic

This document outlines the core statistical methods and mathematical decisions implemented in the MCP tools serving the `PipelineArchitectAgent`.

## 1. Seasonality Detection (`detect_seasonality.py`)

The seasonality tool relies on Autocorrelation (ACF) to find repeating cyclic patterns. To ensure the LLM agent receives precise hints without hallucinations caused by noise or trend, the following formulations are used:

### A. Trend Removal (Differencing)
If data has a strong non-stationary trend, ACF will artificially remain high across all lags. We apply first-order differencing before calculating the ACF:
$$ y'_{t} = y_{t} - y_{t-1} $$
This strips linear trend while preserving cyclic patterns.

### B. FFT-Based Autocorrelation
Calculating full cross-correlation is mathematically $O(N^2)$, which is too slow for large time series. We use the Fast Fourier Transform (FFT) to reduce the time complexity to $O(N \log N)$:
$$ F = \mathcal{F}(y' - \mu) $$
$$ \text{ACF}_{\text{raw}} = \mathcal{F}^{-1}(F \cdot F^*) $$
Where $F^*$ is the complex conjugate. The sequence is then normalized by the variance (the $0$-th lag).

### C. Statistical Significance
To avoid treating random noise as a seasonal period, we apply a 95% confidence interval bound for white noise:
$$ \text{Conf} = \frac{1.96}{\sqrt{N}} $$
Any ACF peak must strictly exceed this bound (and a base noise floor of $0.2$) to be considered a valid seasonal candidate.

---

## 2. Structural Break Detection (`check_structural_break.py`)

The structural break tool utilizes a Global Retrospective CUSUM (Cumulative Sum) test. It is designed to find sudden regime changes (level shifts) in the data.

### A. The No-Differencing Rule
Unlike seasonality, we **do not** difference the data for CUSUM. Differencing converts a step-change (level shift) into a single impulse point, destroying the cumulative plateau that CUSUM relies on.

### B. Cumulative Sum Formulation
The standard CUSUM array is built by cumulatively summing the mean-centered data:
$$ S_t = \sum_{i=1}^{t} (y_i - \mu) $$
The location of the structural break is exactly the point of maximum divergence from the mean:
$$ \text{Location} = \text{argmax}(|S_t|) $$

### C. The Allowance Parameter ($k$)
In online sequential CUSUM (Page's CUSUM), $k$ (often $0.5\sigma$) acts as "slack" to prevent the sum from drifting due to continuous mild white noise. 
* **Our Implementation:** We explicitly set $k = 0$. 
* **Reasoning:** Because we are evaluating the entire historical dataset at once (retrospective global curve), adding a slack parameter distorts the smooth arc of the Brownian bridge, making it impossible to cleanly extract the exact breakpoint via $\text{argmax}$.

### D. The Threshold Parameter ($h$)
The boundary that decides if a break is statistically "real" is derived from the maximum excursion of a standard Brownian bridge. 
* The base test statistic is normalized by the standard deviation and dataset size:
$$ \text{Test Stat} = \frac{\max(|S_t|)}{\sigma \sqrt{N}} $$

* Our threshold bound ($h$) is:
$$ h_{\text{threshold}} = 1.358 \times (1 + 0.1 \ln(N)) $$
Where $1.358$ corresponds to the 95% confidence bounds of a Kolmogorov-Smirnov test. We dynamically scale it up slightly ($\ln(N)$ penalty) for larger datasets to prevent long aggregations of pure white noise from occasionally piercing the standard bound by sheer random walk drift.
