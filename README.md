# Event-Driven Traffic Forecasting in FTTH Networks

## Overview
This repository contains a comprehensive analysis of traffic dynamics in a nationwide Fiber-To-The-Home (FTTH) network operated by Open Fiber in Italy, with a focus on identifying and forecasting rare but high-impact traffic events driven by external factors such as live sports broadcasts.

The project demonstrates that network traffic at the aggregation layer exhibits strong daily and weekly seasonality superimposed with sharp, transient peaks that can push utilization close to capacity despite relatively modest average loads. Using unsupervised learning techniques, we identify and cluster these anomalous events and reveal a systematic alignment between the most significant traffic peaks and scheduled football matches (Serie A and Champions League). Building on this empirical characterization, we propose an **event-aware probabilistic forecasting framework** that incorporates deterministic calendar information to improve prediction reliability during congestion-prone periods.

## Motivation and Background
Traffic dynamics in modern telecommunications networks are increasingly shaped by highly synchronized user behavior, driven by the proliferation of digital services and live content. These phenomena, commonly referred to as *flash crowds* [1, 2], arise when large populations of users simultaneously access network resources within short time intervals. Typical examples include live sports streaming, online events, and real-time content delivery [3]. As a result, network traffic exhibits pronounced short-term peaks, with peak-to-average ratios (PARs) that can significantly exceed typical daily patterns and put pressure on available capacity at the aggregation level.

Conventional capacity management strategies are largely based on static thresholding mechanisms, where operators react only when utilization exceeds predefined limits. While effective under relatively stable demand conditions, this paradigm is increasingly inadequate in the presence of bursty, high-simultaneity traffic. In particular, transient peaks may approach capacity limits even when average utilization remains moderate, leading to degradation of Quality of Experience (QoE) without triggering standard control mechanisms.

At the same time, regulatory constraints further complicate congestion management. In the European context, the principle of net neutrality, established by the Open Internet Regulation [4], requires equal treatment of traffic and restricts traffic management to exceptional cases, such as security threats or severe congestion events. The proposed Digital Network Act (DNA) [5] reinforces these requirements by introducing stricter transparency and compliance obligations. As a consequence, operators are increasingly required to anticipate congestion scenarios in advance, rather than mitigate them reactively.

Despite extensive research on traffic forecasting, anomaly detection, and flash crowd analysis, most existing approaches model traffic peaks as stochastic fluctuations inferred solely from historical observations. However, many high-impact traffic surges are not purely random but are instead driven by exogenous and partially predictable events, such as scheduled public activities. While prior work has explored the use of event information for traffic prediction in specific contexts (e.g., mobile networks or localized settings), a systematic, large-scale integration of deterministic event knowledge into probabilistic forecasting frameworks for Fiber-To-The-Home (FTTH) aggregation networks remains largely unexplored, with empirical studies on nationwide deployments being particularly scarce.

A further limitation of existing approaches is their focus on average prediction accuracy, rather than on the accurate characterization of rare but operationally critical events. In well-provisioned networks, congestion episodes are inherently sparse, making them difficult to capture using purely supervised machine learning approaches trained on historical data. This imbalance limits the ability of standard models to reliably predict congestion-prone conditions.

In this work, we advocate a complementary perspective in which data-driven techniques are used not only for prediction, but also to identify, characterize, and model the structure of event-driven traffic dynamics. By combining measurement-driven analysis with event detection and deterministic calendar information, forecasting can move beyond purely historical extrapolation and explicitly account for the mechanisms generating traffic peaks.

### Key Contributions
- **Large-scale empirical characterization**: Analysis of ~80 aggregation links across Italy covering January 2024 – August 2025
- **Event detection and clustering**: Unsupervised identification of anomalous traffic patterns using functional data analysis and Dynamic Time Warping
- **Event-to-trigger mapping**: Quantification of the relationship between football matches and peak traffic excursions
- **Event-aware forecasting**: Probabilistic predictions that condition residual distributions on calendar information
- **Improved prediction reliability**: Substantial gains in Prediction Interval Coverage Probability (PICP) during event periods while maintaining forecast sharpness

## Data

### Dataset Characteristics
- **Network coverage**: 180 links in Open Fiber's regional aggregation network (L2 level), distributed across:
  - 91 links in northern Italy
  - 29 links in central Italy
  - 60 links in southern Italy
  
- **Measurements**: 
  - Maximum and average throughput at 15-minute intervals
  - Link capacity information over time
  - Coverage period: January 2024 – August 2025

- **Preprocessing filters**: Links are retained if they satisfy:
  - Less than 30% missing data
  - Average traffic ≥ 1% of link capacity
  - Maximum peak load ≥ 10%
  
  **Result**: ~80 links retained for analysis

- **Statistics** (after filtering):
  - Link capacities: 10–100 Gbps
  - Median utilization: 22%
  - 95th-percentile utilization: 58%
  - Peak-to-Average Ratio (PAR): median 1.53, range [1.20, 3.04]
  - Saturation events: 0.0006% of observations

## Methodology

### 1. Time Series Decomposition
Each link's traffic signal is decomposed into three components using **Singular Spectrum Analysis (SSA)**:

- **Trend**: Long-term evolution capturing gradual changes in demand
- **Seasonal**: Recurring patterns at multiple time scales (6h, 8h, 12h, 24h, 7d cycles)
- **Residual**: Short-term fluctuations and stochastic variations

Parameters: 28-day window, top-10 components extracted

### 2. Correlation Analysis
- Compute Spearman rank correlation between all pairs of links for each component
- Apply hierarchical clustering to correlation-based distance matrix
- Identify groups of links with similar traffic dynamics
- Use elbow criterion to determine optimal number of clusters

**Finding**: Two main clusters emerge with clear geographic separation (northern/central vs. southern Italy)

### 3. Event Detection and Clustering
**Step C1**: Anomalous segment extraction
- Apply **FastMUOD** (Fast Multivariate Outlier Detection) to identify outlier days
- Characterize non-outlier day distribution
- Apply **modified z-score anomaly detection** to extract segments exceeding ±3·MAD threshold
- Group consecutive anomalous 15-minute slots into segments

**Step C2**: Event clustering
- Compute pairwise distances between segments using **Dynamic Time Warping (DTW)**
- Apply hierarchical clustering to distance matrix
- Select optimal cluster count via elbow method on Within-Cluster Sum of Squares (WCSS)

### 4. Event Influence Estimation
For each link, estimate event-conditioned probability density functions (PDFs) using kernel density estimation:

- **$\tilde{f}_{All}$**: Distribution of all residual traffic values (baseline)
- **$\tilde{f}_{\emptyset}$**: Distribution without any Serie A or Champions League matches
- **$\tilde{f}_{S_n}$**: Distribution with $n$ Serie A matches ($n \in \{1,...,10\}$)
- **$\tilde{f}_{C_m}$**: Distribution with $m$ Champions League matches ($m \in \{1,...,6\}$)

Use statistical tests (Kolmogorov–Smirnov, Mann–Whitney U) to verify significance of distributional differences.

### 5. Probabilistic Forecasting
**Prediction components**:
- **Trend**: Repeat last observed value from training set
- **Seasonal**: Repeat seasonality of final week
- **Residual**: Monte Carlo sampling (1000 samples) from event-conditioned PDFs

**Prediction intervals**: 99% confidence intervals $[q_{0.01}, q_{0.99}]$ computed from sampled residuals

**Final forecast**: Sum trend + seasonal + residual bounds

### 6. Evaluation Metrics
- **PICP** (Prediction Interval Coverage Probability): Fraction of observed values falling within prediction interval
  - Computed over full period and restricted to match time slots
  - Target: 0.99 (99% coverage)

- **PINAW** (Prediction Interval Normalized Average Width): Sharpness of intervals
  - Average interval width normalized by data range
  - Lower is better (tighter intervals while maintaining coverage)

## Data Availability
Due to confidentiality constraints, the complete network traffic dataset cannot be made publicly available. The analysis presented in this repository is based on proprietary data provided by Open Fiber. However, a one-month normalized sample from the two analyzed links is provided.

**If you use this data in your research or work, please cite this paper:**

...

For researchers interested in applying or adapting this methodology to their own datasets, the code structure and analytical pipeline are fully documented and transferable. The methodological framework is independent of the specific network or data provider and can be applied to other FTTH operators or similar aggregation networks, subject to data access agreements with the respective operators.

## Referencies
[1] . Ari, B. Hong, E. L. Miller, S. A. Brandt, and D. D. E. Long, “Managing flash crowds on the internet” in Proceedings of the 11th IEEE/ACM International Symposium on Modeling, Analysis and Simulation of Computer Telecommunications Systems (MASCOTS 2003). IEEE, Oct. 2003, pp. 246–249.

[2] J. David and C. Thomas, “Discriminating flash crowds from DDoS attacks using efficient thresholding algorithm”, Journal of Parallel and Distributed Computing, vol. 152, pp. 79–87, 2021.

[3] F. Liu, B. Li, L. Zhong, B. Li, H. Jin, and X. Liao, “Flash crowd in P2P live streaming systems: Fundamental characteristics and design implications”, IEEE Transactions on Parallel and Distributed Systems, vol. 23, no. 7, pp. 1227–1239, 2012.

[4] European Parliament, “Regulation (EU) 2015/2120,” 2015.

[5] European Commission, “Digital networks act (dna),” 2026.

## Copyright and license
Copyright _Daniele Baccega, Paolo Castagno, Edoardo Acquarone, Matteo Sereno, Francesca Parasecolo, Sara Mazzarella, Francesco Carpentieri, Claudio Valenti, Alessandro Dellaqueva_

![CC BY-NC-SA 3.0](http://ccl.northwestern.edu/images/creativecommons/byncsa.png)

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.  To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/3.0/ or send a letter to Creative Commons, 559 Nathan Abbott Way, Stanford, California 94305, USA.