
[1] J. Durbin and S. J. Koopman, *Time Series Analysis by State Space Methods*, 2nd ed. Oxford, U.K.: Oxford Univ. Press, 2012. 

[2] A. C. Harvey, *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge, U.K.: Cambridge Univ. Press, 1990.

[3] S. Moritz and T. Bartz-Beielstein, “imputeTS: Time Series Missing Value Imputation in R,” *The R Journal*, vol. 9, no. 1, pp. 207–218, 2017.

[4] E. Afrifa-Yamoah, U. A. Mueller, S. M. Taylor, and A. J. Fisher, “Missing data imputation of high-resolution temporal climate time series data,” *Meteorological Applications*, vol. 27, e1873, 2020. 

[5] S. Moritz, A. Sardà, T. Bartz-Beielstein, M. Zaefferer, and J. Stork, “Comparison of different methods for univariate time series imputation in R,” *arXiv preprint* arXiv:1510.03924, 2015. 

[6] N. Niako, J. D. Melgarejo, G. E. Maestre, and K. P. Vatcheva, “Effects of missing data imputation methods on univariate blood pressure time series data analysis and forecasting with ARIMA and LSTM,” *BMC Medical Research Methodology*, vol. 24, art. 320, 2024.

[7] A. Hadeed *et al.*, “Imputation methods for addressing missing data in short-term monitoring of air pollutants,” *Science of the Total Environment*, vol. 730, art. 139140, 2020.

[8] L. E. Alejo-Sanchez *et al.*, “Missing data imputation of climate time series: A review,” *MethodsX*, vol. 15, art. 103455, 2025.

[9] A. Cunha, J. Poças, and S. Rodrigues, “Process automation for treatment of missing values in e-invoice data,” in *Proc. uRos 2022 (Use of R in Official Statistics)*, 2022, conference abstract (Statistics Portugal).

[10] C. Mantuano *et al.*, “Data imputation methods for intermittent renewable energy sources: Implications for energy system modeling,” *Energy Conversion and Management*, vol. 339, 119857, 2025.

---

#### [1] Durbin & Koopman, 2012 – core theory for state-space + Kalman

* Presents the general state-space framework and derives Kalman filtering and smoothing for linear Gaussian models.
* Shows how trends, seasonality, AR components, and regression on covariates (e.g. temperature, calendar) can be written as a structural time-series (STS) model.
* Explains how to handle missing observations: the filter simply skips the update step when an observation is missing, and the smoother reconstructs optimal estimates using past and future data.
* This is the main theoretical reference to quote when justifying using STS + Kalman smoothing for imputation in the project. 

#### [2] Harvey, 1990 – structural time-series models and Kalman filter

* Classical reference focused specifically on structural models (local level, local linear trend, seasonal, cycle, regression) plus the Kalman filter.
* Emphasizes interpretability: each component (trend, seasonality, regression on temperature/holidays) is explicitly parameterized.
* Shows how to fit STS models by maximum likelihood (using Kalman filter) and then use the Kalman smoother to reconstruct hidden states and missing data.
* Good high-level reference to motivate “unobserved components” models for energy/irradiance series.

#### [3] Moritz & Bartz-Beielstein, 2017 – imputeTS package

* Introduces the R package **imputeTS**, which provides several univariate time-series imputation methods, including `na.kalman`.
* `na.kalman` has two main options:

  * `StructTS`: a **structural time-series model with Kalman smoothing** (local level/seasonal).
  * `auto.arima`: ARIMA in state-space form with Kalman smoothing.
* Using benchmark sensor datasets (including a heating system with >600k points and realistic missing patterns), they show:

  * Methods exploiting trend and seasonality (Kalman + seasonal decomposition) typically outperform simple mean/LOCF for strongly seasonal series.
  * Kalman-based imputation remains reliable even for long gaps, although it can be computationally heavy for very long series. 
* For you: this is a direct precedent of using Kalman/STS to impute **large, high-frequency sensor data** (very similar to the production/irradiance signals).

#### [4] Afrifa-Yamoah et al., 2020 – hourly climate data

* Task: impute missing values in **hourly** temperature, humidity, and wind speed from 4 weather stations (12-month series).
* Methods compared:

  * Structural time-series model with Kalman smoothing.
  * ARIMA model with Kalman smoothing.
  * Multiple linear regression using other climate variables.
* They artificially create 10% MCAR gaps and compare MAE, RMSE, sMAPE:

  * Multiple linear regression is slightly best on average.
  * ARIMA+Kalman is second.
  * STS+Kalman is close behind; all three yield *very low* errors and very plausible imputations.
* Conclusion: for **high-resolution climate series**, Kalman-based time-series models (structural or ARIMA) are fully competitive with regression and can be recommended for practical use. 
* For you: this is almost exactly the context for exogenous inputs (temperature, possibly wind; in the case also GHI/DNI).

#### [5] Moritz et al., 2015 – systematic univariate TS imputation study

* Compares multiple univariate imputation methods (mean, LOCF, linear/spline, moving averages, etc.) on synthetic and real series with various missingness patterns.
* Shows there is no universally best method, but **Kalman-based and other structure-aware methods** are consistently among the strongest for seasonal data. 
* Inspired the design of imputeTS and its default recommendations (try `na.kalman` / seasonal methods on complex data).

#### [6] Niako et al., 2024 – Kalman filtering vs other imputers for forecasting

* Application: 24-hour **ambulatory blood pressure** time series; they inject MCAR gaps at 10%, 15%, 25%, and 35% missingness.
* They compare 10 methods: mean, Kalman filtering (two variants: “Kalman ST” structural and “Kalman AR” ARIMA), linear/spline/Stineman interpolation, EWMA, SMA, KNN, LOCF.
* Results in short:

  * For small missingness (10–15%), interpolations and some simple methods can perform well.
  * For larger missingness (25–35%), **Kalman ST and Kalman AR become among the best methods** in terms of reconstruction error and downstream ARIMA/LSTM forecasting; EWMA is also strong.
* Key conclusion: Kalman smoothing maintains reasonable autocorrelation structure and supports both classical (ARIMA) and deep learning (LSTM) forecasting after imputation.
* For you: strong experimental evidence that **Kalman/STS imputation is robust when missingness is moderate-to-high**, which is exactly when naive methods break.

#### [7] Hadeed et al., 2020 – air pollution monitoring

* Domain: 1-minute PM2.5 concentrations from short-term indoor monitoring; missingness up to 80%.
* Methods: mean, median, random, Markov, LOCF, Kalman filter, and multivariate methods like predictive mean matching (PMM), row-mean method (RMM), etc. 
* Findings:

  * At 20–40% missingness, **Kalman filters and median** imputation give very low RMSE/MAE and good R².
  * At very high missingness (60–80%), simpler Markov/mean/random methods can outperform Kalman, because there is too little information to exploit fine structure.
  * Multivariate methods using cross-section information (PMM/RMM) perform surprisingly poorly here, partly due to heterogeneity between households.
* For you: illustrates that **Kalman works very well for short/medium gaps**, but for extremely long outages with little surrounding data, no method can do miracles; simple robust baselines may even be safer.

#### [8] Alejo-Sanchez et al., 2025 – climate TS imputation review

* Broad review of climate time-series imputation methods (mainly temperature/precipitation, monitoring networks).
* Classifies methods into:

  * Simple statistical (means, regression, interpolation, PCA).
  * Time-series models (ARIMA, state-space / Kalman, etc.).
  * Machine learning / deep learning (MLP, RNN, GAN, etc.). 
* Main messages:

  * No single method dominates; performance depends on variable, resolution, missingness mechanism, and network structure.
  * **Classical time-series models with Kalman smoothing** remain widely used because they are transparent, relatively easy to tune, and work well when seasonal patterns are strong and long histories exist.
  * Recent work shows promise for GANs and other deep models, but they demand more data, more compute, and come with interpretability issues.
* For you: this gives “state-of-the-art context” and justifies positioning Kalman/STS as a strong, well-established baseline rather than something exotic.

#### [9] Cunha et al., 2022 (uRos) – Stats Portugal e-invoice pipeline

* Use case: monthly **taxable amounts from e-invoice data** for >2000 issuers, with both total and “partial” missingness (gross under-reporting).
* Workflow:

  * Detect potential missing or erroneous values with Isolation Forest.
  * Impute taxable amount **using Kalman smoothing in structural time-series models**, at issuer × acquirer-class level.
  * Implemented in R using `imputeTS` and a fully automated pipeline.
* Shows that structural TS + Kalman smoothing is mature enough to be deployed in **production pipelines for official statistics**, not just in research.
* For you: this is a concrete “industrial-strength” precedent of Kalman/STS being used as the core imputation engine in a continuous data-processing pipeline.

#### [10] Mantuano et al., 2025 – renewable energy resource data

* Domain: **intermittent renewable resources** (e.g., PV irradiance, wind), with high-frequency data and realistic missing patterns.
* Compares several imputation strategies (statistical and machine-learning) and evaluates how they affect downstream **energy system modelling** outputs (costs, capacity decisions, etc.).
* Main relevant points:

  * Different imputation methods can noticeably change model outputs, but for realistic missingness levels (e.g., <30%) multiple reasonable methods give similar planning results.
  * Time-series-aware methods (including Kalman-based models) tend to better preserve distributional properties and variability of the resource, which is crucial for reliability/adequacy studies.
* For you: directly supports the project’s objective of studying the **impact of backup data / imputation on downstream prediction or optimization**.

---

### 3. Overall report: Kalman / structural time-series imputation

#### 3.1. What are structural time-series (STS) models and Kalman smoothing?

A (linear Gaussian) state-space model is:

* Observation equation:
  $$
  y_t = Z_t \alpha_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, H_t)
  $$
* State equation:
  $$
  \alpha_t = T_t \alpha_{t-1} + R_t \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)
  $$

where:

* (y_t) is the observed time series (e.g., PV production, GHI, temperature).
* $\alpha_t$ is the hidden **state vector**: trend, seasonal components, regression effects, etc.
* (Z_t, T_t, R_t, H_t, Q_t) are known system matrices; their unknown hyper-parameters (variances, etc.) are estimated from data.

A **structural time-series model** is a particular state-space model where the state is explicitly decomposed into interpretable components, e.g.:

* Local level or local linear trend.
* Seasonal component (dummy-seasonal or trigonometric).
* Cyclical component.
* Regression terms on exogenous covariates (temperature, calendar indicators, holidays, etc.).

The **Kalman filter** computes the optimal (minimum-MSE) estimate of the state $\alpha_t$ at each time using data up to time (t). The **Kalman smoother** refines these estimates using the whole series (past + future), producing smoothed states and smoothed predictions $\hat{y}_t$.

When some (y_t) are missing, the Kalman filter simply:

* Skips the measurement update for those times (no likelihood contribution, state is just propagated).
* The smoother then uses surrounding observed data and the model structure to reconstruct plausible values at missing timestamps.

This is exactly what tools like `na.kalman` in imputeTS implement, using either a structural model (`StructTS`) or an ARIMA state-space representation. 

#### 3.2. How is imputation performed in practice?

For a given univariate series (y_t):

1. **Model specification**
   Choose a structural specification, for example:

   * Local linear trend.
   * Daily and yearly seasonality (for 10-minute data, e.g. daily seasonality with period 144; weekly/yearly as needed).
   * Optional AR(1) residual to capture extra autocorrelation.

2. **Parameter estimation**

   * Estimate variance parameters (level/trend/seasonal disturbance variances, observation noise) by maximizing the likelihood using the Kalman filter.
   * This is what standard libraries do automatically (statsmodels `UnobservedComponents`, R `StructTS`, `KFAS`).

3. **Kalman filtering and smoothing**

   * Run the filter across the full series, treating missing observations as `NaN` (no update).
   * Run the smoother backward to obtain smoothed state estimates $\hat{\alpha}_t$ for all t.

4. **Imputing missing observations**

   * For each missing (y_t), reconstruct:
     $$
     \hat{y}_t = Z_t \hat{\alpha}_t
     $$
   * Optionally also use the smoothed prediction variance to get uncertainty bounds for the imputed value.

For ARIMA-based Kalman imputation, the logic is the same, but the state vector represents ARIMA lags rather than structural components.

In a **multivariate** setting (e.g., power, GHI, DNI, temperature), one can either:

* Build separate univariate STS models for each series, or
* Use a multivariate state-space model where the state jointly drives several series (more complex but can borrow strength across variables).

#### 3.3. Where does historical / seasonal information enter?

Structural components explicitly encode:

* Long-term trend of consumption/production (e.g., slow growth or degradation).
* Daily seasonality: repeated pattern over 144 10-minute points.
* Weekly/weekend patterns via regression on calendar dummies.
* Temperature sensitivity via regression on current or lagged temperature.

Because of this, when data is missing:

* For **short gaps** (minutes–hours), the smoother uses nearby observations and the local dynamics to fill the gap with very realistic values.
* For **medium gaps** (e.g., a few hours or a day), the model leverages seasonality and the estimated trend to reconstruct plausible shapes, even if no observation lies inside the gap.
* For **long gaps** (days–weeks), reconstruction becomes more uncertain; the model heavily relies on seasonal averages and the prior dynamics. Peaks and rare events inside the gap are likely to be smoothed away.

This behavior is consistent with the empirical results of Afrifa-Yamoah (hourly weather) and Niako (24-hour blood pressure), where Kalman/STS performs well up to moderate missingness but cannot fully recover fine-scale structure for very long gaps.

#### 3.4. Benefits in the scenario (continuous energy forecasting with missing data)

From the literature above, plus the textbooks:

1. **Model-based and structure-aware**

   * Uses explicit trend + seasonality + regression on external factors, which matches the energy/irradiance context much better than generic interpolation.

2. **Native handling of missing data**

   * The Kalman filter does not break when data is missing; it naturally propagates states without observations.
   * This allows the daily midnight forecast pipeline to continue even if some sensors fail during the input window.

3. **Uncertainty quantification**

   * Kalman smoothing provides variance of imputed values, which you can propagate into the prediction uncertainty or use to flag “low-confidence” forecasts.

4. **Compatibility with downstream models**

   * Niako et al. demonstrate that Kalman-based imputations yield good forecasting performance for both ARIMA and LSTM, especially at higher missingness. 
   * This is exactly the case: an NN model that expects complete windows.

5. **Operational robustness and automation**

   * imputeTS + Structural TS + Kalman smoothing are already used in production pipelines (e.g., Stats Portugal’s e-invoice system). 
   * This gives you a strong argument that such methods are reliable for continuous, automated data processing.

6. **Interpretability and control**

   * Structural components are directly interpretable (e.g. “level,” “daily seasonality,” “holiday effect”), making it easier to explain and debug imputation behavior to non-ML stakeholders.

#### 3.5. Limitations and caveats

1. **Model mis-specification**

   * If trend/seasonality are mis-specified (wrong periods, missing change-points), imputations can be biased.
   * Structural breaks (e.g. a sudden change of production regime) must be handled explicitly (change-point modelling or re-estimation after the break).

2. **Distributional assumptions**

   * Standard Kalman methods assume Gaussian errors. Heavy tails, spikes, or constraints (e.g., non-negative power) are not perfectly captured.
   * Some robustness can be introduced (e.g., heavy-tailed errors, outlier modelling), but this complicates estimation.

3. **Long gaps and information loss**

   * For very long outages, Kalman smoothing essentially interpolates using the model’s average behavior; it cannot hallucinate unobserved peaks that have no analog in the observed data.
   * Hadeed et al. show that at very high missingness (60–80%), simpler methods (mean/Markov) can rival or beat Kalman; in such regimes, you should treat reconstructed data as low-confidence “filler,” not as ground truth. 

4. **Missingness mechanism**

   * Most implementations assume missing completely at random (MCAR) or MAR; in reality, sensor failures may correlate with extreme conditions.
   * This can bias imputations (e.g., systematically under-estimating rare high-production or high-load events that coincided with outages).

5. **Computational cost**

   * For extremely large series (millions of points), naïve structural models can be slow; imputeTS notes that `na.kalman` may take hours or days on a 600k-point series with complex structure on normal hardware. 
   * However, the case (5 years × 144 points/day ~= 262k points per series) is very manageable with modern hardware.

#### 3.6. Practical methods and variants

For the project, the most relevant variants are:

1. **Univariate STS + Kalman smoothing**

   * For each series (e.g. GHI, DNI, temperature, power), fit a local linear trend + daily seasonality model.
   * Use the Kalman smoother to impute missing points; simple to implement with `statsmodels.tsa.statespace.UnobservedComponents` (Python) or `StructTS` / `KFAS` (R).

2. **ARIMA state-space + Kalman smoothing**

   * For series with strong autocorrelation but less clear physical structure, auto-ARIMA (e.g., `auto.arima` in R) can be put into state-space form.
   * imputeTS uses this approach via `na.kalman(..., model = "auto.arima")`. 

3. **Structural regression models (dynamic regression)**
 
   * Include exogenous regressors in the observation equation, e.g.:
     $$
     y_t = \beta^\top x_t + \text{(trend + seasonal + noise)}
     $$
   * Here (x_t) can be temperature, calendar indicators, other operational variables.
   * This is particularly appropriate for the consumption/production signals.

4. **Two-stage approach with covariates**

   * First, impute missing exogenous variables (temperature, GHI, DNI) using their own STS models.
   * Then, fit a structural regression model for production/consumption using fully imputed exogenous inputs.

5. **Online vs offline imputation**

   * **Online**: during the day, run the Kalman filter and use one-step-ahead predictions to temporarily fill missing values (for real-time control).
   * **Offline (end-of-day)**: run the full smoother (using that day’s data plus historical data) to re-impute gaps more accurately; keep both on-line and smoothed versions for analysis.

#### 3.7. Evaluation strategies (data quality and impact on forecasting)

The literature suggests a few evaluation ideas you can reuse:

1. **Artificial missingness experiments**

   * Randomly remove 10%, 25%, 35% of points under various patterns:

     * Isolated points.
     * Short blocks (e.g. 30–60 minutes).
     * Long outages (hours or days).
   * Compare Kalman/STS imputation to baselines (mean, LOCF, interpolation, KNN, EWMA, ML methods) on MAE, RMSE, sMAPE, and correlation (as in Afrifa-Yamoah, Niako, Hadeed).

2. **Effect on autocorrelation and seasonality**

   * Examine how each imputation method changes the series’ ACF, seasonal patterns, and distribution (Niako explicitly studies this).

3. **Effect on downstream models**

   * Train the forecasting model (NN or otherwise) on:

     * Original complete data (where you have it).
     * Same data with artificially introduced missingness + imputation.
   * Compare forecasting metrics to quantify how robust the model is to different imputation schemes, similar in spirit to Niako and Mantuano.

4. **Operational metrics**

   * In a continuous system, also track:

     * Fraction of imputed points in each input window.
     * Fraction of imputed points in “critical” periods (peaks, ramping periods).
     * How prediction error correlates with these fractions.

---

### 4. What you can say orally (very short summary)

If you need a quick spoken summary tomorrow, you can compress it into something like:

* We focus on **structural time-series models in state-space form**, estimated with the **Kalman filter and smoother**. They decompose the series into interpretable components (trend, seasonality, regression on temperature and calendar effects).
* Missing observations are handled natively: the filter skips the update step, and the smoother reconstructs optimal estimates using past and future data. Imputed values are then obtained from the smoothed states.
* Empirical studies on **high-resolution climate data**, **sensor data**, **medical time series**, **air quality**, and **renewable energy** show that:

  * For realistic levels of missingness (10–35%) and structured series, **Kalman/STS imputation is among the best methods**, often outperforming or matching simpler interpolations and being robust for downstream forecasting (ARIMA, LSTM).
  * For extremely long gaps or very high missingness, no method can perfectly recover the signal; Kalman behaves sensibly but tends to smooth out peaks, and simple baselines can sometimes perform similarly.
* These methods are already used in production pipelines (e.g. official statistics for e-invoice data), which supports using them in our continuous prediction system as a **principled, interpretable, and robust imputation strategy**.