In economic literature, the analysis of regime in time series data focuses on the business cycle. Hamilton[Hamilton, 1989] utilized the business cycle dates from the NBER when constructing a regime-switching model. Rather than using a single variable to distinguish the economic cycles, it is more common to differentiate the cycles based on two variables: the business cycle and inflation.

The regime cycle, which incorporates both the business cycle and inflation variables, is based on the directionality of each indicator. For example, the Expansion regime refers to a period when the economy is expanding and inflation is rising. The summarized information can be presented in the following table:

<p align ='center'>
  
|Regime |Business|Inflation|
|------|---|---|
|Recovery|+|-|
|Expansion|+|+|
|Slowdown|-|+|
|Contraction|-|-|
  
</p>

To determine the regimes, two indicators were used.[Uysal and Mulvey, 2021 and Kim, 2022] The business indicator utilized the fluctuations in the KOSPI index, while the inflation indicator was based on the monthly consumer price inflation rate. The z-score of the inflation rate, calculated based on the past four years, was employed.

The indicator using the stock market is defined by the L1-trend-filtering algorithm[Mulvey and Liu, 2016]. L1-trend-filtering is a nonparametric unsupervised learning method used to identify trends in time series data[Burder et. al, 2011]. The given return time series data, denoted as $r_t(t = 1,\dots,n)$, consists of a slowly changing trend component $x_t$ and a more rapidly changing random component $z_t$ ($z_t = r_t - x_t$). This leads to an optimization problem with two competing objectives. The random component $z_t$ needs to be minimized to identify the trend, while $x_t$ should be smooth. This problem can be formulated as follows:


$$
\min_{x\in \R^n}\frac{1}{2}\sum_{t=1}^n(r_t -x_t)^2 + \lambda \sum_{t=2}^n \vert x_{t-1} - x_t \vert  \quad (\lambda \geq 0),
$$

where $\lambda$ represents a parameter that controls the balance between smoothness and residuals. To determine the appropriate value for this parameter, various $\lambda$ values were used to visualize the graph, and the results are as follows:

<p align="center">
<img src = 'https://github.com/hynacin121/IE471_TermProject/blob/2f2d538b724a81fe4bc858260fd5af6fee55650c/data/l1_lambda.png' >
</p>

Setting $\lambda$ too small exposes the result to noise, while setting it too large may miss the points of regime change. Therefore, in this study, $\lambda = 1$ is utilized.

Based on the previously explained KOSPI index and consumer price index, the regimes were labeled monthly, and the results are as follows:


<p align ='center'>
  
|Date|inflation|stock|	regime|
|------|---|---|---|
|2004-01|0.2722|1.0|	2
|2004-02|0.1319|1.0|	2
|2004-03|-0.1283|-1.0|	4
|2004-04|0.0773|-1.0|	1
|2004-05|0.0331|-1.0|	1
  
</p>


# REFERENCES


Bruder, B., Dao, T. L., Richard, J. C., & Roncalli, T. (2011). Trend filtering methods for momentum strategies. *SSRN*

Hamilton, J. D. 1989. “A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.” *Econometrica*

Kim D. Y. (2022). 경기 국면에 따른 리스크 팩터 배분전략. *삼성증권*

Kim, S. J., Koh, K., Boyd, S., & Gorinevsky, D. (2009). $\ell_1$-trend filtering. *SIAM review*

Mulvey, J. M., and H. Liu. 2016. “Identifying Economic Regimes: Reducing Downside Risks for University Endowments and Foundations.” *The Journal of Portfolio Management*

Uysal, A. S., & Mulvey, J. M. (2021). A machine learning approach in regime-switching risk parity portfolios. *The Journal of Financial Data Science*.
