경제학 문헌에서 시계열 데이터의 경기 순환 분석은 business cycle에 중점을 둔다. Hamilton은 국면 전환모델을 구성할떄 NBER의 business cycle dates를 활용하였다(Hamilton 1989). 경제 사이클을 구분하는 기준으로 business cycle 한개를 사용하기보단, business cycle과 물가라는 2개의 변수를 기준으로 경제 사이클을 구별하는 경우가 더 많다. 

경기 변수에 물가 변수를 추가해 만든 국면 사이클은 각 지표의 방향성을 기준으로 한다.  예를 들어 Expansion 국면은 경기가 상승하고 물가는 하락하는 구간을 말한다. 이 내용을 표로 정리하면 다음과 같다.

<center>

|Regime |Business|Inflation|
|------|---|---|
|Recovery|+|-|
|Expansion|+|+|
|Slowdown|-|+|
|Contraction|-|-|

</center>


국면을 판단하기 위해 2가지의 지표를 사용하였다. 경기지표는 KOSPI지수의 등락을 활용하였고 물가지표는 월간 소비자물가 상승률 수치를 기반하여, 해당 수치의 과거 4년 기준 z-score를 활용하였다. 

Stock market을 활용한 지표는 L1-trend-filtering algorithm(Mulvey and Liu (2016) and)으로 정의한다. L1-trend-filtering은 시계열 데이터의 추세를 식별하기 위한 nonparametric unsupervised learning method이다. 주어진 수익률 시계열 데이터 $r_t(t = 1,\dots,n)$ 가 천천히 변화하는 추세 $x_t$와 더 빠르게 변화하는 랜덤 component $z_t$로 이루어져있다($z_t = r_t - x_t$). 이로 인해 2개의 competing obejective가 있는 최적화 문제가 발생한다. 추세를 확인하기 위해서 랜덤 component $z_t$를 최소화 해야하고 $x_t$는 smooth해야한다. 이 문제는 다음과 같이 공식화 될 수있다.
$$
\min_{x\in \R^n}\frac{1}{2}\sum_{t=1}^n(r_t -x_t)^2 + \lambda \sum_{t=2}^n \vert x_{t-1} - x_t \vert  \quad (\lambda \geq 0)
$$

$\lambda$는 smooth와 잔차의 균형을 조절하는 파라미터를 의미한다. 파라미터를 결정하기 위해 다양한 $\lambda$ 값을 사용하여 그래프를 시각화하였고 결과는 다음과 같다. 


<p align="center">
<img  src = 'https://github.com/hynacin121/IE471_TermProject/blob/2f2d538b724a81fe4bc858260fd5af6fee55650c/data/l1_lambda.png' >
</p>

$\lambda$를 너무 작게 설정하면 노이즈에 강하게 노출되고 $\lambda$를 너무 크게 설정하면 Regime이 변하는 포인트를 놓치게 된다. 따라서 본 연구에서는 $\lambda = 1$을 활용하고자 한다. 

앞서 설명한 코스피지수와 소비자물가지수를 바탕으로 매월 국면을 라벨링하였고 그 결과는 다음과 같다.

<center>

|Date|inflation|stock|	regime|
|------|---|---|---|
|2004-01-01|0.2722|1.0|	2
|2004-02-01|0.1319|1.0|	2
|2004-03-01|-0.1283|-1.0|	4
|2004-04-01|0.0773|-1.0|	1
|2004-05-01|0.0331|-1.0|	1

</center>



