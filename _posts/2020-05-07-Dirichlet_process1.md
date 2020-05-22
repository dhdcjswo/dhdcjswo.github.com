---

title:  "블로그 첫 글" 

date: 2018-11-30 
use_math: true
categories: ['일반'] 

tags: ['블로그', 'jekyll', 'github', 'github.io'] 

---





## Dirichlet Process 

non-parametric 베이지안 추론에서 가장 유명한(?) ,주로 쓰이는 **'Dirichlet Process(DP)'**  에 대해서 알아보자. 먼저 사전에 알아두어야할 내용들에 대해 간략하게 이야기하고 DP를 이야기 하려고 한다.



**non-parametric**

non-parametric의 방법이라고 하면 모수가 없는 통계 방법 또는 머신러닝 방법등으로 흔히 알려져있다. 하지만 베이지안의 관점에서 non-parametric은 모수가 없다는 뜻이 아니라 모수의 개수가 무한히 늘어나 특정지을수 없다고 생각하는 것이 올바른 표현이다. 예를들면 *Gaussian process* 처럼 시간/공간 에 대해서 확률변수가 계속해서 확장되어가는 형태도 마찬가지로 non-parametric의 방법중 하나라고 할 수 있다. 하지만 조금더 정확하게 'DP'를 정의하자면 모수가 무한개를 가진다는 의미가 아니라, 모수 자체는 discrete하게 정해져 있는데 그 개수가 계속해서 변화 한다는 뜻이다. 오늘의 모수 갯수, 내일의 모수 갯수가 계속해서 변화한다고 생각하면 된다.그리고 DP는 non-parametric한 방법이다



**Dirichlet distribution**

*Dirichlet distribution* 은 multinomial distribution의 conjugate형태의 분포로, 선택지의 갯수가 k개인 문제의 상황을 효과적으로 다루는 continuous 함수이다. 사실 DP의 형태는 Dirichlet으로 표현이 가능하다.



**measure**

확률론에서 다루는 measure의 개념





number of latent group

나타나는 클러스터



## motivation

간단한 밀도 추정 문제를 생각해보자. 학생들의 키에 대해서 모델링을 하려고한다. **베이지안**의 방법을 적용하여 키라는 확률변수 $X_i$ 에 대한 분포를 알아내는게 우리의 목적이다. 간단한 접근은 변수 $X_i$를 정규분포로 가정하고 이는 두개의 모수 $\mu, \sigma^2$ 에 의해서 실현된다고 생각하는 것이다.

그런데 단순 정규분포를 가정해서 이 문제를 해결하는데는 약간의 문제가 있다. 학생들의 키의 밀도를 추정하는 지금의 문제에서 남/여 라는 sub-population은 엄연히 다른 평균과 분산을 가지는데 단일 분포로 간주하게 되면 분포의 형태에서 skewness가 발생해버린다.

![1](img/DP1_1.PNG)



단일 분포가 아니라 남/여 각각의 정규분포의 mixture 하는 것이 더 정확한 추론이 될것이다. 남/여 라는 클러스터에 따라서 분포는 구분되어 추정되어야 한다.

문제는 이러한 카테고리가 성별만 있는게 아니다. 나이가 될수도 있고 인종이 될수도 있다. 수많은 sub-population이 존재할 수있다. 남/녀의 경우는 선택의 경우가 두가지이다. 남자이거나 여자. 그렇다면 해당 카테고리에 속할 확률에 대한 추정은 베타분포를 이용할 수 있다. 하지만 인종이나 나이처럼 카테고리를 두가지로 표현할 수 없다. 이런 경우에는 베타분포의 일반화된 형태인 디리클레(dirichlet) 분포를 사용할수 있다.

명시적으로 드러나 있는 정보가 아니라 latent하게 눈에 보이지 않는 정보도 분명 존재한다. 영화에 대한 관객들의 선호에 따른 클러스터를 나누는 것은 명시적으로 알 수 없는 문제이다. 몇개의 클러스터가 존재하는지 알 수 없다. 결국 문제는 몇개의 클러스터를 정하느냐이다. 어떤 기준에 의해서 클러스터를 정할 때, 너무 많이 정하면 train데이터에 오버피팅될것이고 `너무 카테고리가 작다면 주어진 정보를 잘 이용하지 못하고 정확한 추론을 못하게 된다.



클러스터가 두개인 모델을 먼저 생각해보자.

* $\Omega$ : 모수들의 집합 (위의 예제에서는 평균과 분산, $\Omega = R\times R^+$)
* $F_{\Omega}$ : 사건의 집합. sigma-algebra



키를 추정하는 문제에서 A가 모수들의 집합이고 $A \in F_{\Omega}$ 라면,  A라는 모수들의 집합으로 발생할 수 있는 사건의 공간은 아래와 같을 것이다.![2](img/DP1_2.PNG)

이 때, $\delta_{\theta}$ 를 어떤 사건들의 집합을 실수인 0 또는1로 mapping 시켜주는 'measure'라고 한다면 그 정의는 아래와 같다.
$$
\delta_{\theta}(A) = \begin{cases} 1 & if \space \theta \in A \\
0 & o.W. \end{cases}
$$


모수의 집합인 $\theta$ 가 A라는 공간에 속하면 1, 그렇지 않으면 0으로 mapping시켜준다.

자 이제는 또다른 measure 인 G를 정의해보려고 한다. G를 다음과 같이 생각해보자


$$
\begin{matrix}
G &=& \pi \delta_{\theta_1} + (1-\pi)\delta_{\theta_2} \\
&=& weight_1 \times measure(custer_1) + weight_2 \times measure(cluster_2)
\end{matrix}
$$


우리는 처음에 두개의 클러스터의 존재를 간주했다. 위처럼 A라는 단일 분포의 measure가 아니라 개별 분포의 measure를 합한 새로운 measure G 이다. **가중치는 상황에 따라 달라질수 있고** measure G 역시도 고정된 값이 아닌 가변적이다. 그래서 random-distribution이다.  G가 실현화되면 다음과 같아진다.

![3](img/DP1_3.PNG)

그림에서 위쪽의 클러스터가 좀더 높은 가중치를, 아래쪽 클러스터가 낮은 가중치를 가진다. G는 discrete 밀도와 같다. 





![4](img/DP1_4.PNG)

만약 클러스터의 개수가 아주 많다면 위의 그림처럼 G가 실현화 될것. G가 확률변수의 개념. ,G~DP.

A공간에서 실현되는 클러스터가 있고 다른곳에서 실현되는 클러스터가 있다. DP는 수많은 measure $\delta$ 의 분포의 형태이다. 위 그림과 같이 확률공간에서의 분포. 



### formal definition

DP는 두개의 모수를 가짐. 

* $\alpha$ precision parameter
* 분포 $G_0$ : $F_{\Omega} \rightarrow[0,1]$  를 base measure. .mean parameter 

DP에서 marginals 은 디리클리 분포임.

$G \sim DP(\alpha_0, G_0)$ . $\Omega$ 의 모든 measureable 한 partition을 $(A_1,...,A_K)$ 라 한다면 (A는 disjoint이고 union은 $\Omega$) 
$$
(G(A_1), G(A_2), ..., G(A_K)) \sim Dir(\alpha_0G_0(A_0),\alpha_0G_0(A_1),...,\alpha_0G_0(A_K))
$$


$G(A):F_{\Omega'} \rightarrow [0,1]$ 는 fixed set의 random measure이다. 





$$
G = \sum_{k=1}^{\infin}\pi_k\delta_{\theta(k)}
$$



Stochastic processes are distributions over function spaces

 In the case of the DP, it is a distribution over probability measures, which are functions with certain special properties which allow them to be interpreted as distributions over some probability space Θ. Thus draws from a DP can be interpreted as random distributions. For a distribution over probability measures to be a DP, its marginal distributions have to take on a specific form which we shall give below. We assume that the user is familiar with a modicum of measure theory and Dirichlet distributions





## motivation2

존재하는 종은 몇가지일까?  계속해서 새로운 종을 발견/구분.

소셜 네트워크. 커질수록 더많은 그룹

데이터를 수집할수록 파라미터가 커짐

gaussian mixture model이 있음.클러스터링(k=2) generative model 이라면,

$P(parameter|data) \propto P(data|parameter)P(parameter)$
$$
\begin{matrix}
&\mu_k& \sim N(\mu_0, \Sigma_0) \\
&\rho_1& \sim Beta(a_1,a_2) \\
&\rho_2& \sim 1 - \rho_1 \\
&z_n& \sim Categorical(\rho_1, \rho_2) \\
&x_n& \sim N(\mu_{z_n}, \Sigma)
\end{matrix}
$$
위와 같이 데이터들이 만들어짐

beta분포의 특징.파라미터가 0에 가까워지면 또는 그리고 점저커지면

beta(1,1)은 uniform임 

beta(0.01, 0.01) 은 양 극단 0 또는 1의 밀도가 높은 분포임.  0 또는 1인  극단적인 rvs 밀도가됨.

beta(1000,1000) 은 두 카테고리의 빈도가 비슷. 0.5 근방.

beta(1,5) 는 한쪽 확률이 높음.

dirichlet는 beta의 확장임.

dirichlet(1,1,1,1) 은 uniform임. 4개의 선택지가 랜덤하게 나올것.

마찬가지..



만약 클러스터의 개수가 데이터 포인트 보다 많다면..? 해당클러스터의 속할 확률들의 합은 1

실제 component는 1000개인데 데이터로 실현화된 것이 100개 라면..? 먼저 0 부터 1의 범위를 1000개로 나눔. 랜덤넘버 하나를 생성하면 하나의 클러스터에 속할것. 반복해서 생성. 계속해서 던지지만 좀처럼 모든 클러스터를 색칠하지 못할듯.

y : 클러스터, x : 샘플 index .y축은 실현화된 클러스터의 개수정보를 제공해줌.

실현화되는 클러스터의 개수는 랜덤할듯. 데이터포인트가 늘수록 클러스터는 증가할것.

클러스터K를 고를수 없음. k는 inf. 

각 클러스터의 확률~dirichlet(a) 아니면 $\rho_1 = beta(a_1, \sum a_k - a_1)$.

stick breaking : 네개의 클러스터라면 $v_1 \sim Beta(a_1, a_2+a_3+a_4)$ , v1 $v_2 \sim Beta(a_2, a_3+a_4)$  (1-v1)v2 ...

beta(a_1, b_1) , beta(a_2, b_2) ... beta(a_k, b_k)

DP stick breaking은 a_k =1, b_k= alpha 이다.

rho ~ GEM(alpha). alpha가 커지면 촘촘하게 stick breaking 되어감. GEM(1)은 큼직. gem(10)은 빽빽.

옵션이 두개면 beta, 3개이상이면 dirichlet, 무한하면 GEM

$\mu_k \sim N(\mu_0, \Sigma_0), k=1,2,..$

$G = \sum \rho_k \delta_{\mu_k} = DP(\alpha, N(\mu_0,\Sigma_0))$

$\mu_n^* \sim G$

$x_n \sim N(\mu^*_n,\Sigma)$

 uniform random draw



A라는 공간을 고정시켰을 떄 G(A)가 random. 고정된 A에 대해서 G(A)가 r.v



옵션이 m개인 dirichlet dist에서 샘플링을 하면 m-1차원의 simplex 공간에 놓이게 된다. 뽑힌 샘플은 m차원의 분포의 형태를 띔. theta는 확률''변수''가 아니라 확률''분포''.  DP는 distribtion over distribution. 

DP의 확률분포 G는 base distribution과 같은 support.

base_dist 는 연속형. 두개의 샘플이 같을확률은 zero. 그러나 G는 discrete. infinite한 countable 포인트롤 구성됨.

alpha는 G가 base-dist에 얼마나 가까울지를 통제.



잘모르는분포가 있고 주어진 데이터들로 추론하고 싶음. n개의 데이터는 우리가 잘 모르는 F라는 분포에서 나왔다 생각. 베이지안이라면 F에 대해서 prior을 가정하고 주어진데이터로 posterior을 구함. 전통적으로 prior은 모수의 familiy로 정했었음. 그러나 어떤 모수의 familiy로만 좁혀서 이야기를 하면 추론에 제약이 생김. non-parametric한 접근은 분포에 대한 prior을 set함.일반적으로 stochastic process는 함수에 대한 분포. DP는 prob-measure에 대한 분포.random distribution. 

파라미터 벡터에 대한 분포가 dirichlet-dist. 밀도합이 1인 dirichlet에서 샘플링을 하게되면 각 option별로의 밀도의 확률이 샘플링되는것.







**motivation**

Probabilistic models are used throughout machine learning to model distributions over observed data. Traditional parametric models using a fixed and finite number of parameters can suffer from over- or under-fitting of data when there is a misfit between the complexity of the model (often expressed in terms of the number of parameters) and the amount of data available. As a result, model selection, or the choice of a model with the right complexity, is often an important issue in parametric modeling. Unfortunately, model selection is an operation that is fraught with difficulties, whether we use cross validation or marginal probabilities as the basis for selection. The Bayesian nonparametric approach is an alternative to parametric modeling and selection. By using a model with an unbounded complexity, underfitting is mitigated, while the Bayesian approach of computing or approximating the full posterior over parameters mitigates overfitting. A general overview of Bayesian nonparametric modeling can be found under its entry in the encyclopedia

 Traditionally, this prior over distributions is given by a parametric family. But constraining distributions to lie within parametric families limits the scope and type of inferences that can be made. The nonparametric approach instead uses a prior over distributions with wide support, typically the support being the space of all distributions. Given such a large space over which we make our inferences, it is important that posterior computations are tractable. The Dirichlet process is currently one of t

문제의 시작: 식물의 종을 탐구하는 연구자 인 '나'는 종의 종류를 분류하고 싶어한다. 지금까지 수많은 관측치를 통해서 k개의 종으로 분류할 수 있었다. 그런데 이번에 발견한 식물은 전에 본적이 없는 새로운 식물이고 이 식물때문에 기존의 분류했던 종의 구분 k의 변화가 필요하게 되었다. 관측치가 계속 쌓이면서 못봤던 식물들이 관측이 되고 그에 따른 종의 구분을 변화시켜야 한다.



일반적으로 DP는 클러스터링 방법 중에서 분석가가 클러스터의 갯수를 미리 정해주지 않아도 되는 unsupervised learning의 방법과 유사하다.  KNN 또는 DBSCAN이 마찬가지로 'k'를 특정하지 않아도 되는 클러스터링 방법이다. 하지만 이들과 다른점은 위의 문제상황에서 보았듯이 관측치가 증가함에 따라서 클러스터의 개수의 변화를 찾을 수 있다는 점이다.



어떤 랜덤확률분포 G 가 DP를 따른다고 할때 다음과 같이 쓸 수 있다.
$$
(G(A_1), .., G(A_r))|\alpha,H \sim Dir(\alpha H(A_1),...,\alpha H(A_r)) = DP(\alpha,H)
$$

* $\alpha$ : positive scaling parameter
* H : base distribution

DP는 확률에 대한 measure의 분포이다. 또 'distribution over distribution' 이라고 일컬어진다. 상당히 잘 와닿지 않는데 이게 무슨 의미이냐면 일반적으로 stochastc process라 하면 어떤 주어진 분포에서 샘플링 된 확률변수들의 값으로 이루어지는데, DP는 어떤 분포에서 분포를 뽑는것. random variable이 아니라 random distribution이라고 불림.

유한차원의 분포들의 집합. 어떤 확률공간 오메가 내의 A라는 공간을 G라는 measure로 측정. 서로 다른 A(벡터) 공간들의 평균은 G0(A)



**DP : inifinte dimensional generalization Dirichlet**
$$
\begin{matrix}
\pi|\alpha &\sim& Dir(\frac {\alpha} {K},..,\frac {\alpha} {K})  \qquad \theta_k^*|H \sim H \\
z_i|\pi &\sim& Mult(\pi) \qquad x_i|z_i,\{\theta_k^*\} \sim F(\theta_{z_i}^*) \\
\end{matrix}
$$





**DP**

random-dist G는 DP에 의해 분포되있음. marginal-dist는 dirichlet dist. H를 omega 에 대한 dist, A는 omega의 partitions.  base dist H는 DP의 평균을 의미함. 어떠한 measureable한 set에 대해서 E[G(A)] = H(A) . alpha는 inverse variance로 이해할수 있음. V[G(A)] = H(A)(1-H(A))/(alpha+1). alpha가 커질수록 분산이 작아짐. 그러면 DP는 평균 근처로 집중됨. 그래서 alpha가 concentration param이라고 불림. DP의 정의에서 alpha ,H는 곱으로 정의됨. 만약 alpha가 무한대로 가면, G(A) -> H(A)로 수렴. 그렇지만 G->H는 아님. DP는 확률 1에 대한 discrete distribution임. 그래서 g,H가 완전히 같아질수는 없음.

 It is a distribution over distributions, i.e. each draw from a Dirichlet process is itself a distribution. It is called a Dirichlet process because it has Dirichlet distributed finite dimensional marginal distributions, just as the Gaussian process, another popular stochastic process used for Bayesian nonparametric regression, has Gaussian distributed finite dimensional marginal distributions









The original definition of the DP is due to Ferguson (1973), who considered a probability space (Θ, A, G) and an arbitrary partition {A1,...,Ak} of Θ. A random distribution G is said to follow a Dirichlet process prior with baseline probability measure G0 and mass parameter M, denoted G ∼ DP(M,G0)





* distribution of distribution
* 파라미터 : alpha, G
* concentration param
* base distribution
*  초기 G라는 분포를 블랙박스에 넣으면 G' 가나옴
* alpha는 얼마나 두 분포가 유사한지를 나타냄
* break off sticks : v1,v2,... ~ Beta(1,alpha). 0<alpha<1
* 긴스틱을 가지고 있는데 얼마나의 길이로 자를건지. 남은 거에서 계속 해나감*
* draw atoms
* Edp = Eg



alpha가 작으면 sparse 크면 초기분포랑 비슷

연속형 분포를 이산형분포의형태로 복사하는것

DP mixture model

* 

Chinese restaurant process

중식당을 갔는데 테이블이 3개가 있고 각각 자리가 2, 3, 2개의 자리가 남아있음. 각 테이블에 앉을 확률은 2/7,3/7,2/7 을 평균을 가지는 정규분포일것같음. 근데 자리를 앉으니깐 또다른 테이블이 있는걸 발견한거임. 그래서 확률의 변화가 필요한상황



* 클러스터링에 대해서 공부해야함

example1

다가오는 선거에서 어떻게 투표할지. 합리적인 방법은 사람들의 정치성향을 분류해서 베르누이 문제로 어떤 문제에 대답하는 걸 확률로 나타내는것. 그 확률로 어느 클러스터에 속할지 보는것인가.? k-means 쓰거나. 하지만 미리 클러스터의 갯수를 알아야함.일반적인경우는 모름. 또 우리가 정한 분류가 잘못될수도 있음.정치성향이 아니라 인종, 종교 등이 투표에 영향을 끼칠수도 있음.

example2

은하의 속도에 대해서, 클러스터라고 가정함. 예를들면 i번쨰 관측치는 k클러스터에서 정규분포를 가지는 것임.클러스터마다 평균이 다르지만 분산은 동일함. 몇개의 클러스터가 있는지 몰라서 사전분포를 정의하기가 난감. 클러스터의 분포에 대해서 dirichlet process는 .

나이브모델은 K개의 클러스터가 있다고 미리 정하는것임.

처음에 데이터가 어떤 클러스터에 속해있다고 생각하고 그후 속한 클러스터으 분포에 의해 결정된다고 가정하는거 대신에, 각 관측치는 파라미터 $\mu$ 와 연관이 되어있고 이는 G 분포에서 나왔으며 K 평균을 support.



non-parametric의 의미는?

* 전통적인 의미 : 파라미터가 없는거
* 베이지안에서의 의미 : 무한한 숫자의 파라미터
* 더정확하게 : finite하지만 파라미터의 수가 변함 계속

얼마나 많은 음악장르가 있을까? 계속 변함 증가함. non-parametric

파라미터 벡터 

"tutorial on dirichlet processes and hierachical dirichlet processes"

prior형태
$$
(G(A_1), .., G(A_r))|\alpha,H \sim Dir(\alpha H(A_1),...,\alpha H(A_r)) = DP(\alpha,H)
$$
posterior 형태
$$
G|\theta_1...\theta_n,\alpha,H \sim DP(\alpha+n,\frac {\alpha} {\alpha+n}H + \frac {n} {\alpha+n}\frac {\sum^n_{i=1}\delta_{\theta_i}} {n})
$$






H라는 베이스확률분포에 따라

오메가라는 확률공간에서 A는 몇개인거지? 무한대로 증가할수도

DP는 density에 관한 내용임.



## mixture model 의 한계

종의 개수는 3개? 100개? 100000개? 

만약 내가 N개의 동물종을 봐싸면 K_N 개의 선택지가 생기는것.만약 다음번에 내가 만나는 동물이 내가 전에 한번도 본적이 없는 동물인확률은..?
$$
p(novel specise|n,\alpha) = (K-K_N) \times \frac {\alpha/K} {\alpha + N}
$$
 vanilla gibbs sampling



## DP

measure space : ($\theta$,$\sum$)

measureable finite partitionin $\theta$ $A_1, A_2,.. A_k$

* Finite : K< $\infin$

* measureable: $A_k \in \sum$
* disjoint : $A_j  A_k$



DP는 random measure  G

prior를 조작하면서 k 를 정하게됨 . DP는 prior임.



generation schem - 무한한 클러스터가 존재한다는걸 증명하기위한?..?

* stick breaking scheme
* chinese restaurant process schem



무한대의 차원에서 샘플링하고싶다.

DP에서 샘플링을 어떻게하는걸까? 무한차원의 dir dist 구성을 찾아보자

**stick breaking construction**

pmf 함수, 무한선택지.

k=1,2,3,...

v_k | alpha ~ beta(1,alpha) : k번재 선택지의 거리.? 사이즈?

베타분포에서 샘플링한값을 그냥 쓰면 확률값 1을 넘어갈수도 있음

스틱의 비율만큼 짤라나감. 근데 남는스틱도 생김.broken stick = atom.

베타분포의 모수에 따라 스틱의자르는 양상이 달라짐.

**polya urn scheme**

$\theta_n|\theta_1...\theta_{n-1}, \alpha,H \sim DP(\alpha+n-1,\frac {\alpha} {\alpha+n-1}H + \frac {n-1} {\alpha+n-1}\frac {\sum^{n-1}_{i=1}\delta_{\theta_i}} {n-1}$

theta는 샘플링. 그떄그떄 다를수 있음. 데이터 포인트 마다의 선택?

**chinese restaurant process**





## reference

https://en.wikipedia.org/wiki/Dirichlet_process

https://www.youtube.com/watch?v=I7bgrZjoRhM

https://www.stat.ubc.ca/~bouchard/courses/stat547-sp2011/notes-part2.pdf