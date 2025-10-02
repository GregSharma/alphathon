\title{
A model-free approach to multivariate option pricing
}

\author{
Carole Bernard ${ }^{\mathbf{1 , 2}}$ (D) • Oleg Bondarenko ${ }^{\mathbf{3}} \cdot$ Steven Vanduffel ${ }^{\mathbf{2}}$
}

Accepted: 5 October 2020 / Published online: 27 October 2020
© Springer Science+Business Media, LLC, part of Springer Nature 2020

\begin{abstract}
We propose a novel model-free approach to extract a joint multivariate distribution, which is consistent with options written on individual stocks as well as on various available indices. To do so, we first use the market prices of traded options to infer the risk-neutral marginal distributions for the stocks and the linear combinations given by the indices and then apply a new combinatorial algorithm to find a compatible joint distribution. Armed with the joint distribution, we can price general path-independent multivariate options.
\end{abstract}

Keywords Multivariate option pricing • Rearrangement algorithm • Risk-neutral joint distribution $\cdot$ Option-implied dependence $\cdot$ Entropy $\cdot$ Model uncertainty

Mathematics Subject Classification C63 • C65 • G13

\footnotetext{
The authors gratefully acknowledge funding from the Canadian Derivatives Institute (formerly called IFSID) and the Global Risk Institute (GRI) for the related project"Inferring tail correlations from options prices".

Carole Bernard
carole.bernard@grenoble-em.com
Oleg Bondarenko
olegb@uic.edu
Steven Vanduffel
steven.vanduffel@vub.ac.be
1 Department of Accounting, Law and Finance, Grenoble Ecole de Management (GEM), Grenoble, France
2 Department of Economics and Political Sciences, Vrije Universiteit Brussel (VUB), Ixelles, Belgium
3 Department of Finance, University of Illinois at Chicago (UIC), Chicago, USA
}

\section*{Introduction}

In this paper, we propose a novel model-free approach for pricing multivariate options. The approach is entirely driven by prices of existing options, which we use for inferring the risk-neutral joint distribution among the stocks. Pricing methods that appear in the literature typically amount to selecting a parametric multi-stock model that reflects salient features observed in financial markets (e.g., jumps, fat tails, tail dependence) and next calibrating it in a manner that renders it consistent with the prices of existing options. For instance, Cont and Deguest (2013) propose a mixture of models that can reproduce some set of multivariate options prices and individual options. Other approaches build on Lévy processes. However, while their application in univariate stock modeling is well understood, the multivariate setting still represents challenges. Issues include difficulties with their estimation (curse of dimensionality) and concerns about their ability to closely match dependence patterns observed in the data. Loregian et al. (2018) develop an efficient estimation methodology for a multivariate model driven by Lévy processes. The approach allows them to model a possibly complex dependence among assets, with different tail behaviors and jump structures for each stock; see also Ballotta and Bonfiglioli (2016). Other studies are conducted by Avellaneda and Boyer-Olson (2002) and Jourdain and Sbai (2012).

The above approaches rely on strong parametric assumptions and, thus, might be prone to misspecification and overfitting. In contrast, our approach makes it possible to build a joint model in a completely model-free fashion and ensures that it is fully consistent with the market prices of options on both individual stocks and their indices. The estimated multivariate risk neutral distribution can then be used to price other path-independent derivatives written on the stocks, such as basket options, exchange options, and rainbow options.

The new approach that we develop can be seen as a generalization of the wellknown methodology to infer the risk-neutral density (RND) of a single stock when a wide range of traded options on this stock is available. The methodology relies on the no-arbitrage relationship first discovered by Ross (1976), Breeden and Litzenberger (1978), and Banz and Miller (1978) and makes it possible to estimate risk-neutral densities in a model-free way; see, for example, Jackwerth and Rubinstein (1996), Aït-Sahalia and Lo (2000), and Bondarenko (2003). In a recent survey, Figlewski (2018) reviews pros and cons of the numerous methods that have been used to extract a univariate RND from options prices. Here, we propose a model-free way to obtain a joint multivariate RND of several stocks.

Specifically, we use the market prices of options written on individual stocks to fully describe the marginal distributions of these stocks and we use the prices of options on available indices (which could possibly overlap) to obtain the marginal distributions of linear combinations. Then finding a joint distribution among the stocks can be cast as a combinatorial problem in which a matrix of stock returns needs to be arranged in a suitable manner. Our numerical procedure builds on the so-called Rearrangement Algorithm ${ }^{1}$ (RA) and its generalization Block Rearrangement Algorithm (BRA).

\footnotetext{
${ }^{1}$ The Rearrangement Algorithm was introduced by Puccetti and Rüschendorf (2012) and Embrechts et al. (2013) as a practical tool for assessing the impact of model uncertainty on Value-at-Risk estimates for portfolios.
}

Bernard et al. (2018) use BRA to solve a related, but simpler problem of constructing a multivariate model consistent with information on the marginal distributions of stocks and on the distribution of one sum. Here, we assume that prices of options on multiple indices are available and generalize their method to a much richer environment where several linear combinations have known distributions. For example, Microsoft Corp (MSFT) can be a constituent of several market indices, including Dow Jones Industrial Average (DJIA), S\&P 500, S\&P 100, NASDAQ 100, S\&P Technology Select Sector, and others. These indices differ in the number of constituents and their weights (some indices are price-weighted, while others are capitalization-weighted). Importantly, for many of these market indices, liquid options are traded for a wide range of strikes and we assume that the marginal distributions for those indices can be accurately estimated.

Finding a joint distribution consistent with distributions of multiple indices, as opposed to one index, is a much more challenging problem. To this end, we propose a novel generalization of BRA, termed the Constrained Block Rearrangement Algorighm (CBRA), which is able to handle multiple constraints. When the constraints are compatible, CBRA constructs a joint distribution which perfectly reproduces options on both the individual stocks and indices. However, when a suitable multivariate model does not exist, this indicates that the prices of options on the individual stocks are inconsistent with those on the indices, implying that there are arbitrage opportunities available in the market.

In general, it cannot be expected that there is a single joint model consistent with the input data. The marginal distribution for a given stock or index is inferred from its options and, assuming that a continuum of such options is available, the marginal distribution is uniquely determined. However, the dependence (copula) of a multivariate model that reproduces a set of options on indices is not uniquely determined unless all possible linear combinations are available, which is never the case in practice. While there might be many candidate joint distributions, the evidence in Bernard et al. (2018) shows that the BRA maximizes entropy and thus yields the most plausible model given the information contained in available option prices.

The rest of the paper is organized as follows. In Sect. 1, we motivate and set up the financial problem. Section 2 formally presents the new algorithm to infer a joint distribution. Section 3 then discusses how the algorithm can be implemented in practice to price multivariate derivatives. Section 4 presents two pedagogic illustrations, one with compatible and one with incompatible constraints. Section 5 provides some conclusions.

\section*{1 Problem setup}

Consider a financial market with $d \geq 2$ risky stocks. Stocks pay no dividends and their prices are modeled as a $\mathbb{R}_{+}^{d}$-valued random process $\left(X_{1}(t), \ldots, X_{d}(t)\right)_{t \in[0, T]}$ on a stochastic basis $(\Omega, \mathcal{A}, \mathbb{P})$. There also exists a risk-free stock with interest rate $r$.

We are interested in finding time-0 price of a general path-independent derivative, i.e., a derivatives whose payoff is a function of $\left(X_{1}(T), \ldots, X_{d}(T)\right)$. In what follows, we omit the reference to time in notation, i.e., we write $X_{j}$ instead of $X_{j}(T)$. Let $C_{j}(L)$
and $P_{j}(L)$ denote the time-0 price of the European-style call and put options with strike $L$ and maturity $T$ on the underlying stock $X_{j}$. Under the standard assumptions, the call price is equal to the expected value of its payoff under a suitably chosen risk-neutral probability measure $\mathbb{Q}$ :
$$
\begin{aligned}
& C_{j}(L)=e^{-r T} E^{\mathbb{Q}}\left[\left(X_{j}-L\right)^{+}\right]=e^{-r T} \int_{0}^{\infty}(x-L)^{+} f_{j}(x) d x \\
& P_{j}(L)=e^{-r T} E^{\mathbb{Q}}\left[\left(L-X_{j}\right)^{+}\right]=e^{-r T} \int_{0}^{\infty}(L-x)^{+} f_{j}(x) d x
\end{aligned}
$$
where $f_{j}(x)$ denotes the risk-neutral density (RND). The RND satisfies the relationship first established by Ross (1976), Breeden and Litzenberger (1978), and Banz and Miller (1978):
$$
\begin{equation*}
f_{j}(x)=\left.e^{r T} \frac{\partial^{2} C(L)}{\partial L^{2}}\right|_{L=x}=\left.e^{r T} \frac{\partial^{2} P(L)}{\partial L^{2}}\right|_{L=x} . \tag{1}
\end{equation*}
$$

Similarly, the risk-neutral cumulative distribution (RNCD) satisfies:
$$
\begin{equation*}
F_{j}(x)=\left.e^{r T} \frac{\partial C(L)}{\partial L}\right|_{L=x}=\left.e^{r T} \frac{\partial P(L)}{\partial L}\right|_{L=x} . \tag{2}
\end{equation*}
$$

Although not directly observable, the RNCD can be recovered using the relationship in (2), provided that prices of options with continuum of strikes $L \in \mathbb{R}$ are available. In practice, option prices are only available for a finite number of strikes. Nevertheless, a number of efficient nonparametric approaches have been proposed in the literature that make it possible to circumvent this shortcoming. ${ }^{2}$ Note that for our approach we only need the information on RNCD, and not on RND. The former can be estimated considerably more accurately. This is because it requires estimating the first, rather than the second, derivative of a function and the so-called curse of differentiation is not as severe.

In what follows, we assume that the options market offers sufficiently enough strikes that the risk-neutral cumulative distributions $F_{1}, \ldots, F_{d}$ for the stocks $X_{1}, \ldots, X_{d}$ can be accurately estimated. Besides the individual stocks, we assume that market participants can also trade various indices, as well as their options. Specifically, suppose that there are $K$ market indices (or, baskets) available, where
$$
\begin{equation*}
S_{k}=\sum_{i=1}^{d} \alpha_{i}^{k} X_{i}, \quad k=1, \ldots, K . \tag{3}
\end{equation*}
$$

For example, Microsoft Corp (MSFT) can be a constituent of several market indices, including Dow Jones Industrial Average (DJIA), S\&P 500, S\&P 100, NASDAQ 100,

\footnotetext{
${ }^{2}$ Examples of RND estimation techniques can be found in Jackwerth and Rubinstein (1996), Aït-Sahalia and Lo (2000), and Bondarenko (2003). See also Figlewski (2018) for the review of the pros and cons of the methods proposed in the literature.
}

S\&P Technology Select Sector, and others. These indices differ in the number of constituents and their weights. Importantly, for many of these market indices, liquid options are traded for a wide range of strikes. Thus, we assume that the relationship in (2) can also be used to obtain the corresponding risk-neutral distributions $F_{\alpha^{k}}$ of $S_{k}$ for $k=1, \ldots, K$.

Armed with the risk-neutral distributions for the $d$ individual stocks and $K$ indices (i.e., $F_{j}$ for $j=1, \ldots, d$ and $F_{\alpha^{k}}$ for $k=1, \ldots, K$ ), we are left with solving the problem of finding a compatible joint distribution given the information on the marginal distributions of each individual stock and on some linear combinations of stocks. In fact, if we only use the marginal distributions $F_{j}$ as source of information (i.e., when $K=0$ ), then for each $d$-dimensional copula $C$ the map from $\mathbb{R}^{d}$ to $[0,1]$
$$
\left(x_{1}, \ldots, x_{d}\right) \mapsto C\left(F_{1}\left(x_{1}\right), \ldots, F_{d}\left(x_{d}\right)\right)
$$
defines a $d$-dimensional joint distribution function $F$ with univariate marginal distributions $F_{1}, \ldots, F_{d}$. In this case, every possible dependence (copula) can be used to build a joint distribution for the random vector $\left(X_{1}, \ldots, X_{d}\right)$. However, when we also want to incorporate additional information contained in the distributions $F_{\alpha^{k}}, k=1, \ldots, K$, then only a subset of copulas will still be admissible. However, it is no longer clear how to specify them. In this paper we address this problem.

In some situations, the subset of admissible copulas might actually be empty, suggesting that no joint distribution exists which is consistent with the observed prices of options on individual stocks and the indices. Such a situation would indicate a potential arbitrage opportunity, i.e., when individual options are mispriced relative to the index options. ${ }^{3}$

\section*{2 Inferring a joint distribution}

In this section, we formalize the problem and then present the algorithm. This section is voluntarily a bit abstract, as it presents the algorithm in its full generality, while the application to pricing multivariate derivatives and detecting arbitrage opportunities is given in the next section.

\subsection*{2.1 Preliminaries}

The information that is available on the marginal distribution functions $F_{j}(j= 1, \ldots, d)$ and $F_{\alpha^{k}}(k=1, \ldots, K)$ is generally not sufficient to identify a unique joint distribution. ${ }^{4}$ In order to find a compatible distribution, we effectively need to specify a joint dependence of a vector $\left(U_{1}, \ldots, U_{d}\right)$ of uniformly distributed variables, i.e., a

\footnotetext{
${ }^{3}$ For example, it is well-known that a portfolio of individual options is always more expensive than an option on the portfolio, where the strikes are chosen appropriately. If this property is violated for some portfolio, then an arbitrage opportunity will exist.
${ }^{4}$ When $F_{\alpha}$ is available for any possible choice of $\alpha \in \mathbb{R}^{d}$, then the characteristic function of ( $X_{1}, X_{2}, \ldots, X_{d}$ ) is also known, from which the unique joint distribution function can be obtained (Lévy 1926; Cramér 1946). When there is only a limited number of linear combinations with known distributions,
}
copula, having the property that for all $\mathrm{x} \in \mathbb{R}$ and for all $k=1, \ldots, K$
$$
\begin{equation*}
\mathbb{P}\left(\sum_{j=1}^{d} \alpha_{j}^{k} F_{j}^{-1}\left(U_{j}\right) \leq x\right)=F_{\alpha^{k}}(x) . \tag{4}
\end{equation*}
$$

Indeed, in this case, a compatible joint distribution $F$ is obtained as
$$
F\left(x_{1}, \ldots, x_{d}\right)=\mathbb{P}\left(F_{1}^{-1}\left(U_{1}\right) \leq x_{1}, \ldots, F_{d}^{-1}\left(U_{d}\right) \leq x_{d}\right), \quad\left(x_{1}, \ldots, x_{d}\right) \in \mathbb{R}^{d} .
$$

Next, we observe that in order to obtain a consistent distribution, it is equivalent to look for rearrangements $q_{j}$ of $F_{j}^{-1}$ such that for a uniformly distributed variable $U$ and all $k=1, \ldots, K$ it holds that
$$
\begin{equation*}
\mathbb{P}\left(\sum_{j=1}^{d} \alpha_{j}^{k} q_{j}(U) \leq x\right)=F_{\alpha^{k}}(x), \tag{5}
\end{equation*}
$$
since then
$$
F\left(x_{1}, \ldots, x_{d}\right)=\mathbb{P}\left(q_{1}(U) \leq x_{1}, \ldots, q_{d}(U) \leq x_{d}\right)
$$

In the sequel, we use the latter formulation for describing a joint distribution and we thus merely aim at finding rearrangements $q_{j}$ of the $F_{j}^{-1}$ such that the $K$ constraints in (5) are all satisfied. To this end, we first need to discretize the problem. Specifically, since each $F_{j}$ can be approximated to any degree of accuracy by selecting a sufficiently large number of discrete states $n$, we represent $F_{j}$ as follows
$$
F_{j}(x)=\frac{1}{n} \sum_{i=1}^{n} \mathbb{1}_{\left(x_{i j}, \infty\right)}(x), \quad j=1, \ldots, d
$$
where the values $x_{i j}$ are chosen as
$$
\begin{equation*}
x_{i j}=F_{j}^{-1}\left(\frac{i-0.5}{n}\right), \quad i=1, \ldots, n . \tag{6}
\end{equation*}
$$

In other words, we sample $n$ equiprobable values from each distribution $F_{j}$ to obtain an $n \times d$ matrix
$$
\mathbf{X}=\left(x_{i j}\right)=\left[\begin{array}{cccc}
x_{11} & x_{12} & \ldots & x_{1 d} \\
x_{21} & x_{22} & \ldots & x_{2 d} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n 1} & x_{n 2} & \ldots & x_{n d}
\end{array}\right],
$$

\footnotetext{
Footnote 4 continued
then many compatible joint distributions will typically exist, but it is no longer clear a priori how to find them.
}
which represents the support of the comonotonic random vector ( $F_{1}^{-1}(U), \ldots$, $F_{d}^{-1}(U)$ ). The rows of matrix $\mathbf{X}$ depict all joint realizations, each occurring with probability $1 / n$ (note that there might be duplicates), and its columns represent the random variables $F_{j}^{-1}(U), j=1, \ldots, d$. This observation motivates why in the sequel we apply probabilistic operators (such as the variance) onto columns of matrices. We use the notation $X_{j}$ for depicting both the $j$-th column of $\mathbf{X}$ as well as the corresponding random variable.

We follow a similar approach to discretize the distribution functions $F_{\alpha^{k}}$ for the $K$ linear constraints and obtain an $n \times K$ matrix
$$
\mathbf{S}=\left(s_{i k}\right)=\left[\begin{array}{cccc}
s_{11} & s_{12} & \ldots & s_{1 K} \\
s_{21} & s_{22} & \ldots & s_{2 K} \\
\vdots & \vdots & \ddots & \vdots \\
s_{n 1} & s_{n 2} & \ldots & s_{n K}
\end{array}\right] \text {, }
$$
where the column vector $S_{k}$ contains the sampled values from $F_{\alpha^{k}}$, i.e.,
$$
s_{i k}=F_{\alpha^{k}}^{-1}\left(\frac{i-0.5}{n}\right), \quad i=1, \ldots, n .
$$

Next we let
$$
\begin{equation*}
\mathbf{Y}=[\mathbf{X} ; \mathbf{S}] \tag{7}
\end{equation*}
$$
denote an $n \times(d+K)$ matrix, which combines the columns for the $d$ marginal distributions and $K$ linear constraints. By rearranging the $n$ elements within each column of matrix $\mathbf{Y}$ we can obtain a new matrix $\mathbf{Y}^{\pi}$ :
$$
\mathbf{Y}^{\pi} \in \mathcal{P}(\mathbf{Y}):=\left\{\left(y_{i j}^{\pi}\right): \quad y_{i j}^{\pi}:=y_{\pi_{j}(i) j}, \quad i=1, \ldots, n, \quad j=1, \ldots, d+K\right\},
$$
in which $\pi_{1}, \ldots, \pi_{d+K}$ denote permutations of $\{1, \ldots, n\}$. In what follows we will call the vector of permutations $\pi:=\left(\pi_{1}, \ldots, \pi_{d+K}\right)$ a rearrangement, $\mathbf{Y}^{\pi}$ a rearranged matrix, and $\mathcal{P}(\mathbf{Y})$ the set of all rearranged matrices. Ideally, we aim to find a rearranged matrix $\mathbf{Y}^{\pi} \in \mathcal{P}(\mathbf{Y})$ such that all $K$ constraints in (5) are exactly satisfied. However, because of discretization errors such rearrangement may not exist, and even when it does exist there might be no clear way to obtain it.

\subsection*{2.2 Problem reformulation to handle multiple constraints}

In order to deal with the multiple constraints in (5), i.e., the available information on the $K$ constraints on the distributions of linear combinations, we consider the following closely related problem of finding a rearranged matrix $\mathbf{Y}^{\pi}$, which minimizes the objective
$$
\begin{equation*}
V:=V\left(\mathbf{Y}^{\pi}\right)=\sum_{k=1}^{K} \operatorname{var}\left(L_{k}^{\pi}\right) \tag{8}
\end{equation*}
$$
where
$$
\begin{equation*}
L_{k}^{\pi}:=\sum_{j=1}^{d+K} \tilde{\alpha}_{j}^{k} Y_{j}^{\pi}, \quad k=1, \ldots, K, \tag{9}
\end{equation*}
$$
and
$$
\widetilde{\alpha}_{j}^{k}=\left\{\begin{array}{cl}
\alpha_{j}^{k}, & \text { for } j=1, \ldots, d \\
-1, & \text { for } j=d+k \\
0, & \text { otherwise }
\end{array}\right.
$$

Intuitively, we want each $n \times 1$ vector $L_{k}^{\pi}$ to be exactly zero, so that the (discretized versions of) constraints in (5) are perfectly satisfied. Since this ideal situation is typically unattainable, we want to get as close as possible to it by minimizing the objective $V$.

Remark 1 In the sum of the variances (8), each constraint has the same weight in the sum of the variances but a weighted sum of the variances can also be considered without any further complications. Such a situation is of interest if some of the distributional constraints are more accurate than others. In the context of multivariate option pricing, each constraint comes from the risk neutral distribution of a given index and its accuracy might depend on the number of options available on that index and the liquidity of these options. Therefore, a constraint coming from a large set of liquid options could in principle be weighted more in the objective (8).

Assessing the objective $V$ for all possible rearranged matrices is not practical as there are $(n!)^{d+K-1}$ possibilities to evaluate. It turns out, however, that a particular kind of rearrangements, called block rearrangements, are suitable to the problem we consider. A block rearrangement consists in swapping entire rows within a given group of columns (a "block") rather than elements within a single column.

Specifically, we represent a block of columns by a ( $d+K$ ) binary vector $\delta$, where $\delta \in \Delta:=\{0,1\}^{d+K}$ and $\delta_{j}=1$ indicates that the column $j$ belongs to the block, $j=1, \ldots, d+K$. A rearrangement $\pi$ of a matrix $\mathbf{Y}$ into $\mathbf{Y}^{\pi}$ is called a block rearrangement if there exists $\delta \in \Delta$ and a permutation $\pi^{\delta}:\{1, \ldots, n\} \rightarrow\{1, \ldots, n\}$ such that $\pi_{j}(\cdot)=\pi^{\delta}(\cdot)$ for $\delta_{j}=1$ and $\pi_{j}(\cdot)=(\cdot)$ for $\delta_{j}=0$ (which means that for all $x, \pi_{j}(x)=x$ ). In other words, a block rearrangement swaps same rows within the selected block of columns and leaves the other columns unchanged.

As a special case, if the vector $\delta \in \Delta$ satisfies the condition $\sum_{j=1}^{d+K} \delta_{j}=1$, i.e., the "block" only consists of a single column, then the rearrangement reduces to rearranging this particular column. Finally, for any binary vector $\delta \in \Delta$, it is convenient to define a set $I_{\delta}$, where $j \in I_{\delta}$ if and only if $\delta_{j}=1$. The set $I_{\delta}$ and its complementary set $I_{\delta}^{c}$ partition the ( $d+K$ ) columns into two groups, i.e., they satisfy two conditions: $I_{\delta} \cap I_{\delta}^{c}=\varnothing$ and $I_{\delta} \cup I_{\delta}^{c}=\{1, \ldots, d+K\}$. In what follows, we refer to $\delta$ as also a block.

\subsection*{2.3 Constrained block rearrangement algorithm}

We propose an algorithm for rearranging $\mathbf{Y}$ into $\mathbf{Y}^{\pi} \in \mathcal{P}(\mathbf{Y})$ such that objective (8) is as small as possible. The algorithm builds on the fact that two random variables $W$ and $Z$ with given distributions have minimum correlation if and only if they are antimonotonic. Intuitively, the two variables $W$ and $Z$ are antimonotonic (or countermonotonic) if they always move in the opposite directions. This is denoted as $W \uparrow \downarrow Z$.

We rely on the following result, which is a well-known corollary of the Hoeffding's identity (Hoeffding (1940)).

Lemma 2.1 (Minimum Covariance) Let $W$ and $Z$ be random variables. Assume that $\widetilde{W}$ has the same distribution as $W$ and is anti-monotonic with $Z$. Then
$$
\operatorname{cov}(W, Z) \geq \operatorname{cov}(\tilde{W}, Z)
$$

The inequality is strict, when $W$ is not anti-monotonic with $Z$.
In order to construct a rearranged matrix that minimizes the objective (8), we note that for any $\delta \in \Delta$, it holds that
$$
\sum_{k=1}^{K} \operatorname{var}\left(L_{k}\right)=C+\sum_{k=1}^{K} \operatorname{cov}\left(\sum_{j \in I_{\delta}} \tilde{\alpha}_{j}^{k} Y_{j}, \sum_{j \in I_{\delta}^{c}} \tilde{\alpha}_{j}^{k} Y_{j}\right),
$$
where $I_{\delta}$ and $I_{\delta}^{c}$ are the corresponding partitioning sets and
$$
\begin{equation*}
C=\sum_{k=1}^{K}\left(\operatorname{var}\left(\sum_{j \in I_{\delta}} \tilde{\alpha}_{j}^{k} Y_{j}\right)+\operatorname{var}\left(\sum_{j \in I_{\delta}^{c}} \tilde{\alpha}_{j}^{k} Y_{j}\right)\right) . \tag{10}
\end{equation*}
$$

Definition 2.2 (Admissible blocks) A block $\delta \in \Delta$ is called admissible if for every constraint $k \in\{1, \ldots, K\}$, the coefficients $\widetilde{\alpha}_{j}^{k}$ do not depend on $j$, i.e., $\widetilde{\alpha}_{j}^{k}:=\widetilde{\alpha}^{k}$, $j \in I_{\delta}$. We denote by $\mathcal{B}$ the set of all admissible blocks $\delta$.

Note that the terms for $C$ given in (10) do not change when applying a block rearrangement $\pi_{\delta}$ where $\delta$ is an admissible block. Note also that the set $\mathcal{B}$ is never empty, that is, admissible blocks always exists. In particular, the stated condition on $\widetilde{\alpha}_{j}^{k}$ is satisfied for all singleton blocks, that is for all $\delta \in \Delta$ such that $\sum_{j=1}^{d+K} \delta_{j}=1$.

The point is then that, for any admissible $\delta$, one can identify a corresponding block rearrangement $\pi^{\delta}$ that decreases the value of the objective $V$ for the rearranged matrix $\mathbf{Y}^{\boldsymbol{\pi}}$. Indeed, it follows from Lemma 2.1 that to lower $V, \pi^{\boldsymbol{\delta}}$ should be taken such that
$$
\sum_{j \in I_{\delta}} \tilde{\alpha}^{k} Y_{j}^{\pi^{\delta}} \uparrow \downarrow\left(\sum_{j \in I_{\delta}^{c}} \tilde{\alpha}_{j}^{k} Y_{j}\right) .
$$

These considerations lead to the so-called Constrained Block Rearrangement Algorithm (CBRA).

\section*{Constrained Block Rearrangement Algorithm (CBRA):}
1. Starting from the initial matrix $\mathbf{Y}$, apply for every $\delta \in \mathcal{B}$ the block rearrangement $\pi^{\delta}$ such that
$$
\sum_{j \in I_{\delta}} \tilde{\alpha}^{k} Y_{j}^{\pi^{\delta}} \uparrow \downarrow\left(\sum_{j \in I_{\delta}^{c}} \tilde{\alpha}_{j}^{k} Y_{j}\right) .
$$
2. Compute $V:=\sum_{k=1}^{K} \operatorname{var}\left(L_{k}^{\pi^{\delta}}\right)$.
3. If there is no improvement in $V$, output the current matrix $\mathbf{Y}^{\pi^{\delta}}$, otherwise, return to Step 1.

From the output matrix, one can then simply isolate the first $d$ columns (corresponding to rearrangements of the $X_{j}$ ), which yields the solution of the problem we consider. The algorithm guarantees that after each block rearrangement the objective $V$ decreases and, thus, the algorithm converges in finite time to a local optimum. In the ideal situation, after the algorithm stops, the rearranged matrix is such that $V=0$ and thus all $L_{k}^{\pi}=0, k=1, \ldots, K$. However, due to discretization errors and to the fact that the algorithm is a heuristic, the final $L_{k}^{\pi}$ will deviate from the zero vector. ${ }^{5}$ We can denote these deviations by column vectors $E_{k}, k=1, \ldots, K$ (one column for each constraint). It is then possible to test whether these deviations (errors) are significant, in that one could adjust $S_{k}$ for the noise $E_{k}$ and assess whether the perturbed sum $S_{k}+E_{k}$ still statistically follows the given prescribed distribution $F_{S_{k}}$.

Remark 2 (Block RA) When $K=1$ and the constraint is that the total sum $X_{1}+\cdots+X_{d}$ is constant (a degenerate distribution), our CBRA reduces to the Block RA (BRA) studied in Bernard et al. (2018). Specifically, this case reduces to the so-called problem of joint mixability: Can we construct random variables with given marginal distributions and having a dependence such that their sum is constant? This reduced problem finds its pedigree in Gaffke and Rüschendorf (1981) and has been extensively studied in series of papers, including (Wang and Wang 2011; Puccetti et al. 2012), and Wang and Wang (2016). However, full characterization of marginal distributions that yield mixability is still an open problem and this is true even when all marginal distributions are identical. That is exactly why Puccetti and Rüschendorf (2012) and Embrechts et al. (2013) have developed the Rearrangement Algorithm (RA), a numerical method to obtain a dependence that yields approximate mixability for given marginal distributions. In later work RA has been generalized to the aforementioned BRA. ${ }^{6}$

\footnotetext{
5 Bernard et al. (2018) considers the special case when there is only one distributional constraint and provides an extensive numerical study to demonstrate the strength of the BRA method and its convergence. However, even for this special case, the formal convergence results are not yet available. Haus (2015) has shown that even the simplest rearrangement problem, in which the only constraint is that the distribution of the sum is constant, is NP-complete and therefore cannot be solved by an algorithm that has polynomial complexity.
${ }^{6}$ (Bernard and McLeish 2016; Bernard et al. 2017), and Boudt et al. (2018).
}

Remark 3 (Complexity) The construction of the set of admissible blocks $\mathcal{B}$ is in general a hard combinatorial problem. However, for the typical applications that we have in mind, this should not be a real issue for two reasons. First, the set $\mathcal{B}$ only needs to be computed once. Indeed, the set remains the same for each day and for each option maturity $T$, as the compositions of market indexes change very infrequently. Therefore, even if it takes considerable amount of time to generate $\mathcal{B}$, this has to be done only once. In practice, the cost of constructing the set $\mathcal{B}$ is much lower than the main iterative optimization and it affects the overall complexity of CBRA in additive way. Second, in financial applications, the number of constraints is likely to be rather small, say $K \leq 3$. This is because there are relatively few market indices, for which liquid options are available. The number of stocks could also be moderate, say $d=30$ or less. In those cases, it is feasible to generate admissible blocks by direct, brute-force methods. ${ }^{7}$

Remark 4 (Generalization) The above algorithm can be generalized to the case in which an arbitrary number of distributions of linear combinations is known, but not necessarily the marginal distributions for all of the $d$ individual variables $X_{j}(j= 1, \ldots, d)$ are available, as we assume in this paper. ${ }^{8}$ We are grateful to Ruodu Wang for this suggestion.

Remark 5 (Incompatibility) The algorithm we propose can always be applied, even when the distributional constraints imposed on the linear combinations of the $X_{i}$ ( $i=1, \ldots, d$ ) are ill-specified in that no joint distribution for ( $X_{1}, \ldots, X_{d}$ ) exists that satisfies these constraints. While in this case there is no hope that the minimum value for $V$ will be close to zero, we will obtain a joint distribution that matches as closely as possible the imposed constraints. We illustrate this feature in Example 4.2. This situation can be used to detect arbitrage opportunities in the option markets.

Remark 6 (Maximum Entropy) We present a generalization of the methodology exposed in Bernard et al. (2018) referred to as BRA. While there might be many compatible joint distributions, BRA is shown to maximize entropy and thus yields the "most likely" joint distribution among asset returns given the available option data. In this regard, it is also worth citing Jaynes (2003) [p. 370], who developed the principle of maximum entropy in its modern form and who stated: "In summary, the principle of maximum entropy is not an oracle telling which predictions must be right; it is a rule for inductive reasoning that tells us which predictions are most strongly indicated by our present information."

To make clear what is meant by "most likely," let us take a step back and assume for a moment that we only have information on the marginal distributions of the assets, i.e., we only agree on the values that appear in the first $d$ columns but not on the order in which they appear, i.e., all permutations within columns are equally plausible

\footnotetext{
${ }^{7}$ For very large applications, say, $d=500$ stocks in the S\&P 500 index, one still has an option to use the Constrained RA (CRA) instead of Constrained BRA. Although less accurate than CBRA, CRA is much faster. CRA cycles through $(d+K)$ singleton blocks, as opposed to all admissible blocks in $\mathcal{B}$ in CBRA. The next subsection provides further discussion of complexity.
${ }^{8}$ Clearly, the marginal distribution for a single variable $X_{j}$ may be viewed as a special case of a linear constraint.
}
and there is no reason to prefer one permutation over another. Hence, randomizing the assignment of realizations to the different states leads to marginal distribution functions (reflected by the columns) that are most likely to be independent, which corresponds precisely to the maximum entropy case. Using the additional known information, namely, the $K$ marginal distributions of the weighted sums, merely implies that the set of admissible permutations reduces to those that satisfy the constraints. Our method implements the idea of randomizing the assignment of realizations to the different states, but now under the additional constraints provided by the distributions of the indices. ${ }^{9}$

\subsection*{2.4 Connection to optimal transport problem}

In the previous section, we have developed CBRA to solve the problem of finding a joint distribution $F$ compatible with a certain number of constraints given in (4). To do so, we have discretized the problem using quantiles and directly focused on solving the primal problem. An alternative is to formulate the problem as an optimal transport (OT) and use the Monge-Kantorovitch duality approach to tackle it [see e.g., Galichon (2018)]. ${ }^{10}$

For the ease of presentation, we illustrate this connection in two dimensions and a single constraint, $d=2$ and $K=1$. We assume that we know the distributions $F_{1}, F_{2}$, and $F_{S}$ for the two stocks $X_{1}$ and $X_{2}$, as well as their weighted sum $S= \alpha X_{1}+(1-\alpha) X_{2}$. The corresponding density functions are denoted $f_{1}, f_{2}$, and $f_{S}$. We are then looking for a three-dimensional joint density function $g\left(X_{1}, X_{2}, S\right)$ for which the marginal distributions are fixed. The classical optimal transport problem can be formulated as follows
$$
\begin{equation*}
\arg \min _{g \in \mathcal{G}}\left\{\int_{\mathbb{R}^{3}} c\left(x_{1}, x_{2}, s\right) g\left(d x_{1}, d x_{2}, d s\right): \quad g_{1}=f_{1}, g_{2}=f_{2}, g_{3}=f_{S}\right\}, \tag{11}
\end{equation*}
$$
where the optimization is taken over the Frechet class $\mathcal{G}$ of all three-dimensional distributions $g$ with given marginal densities $f_{1}, f_{2}, f_{S}$, and aims at minimizing the cost function
$$
c\left(x_{1}, x_{2}, s\right)=\left(\alpha x_{1}+(1-\alpha) x_{2}-s\right)^{2} .
$$

As a solution, we obtain a three-dimensional density $g\left(X_{1}, X_{2}, S\right)$ and, thus, also the joint density between $X_{1}$ and $X_{2}$, i.e., $g_{1,2}\left(X_{1}, X_{2}\right)$. Problem in (11) is the primal problem and, using the Monge-Kantorovitch duality, it can be restated as the following dual problem

\footnotetext{
${ }^{9}$ In the context of inference of marginal RNDs from observed option prices, the principle of maximum entropy has been explored in Rubinstein (1994) and in Jackwerth and Rubinstein (1996) and has been employed in Stutzer (1996).
${ }^{10}$ We are grateful to the anonymous referee for pointing out this interesting alternative formulation. The connection between OT and CBRA is rather fascinating and it opens exciting possibilities for future research. Perhaps the extensive machinery developed for OT can also help in analyzing CBRA.
}
$$
\begin{aligned}
& \sup _{h_{1}, h_{2}, h_{3}}\left\{\int_{\mathbb{R}} h_{1}\left(x_{1}\right) g_{1}\left(d x_{1}\right)+\int_{\mathbb{R}} h_{2}\left(x_{2}\right) g_{2}\left(d x_{2}\right)+\int_{\mathbb{R}} h_{3}(s) g_{3}(d s):\right. \\
& \left.h_{1}\left(x_{1}\right)+h_{2}\left(x_{2}\right)+h_{3}(s) \leq\left(\alpha x_{1}+(1-\alpha) x_{2}-s\right)^{2}\right\} .
\end{aligned}
$$

The dual problem can then be solved using, for instance, the Sinkhorn algorithm, see Peyré (2019). It is straightforward to extend this formulation to the more general case we study in this paper with arbitrary $d$ and $K$. The parallel with the algorithm presented in the previous section is that CBRA finds a $(d+K)$-dimensional joint distribution as the matrix $\mathbf{Y}$ in (7), where the first $d$ columns of the matrix $\mathbf{Y}$ represent the $d$-dimensional joint distribution of ( $X_{1}, X_{2}, \ldots, X_{d}$ ).

The two approaches have their pros and cons. The OT formulation has the advantage of being very general and flexible. It is also well-studied. In contrast, our approach is more limited and designed to solve a specific problem. It cannot incorporate certain constraints that OT can. In particular, it will fail to include martingale constraints as has been done by Henry-Labordere (2017) and De March (2018), among others. Furthermore, the regularization makes the OT problem convex, whereas our optimization is not convex and may lead to local optima.

Nevertheless, we believe that for the specific problem at hand, the CBRA approach has critical computational advantages and does not suffer from the curse of dimensionality as much as the OT approach. To clarify this point, consider an empirically relevant case studied in Bondarenko and Bernard (2020), where there are 9 economic sectors which comprise the the S\&P 500 index. This is probably the smallest possible, but still interesting application for the methodology, with $d=9$ and $K=1$. If the OT approach is used for this application, we would need to find a joint density by discretizing the $\mathbb{R}^{d+K}$ space. Suppose each dimension is discretized into only $m=10$ values, which is arguably a very crude approximation of each margin, then the OT optimization problem will have $m^{d+K}=10^{10}$ variables, which is already too large for most computers to handle. In CBRA, on the other hand, the joint distribution is represented by an $n \times(d+K)$ matrix, where $n$ is the number of discretization points, or quantiles chosen. Bondarenko and Bernard (2020) use $n=1000(n=10,000$, as a robustness check) and are able to easily find joint distributions for thousands of trading days. The point is that, in the CBRA approach, the $n$ discretization states represent the joint distribution collectively, not separately for each dimension. When the number of states is $n=1000$, the joint distribution is already approximated very well, but the approximation can be made as accurate as desired, given that shuffles are dense in the set of copulas and that our representation of the joint distribution as a matrix can also be formulated as a shuffle.

To summarize, the joint distribution is represented as an $n \times(d+K)$ object in the CBRA approach, but as an $m^{d+K}$ object in the OT approach. The curse of dimensionality affect the latter approach, but not the former one - in OT, both memory and computational demands explode exponentially with $(d+K){ }^{11}$

\footnotetext{
${ }^{11}$ In our experiments, we have successfully implemented the CBRA approach for $d=30$ stocks in the DJIA. This case would be completely hopeless for the OT approach.
}

\section*{3 Application to pricing multivariate derivatives}

In this section, we explain how to apply the method outlined in the previous section to price a multivariate derivative in a manner consistent with market prices of options on both the individual stocks and their indices.

Suppose that we observe market prices of European-style options on $d$ stocks $X_{1}, \ldots, X_{d}$ and $K$ indices $S_{1}, \ldots, S_{K}$ with corresponding weights $\alpha_{i}^{k}$ for stock $i= 1, \ldots, d$ in index $k=1, \ldots, K$. All options mature at the same time $T$ and have strikes over wide ranges. Equivalently, we can assume that we have access to the implied volatility curves (IV) of $d$ stocks and $K$ indices.
- Step 1: From available option prices, we estimate the univariate RNCDs of the $d$ individual stocks and $K$ indices. Using the relationship in (2), the univariate RNCDs can be estimated by a number of model-free methods proposed in the literature. We denote them as $F_{i}$ for $i=1, \ldots, d$ and $F_{\alpha^{k}}$ for $k=1, \ldots, K$.
- Step 2: We set the number of states $n$ and discretize distributions $F_{i}$ and $F_{\alpha^{k}}$ as in (6). The corresponding values are used to form the $n \times(d+K)$ matrix $\mathbf{Y}$ :
$$
\mathbf{Y}=\left[\begin{array}{cccc|cccc}
x_{11} & x_{12} & \ldots & x_{1 d} & s_{11} & s_{12} & \ldots & s_{1 K} \\
x_{21} & x_{22} & \ldots & x_{2 d} & s_{21} & s_{22} & \ldots & s_{2 K} \\
\vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots \\
x_{n 1} & x_{n 2} & \ldots & x_{n d} & s_{n 1} & s_{n 2} & \ldots & s_{n K}
\end{array}\right],
$$
- Step 3: Given the matrix $\mathbf{Y}$, we run CBRA described in Sect. 2.3 to build a model consistent with all the risk-neutral distributions $F_{i}$ and $F_{\alpha^{k}}$ obtained in Step 1 (and, thus, with all the available option prices). The output of CBRA is the rearranged matrix $\widetilde{\mathbf{Y}}$, which represents the $n$-state joint model.
- Step 4: We inspect the output matrix $\widetilde{\mathbf{Y}}$ for consistency. Recall that $E_{1}, \ldots, E_{K}$ denote the final deviations for the $K$ constraints in (9). Intuitively, we want to see if the deviations $E_{k}$ are statistically different from zero. To this end, we could test whether columns $S_{k}+E_{k}$ are statistically indistinguishable from the distributions $F_{\alpha^{k}}$ for the indices. Several statistical tests are available, such as a distribution-free multivariate Kolmogorov-Smirnov test (Justel et al. 1997). We then distinguish between two cases: the distributional constraints are satisfied (5a) and not satisfied (5b).
- Step 5a: In this case, a suitable joint RND has been found and we can proceed to price any path-independent multivariate derivative on the $d$ stocks, for example, an exchange option or rainbow option. Specifically, let $G\left(X_{1}(T), \ldots, X_{d}(T)\right)$ denote the time- $T$ payoff of such a derivative. Its price is computed as the discounted average across the $n$ states in the output matrix $\widetilde{\mathbf{Y}}$ :
$$
E^{\mathbb{Q}}\left[G\left(X_{1}(T), \ldots, X_{d}(T)\right)\right]=e^{-r T} \frac{1}{n} \sum_{i=1}^{n} G\left(\tilde{y}_{i 1}, \tilde{y}_{i 2}, \ldots, \tilde{y}_{i d}\right) .
$$

This price of the derivative is arbitrage-free and fully consistent with all options on the individual stocks and indices.
- Step 5b: In this case, the linear constraints are incompatible and no joint distribution can support all the observed option prices. We now can use the output matrix $\tilde{\mathbf{Y}}$ to construct a potential arbitrage opportunity and examine whether its theoretical profitability is sufficiently large to survive after accounting for realistic transaction costs and market frictions.

\section*{4 Illustrations}

In this section, the algorithm is illustrated with a simple six-dimensional example in which the marginal distributions of the six stocks $X_{1}, X_{2}, \ldots, X_{6}$ are given by normal distributions. There are three additional constraints on certain linear combinations of the six random variables, which are also assumed to be normally distributed. In the first example, we study a case where the constraints are compatible and a feasible joint distribution exists. In the second example, we study a case where the constraints are incompatible.

\subsection*{4.1 Compatible normal distributions}

Consider six stocks whose returns are modeled by standard normal random variables, i.e., $X_{j} \sim N(0,1)$, for $j=1, \ldots, 6$. Furthermore, assume that there are three linear combinations (sums) with known normal distribution. Specifically,
$$
\begin{aligned}
X_{1}+X_{2}+X_{3}+X_{4} & \sim N(0,10):=F_{\alpha^{1}}, & & \alpha^{1}=[1,1,1,1,0,0], \\
X_{3}+X_{4}+X_{5}+X_{6} & \sim N(0,10):=F_{\alpha^{2}}, & & \alpha^{2}=[0,0,1,1,1,1], \\
X_{1}+X_{2}+X_{3}+X_{4}+X_{5}+X_{6} & \sim N(0,24):=F_{\alpha^{3}}, & & \alpha^{3}=[1,1,1,1,1,1] .
\end{aligned}
$$

Let $J_{1}=\{1,2,3,4\}, J_{2}=\{3,4,5,6\}$, and $J_{3}=\{1,2,3,4,5,6\}$ denote sets containing the indices of those stocks that appear in the first sum, the second sum and third sum, respectively. For a given set $J$, let $\bar{\rho}(J)$ denote the average pairwise correlation of all stocks with indices in set $J$. That is,
$$
\bar{\rho}(J):=\frac{2}{|J|(|J|-1)} \sum_{i, j \in J, i>j} \rho_{i j},
$$
where $\rho_{i j}$ denotes the correlation between stocks $X_{i}$ and $X_{j}$.
It is easy to check that the above constraints on the three sums are equivalent to the following restrictions on the average correlations:
$$
\begin{equation*}
\bar{\rho}\left(J_{1}\right)=0.5, \quad \bar{\rho}\left(J_{2}\right)=0.5, \quad \bar{\rho}\left(J_{3}\right)=0.6 \tag{12}
\end{equation*}
$$

To infer a possible joint distribution for ( $X_{1}, \ldots, X_{6}$ ), we first discretize the continuous distributions using $n=10,000$ equiprobable values and then run CBRA

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 1 Average Correlations. For the case of compatible constraints, this table reports pairwise correlations for the six stocks averaged across $M$ runs of CBRA}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline & $X_{1}$ & $X_{2}$ & $X_{3}$ & $X_{4}$ & $X_{5}$ & $X_{6}$ \\
\hline $X_{1}$ & 1.000 & 0.858 & 0.500 & 0.499 & 0.782 & 0.782 \\
\hline $X_{2}$ & 0.858 & 1.000 & 0.500 & 0.499 & 0.782 & 0.782 \\
\hline $X_{3}$ & 0.500 & 0.500 & 1.000 & 0.147 & 0.500 & 0.500 \\
\hline $X_{4}$ & 0.499 & 0.499 & 0.147 & 1.000 & 0.500 & 0.500 \\
\hline $X_{5}$ & 0.782 & 0.782 & 0.500 & 0.500 & 1.000 & 0.856 \\
\hline $X_{6}$ & 0.782 & 0.782 & 0.500 & 0.500 & 0.856 & 1.000 \\
\hline
\end{tabular}
\end{table}
$M=1000$ times. ${ }^{12}$ To minimize the objective function, CBRA rearranges along all one-element blocks, as well as three additional two-element admissible blocks: $\{1,2\}$, $\{3,4\}$, and $\{5,6\}$. Depending on initial randomization, each run of CBRA produces a different solution for the joint distribution, but they all satisfy the three constraints in (12) very closely, with the averages across $M$ runs being $0.5006,0.5006$, and 0.5993 , respectively. Table 1 below reports the correlation matrix for the six stocks averaged across all $M$ runs:

Note also that the stocks $X_{1}$ and $X_{2}$ enter the constraints symmetrically (as well as have same marginal distributions). The average pairwise correlations in the above table reflect this symmetry. The same also applies to the stock pairs ( $X_{3}, X_{4}$ ) and ( $X_{5}, X_{6}$ ). However, the symmetry within each of the pairs $\left(X_{1}, X_{2}\right),\left(X_{3}, X_{4}\right),\left(X_{5}, X_{6}\right)$ is not a theoretical requirement for a joint distribution to satisfy the imposed constraints, the symmetry is obtained as a consequence of the algorithm and is consistent with the principle of maximum entropy.

Additional insights are provide by Fig. 1. The left panel shows the distribution of the objective $V$ (normalized by the number of states $n$ ), which is clustered around a small sample mean of 0.00072 . The right panel shows the 10th, 50th and 90th percentiles of distributions of pairwise correlations $\rho_{i j}$. Again, the 15 correlations are all generally tightly clustered, indicating that different runs of CBRA produce similar correlation matrices.

Since CBRA algorithm provides us with a suitable joint distribution, we can now price any path-independent multivariate derivative on the six stocks. For simplicity, the risk-free rate is assumed to be zero. Let $G=G\left(X_{1}, \ldots, X_{6}\right)$ denote the payoff of such a derivative at maturity. We consider three exotic path-independent options written on the final values of the six stocks (or, a subset of them):
- A call on a new basket, which consists of stocks $X_{1}, X_{2}, X_{5}$, and $X_{6}$. The exercise price is $L=5$ and the payoff of the option is given as
$$
G_{1}=\max \left(X_{1}+X_{2}+X_{5}+X_{6}-5,0\right) .
$$

\footnotetext{
${ }^{12}$ We program CBRA algorithm in Matlab and run it in parallel using 60 cores. The running time is approximately 1 minute and on average the algorithm converges to a solution after about 8500 iterations.
}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_09_30_b81faa5a74b6c1ea6646g-17.jpg?height=577&width=1180&top_left_y=185&top_left_x=171}
\captionsetup{labelformat=empty}
\caption{Fig. 1 CBRA results for the case of compatible constraints. The left panel shows the histogram of the objective $V$ (normalized by the number of states $n$ ) across $M$ runs of the algorithm. The dashed vertical line represents the sample mean. The right panel shows the 10th, 50th and 90th percentiles of distributions of pairwise correlations $\rho_{i j}$ generated by $M$ runs. The percentiles are shown along 15 vertical lines, corresponding to correlations $\rho_{12}, \rho_{13}, \ldots, \rho_{56}$. The grey circles are the sample means}
\end{figure}
- A call on the maximum of two stocks $X_{1}$ and $X_{3}$. The exercise price is $L=1$ and the payoff of the option is given as
$$
G_{2}=\max \left(\max \left(X_{1}, X_{3}\right)-1,0\right)
$$
- A call on the maximum of all six stocks $X_{1}, \ldots, X_{6}$. The exercise price is again $L=1$ and the payoff of the option is given as
$$
G_{3}=\max \left(\max \left(X_{1}, \ldots, X_{6}\right)-1,0\right) .
$$

We refer to these three options as $O_{1}, O_{2}$, and $O_{3}$. The prices of these multivariate exotic options depend critically on the joint distribution, especially on its extreme tails. (Since in this simple illustration all marginal distributions are symmetrical, we can focus on call options only, as the results for puts are similar.)

For each run of CBRA and each $i=1,2,3$, we compute the option price $E^{\mathbb{Q}}\left[G_{i}\right]$ as the average across the $n$ states in the output matrix. In Table 2 we report statistics for the distribution of the option price across $M$ runs of the algorithm. This table reveals that these distributions are quite tight. In other words, even though different runs of CBRA yield slightly different candidate joint distributions, those generally agree on prices of various derivatives.

\subsection*{4.2 Incompatible normal distributions}

Consider the previous example with the six stocks and three linear combinations. Everything is kept the same, except for the first two constraints, for which the variance

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 2 Option prices. The table reports statistics for prices of exotic options $O_{1}, O_{2}$, and $O_{3}$, computed for $M$ runs of CBRA. The statistics include mean, standard deviation, and percentiles of $1 \%, 10 \%, 50 \%$, $90 \%$, and $99 \%$}
\begin{tabular}{llllllll}
\hline & Mean & SD & $1 \%$ & $10 \%$ & $50 \%$ & $90 \%$ & $99 \%$ \\
\hline$O_{1}$ & 0.151 & 0.0084 & 0.136 & 0.141 & 0.149 & 0.162 & 0.173 \\
$O_{2}$ & 0.142 & 0.0023 & 0.138 & 0.140 & 0.142 & 0.145 & 0.148 \\
$O_{3}$ & 0.248 & 0.0035 & 0.238 & 0.243 & 0.248 & 0.252 & 0.254 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 3 Average correlations. For the case of incompatible constraints, this table reports pairwise correlations for the six stocks averaged across $M$ runs of CBRA}
\begin{tabular}{|l|l|l|l|l|l|l|}
\hline & $X_{1}$ & $X_{2}$ & $X_{3}$ & $X_{4}$ & $X_{5}$ & $X_{6}$ \\
\hline $X_{1}$ & 1.000 & 1.000 & 0.300 & 0.300 & 1.000 & 1.000 \\
\hline $X_{2}$ & 1.000 & 1.000 & 0.300 & 0.300 & 1.000 & 1.000 \\
\hline $X_{3}$ & 0.300 & 0.300 & 1.000 & -0.820 & 0.300 & 0.300 \\
\hline $X_{4}$ & 0.300 & 0.300 & -0.820 & 1.000 & 0.300 & 0.300 \\
\hline $X_{5}$ & 1.000 & 1.000 & 0.300 & 0.300 & 1.000 & 1.000 \\
\hline $X_{6}$ & 1.000 & 1.000 & 0.300 & 0.300 & 1.000 & 1.000 \\
\hline
\end{tabular}
\end{table}
becomes 6 instead of 10, that is:
$$
\begin{array}{ll}
X_{1}+X_{2}+X_{3}+X_{4} \sim N(0,6):=F_{\alpha^{1}}, & \alpha^{1}=[1,1,1,1,0,0], \\
X_{3}+X_{4}+X_{5}+X_{6} \sim N(0,6):=F_{\alpha^{2}}, & \alpha^{2}=[0,0,1,1,1,1] .
\end{array}
$$

The new constraints now imply the following restrictions on the average correlations:
$$
\begin{equation*}
\bar{\rho}\left(J_{1}\right)=0.167, \quad \bar{\rho}\left(J_{2}\right)=0.167, \quad \bar{\rho}\left(J_{3}\right)=0.6 . \tag{13}
\end{equation*}
$$

It is easy to check that the above constraints are mutually incompatible. Nevertheless, we can still use CBRA to find a joint distribution which minimizes the objective function $V=\sum_{k=1}^{K} \operatorname{var}\left(L_{k}\right)$. In this case, however, the solution will not satisfy the three constraints exactly. In fact, after $M=1000$ runs of CBRA, we find that the averages for sets $J_{1}-J_{3}$ are $0.2297,0.2297$, and 0.5051 , instead of those in (13). That is, the correlations for sets $J_{1}$ and $J_{2}$ are too high, while the correlations for set $J_{3}$ are too low compared to the constraints.

Table 3 reports the correlation matrix for the six stocks averaged across all $M$ runs. The average value of the objective function $V$ normalized by the number of states $n$ is now 0.135 , which is considerably larger than the value of 0.000072 obtained in the previous example. That is, the objective functions in the two examples differ by a factor of about 2000, which makes it easy for the method to distinguish the cases of compatible and incompatible constraints, even without a formal statistical test. Furthermore, Fig. 2 shows that the distributions of the pairwise correlations $\rho_{i j}$ are very extreme and almost degenerate: several correlations are equal to perfect 1 and, for every correlation, the difference between the 10th and 90th percentiles is close to zero. This is another clear sign that the constraints are incompatible and that the algorithm fails to find a proper joint distribution. Intuitively, the tension created by incompatible constraints causes the search algorithm to converge to a "corner solution".

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_09_30_b81faa5a74b6c1ea6646g-19.jpg?height=552&width=1178&top_left_y=185&top_left_x=171}
\captionsetup{labelformat=empty}
\caption{Fig. 2 CBRA Results for the Case of Incompatible Constraints. The left panel shows the histogram of the objective $V$ across $M$ runs of the algorithm. The dashed vertical line represents the sample mean. The right panel shows the 10th, 50th and 90th percentiles of distributions of pairwise correlations $\rho_{i j}$ generated by $M$ runs. The percentiles are shown along 15 vertical lines, corresponding to correlations $\rho_{12}, \rho_{13}, \ldots, \rho_{56}$. The grey circles are the sample means}
\end{figure}

Remark 7 Consider a strategy that buys the variance contracts on the first and second index and sells the variance contract on the third index, that is, the strategy at the maturity has the final payoff equal to $\left(X_{1}+X_{2}+X_{3}+X_{4}\right)^{2}+\left(X_{3}+X_{4}+X_{5}+\right. \left.X_{6}\right)^{2}-\left(X_{1}+X_{2}+X_{3}+X_{4}+X_{5}+X_{6}\right)^{2}$. As is well-known, the variance contract on a given underlying asset can be statically replicated by a particular portfolio of puts and calls, see Bakshi and Madan (2000). ${ }^{13}$ Thus, the proposed strategy can be implemented by trading in standard options on the three indices. Moreover, it is easy to verify that this strategy is an arbitrage opportunity, which is guaranteed to make a profit of at least 4 units.

In practice, a similar approach can be used to detect potential arbitrage opportunities. One starts with a number of "distributional constraints" coming from a set of linear combinations for which the distribution functions are known. Then one runs CBRA to obtain a candidate joint model and tests whether the constraints are satisfied in the statistical sense. Specifically, a number of statistical tests (such as, a distribution-free multivariate Kolmogorov-Smirnov one) can be used to check if the candidate model satisfies the constraints at a given confidence level. If the test is not rejected, then the candidate model can be seen as suitable and there is no arbitrage. Otherwise, one can conclude that the individual options are mispriced relative to the index options.

\section*{5 Conclusions}

We propose a new methodology for inferring a joint distribution among several variables consistent with the known distributions of these variables and of some linear

\footnotetext{
${ }^{13}$ More precisely, our arbitrage strategy involves trading in so-called simple variance swaps, see Martin (2017).
}
combinations thereof. An important application of this methodology is in option pricing, where this algorithm can be applied to extract a joint RND consistent with market prices of all available options on individual stocks and various indices. This modelfree approach can then be used to price path-independent multivariate options in an arbitrage-free fashion.

Our study also extends some results of Hobson et al. (2005) and Chen et al. (2008) on detecting arbitrage opportunities. These authors derive the highest possible price for a given basket option provided that prices of calls and puts on the single stocks are known, and derive a (static) arbitrage strategy when the true price is higher than this bound. Here, we aim at finding the most plausible model (and hence price) that is consistent with the available options. However, if no compatible market model can be found, this signals that potential arbitrage opportunities might exist.

The methodology we develop in this paper makes it possible to study various aspects of option implied dependence. Bondarenko and Bernard (2020) use it to provide new insights on the dependence among the nine industry sectors of the S\&P 500 index and on the correlation risk premium. In particular, they find the option implied dependence is highly non-normal, asymmetric, and time-varying. Furthermore, they study the correlation conditional on the market going down or up and find that the risk-premium for the down correlation is strongly negative, whereas it is positive for the up correlation.

We expect that the methodology presented in this paper will have many more empirical applications, which require the knowledge of the risk-neutral joint distribution, including model selection, arbitrage detection, and others. We leave these potential applications for future work.

\section*{References}

Aït-Sahalia, Y., \& Lo, A. W. (2000). Nonparametric risk management and implied risk aversion. Journal of Econometrics, 94(1), 9-51.
Avellaneda, M., \& Boyer-Olson, D. (2002). Reconstruction of volatility: pricing index options by the steepest descent approximation. In Courant Institute-NYU working paper.
Bakshi, G., \& Madan, D. (2000). Spanning and derivative security valuation. Journal of Financial Economics, 55(2), 205-238.
Ballotta, L., \& Bonfiglioli, E. (2016). Multivariate asset models using Lévy processes and applications. The European Journal of Finance, 22(13), 1320-1350.
Banz, R. W., \& Miller, M. H. (1978). Prices for state-contingent claims: Some estimates and applications. Journal of Business, 51(4), 653-672.
Bernard, C., Bondarenko, O., \& Vanduffel, S. (2018). Rearrangement algorithm and maximum entropy. Annals of Operations Research, 261(1-2), 107-134.
Bernard, C., \& McLeish, D. (2016). Algorithms for finding copulas minimizing convex functions of sums. Asia-Pacific Journal of Operational Research, 33(5), 1650040.
Bernard, C., Rüschendorf, L., \& Vanduffel, S. (2017). Value-at-risk bounds with variance constraints. Journal of Risk and Insurance, 84(3), 1539-6975.
Bondarenko, O. (2003). Estimation of risk-neutral densities using positive convolution approximation. Journal of Econometrics, 116(1), 85-112.
Bondarenko, O., \& Bernard, C. (2020). Option-implied dependence and correlation risk premium. Working paper. Available at SSRN: https://ssrn.com/abstract=3618705.
Boudt, K., Jakobsons, E., \& Vanduffel, S. (2018). Block rearranging elements within matrix columns to minimize the variability of the row sums. $4 O R, 16(1), 31-50$.
Breeden, D. T., \& Litzenberger, R. H. (1978). Prices of state-contingent claims implicit in option prices. Journal of Business, 51(4), 621-651.

Chen, X., Deelstra, G., Dhaene, J., \& Vanmaele, M. (2008). Static super-replicating strategies for a class of exotic options. Insurance: Mathematics and Economics, 42(3), 1067-1085.
Cont, R., \& Deguest, R. (2013). Equity correlations implied by index options: Estimation and model uncertainty analysis. Mathematical Finance, 23(3), 496-530.
Cramér, H. (1946). Mathematical methods of statistics (PMS-9) (Vol. 9). Princeton: Princeton University Press.
De March, H. (2018). Entropic approximation for multi-dimensional martingale optimal transport, arXiv preprint arXiv:1812.11104.
Embrechts, P., Puccetti, G., \& Rüschendorf, L. (2013). Model uncertainty and VaR aggregation. Journal of Banking \& Finance, 37(8), 2750-2764.
Figlewski, S. (2018). Risk-neutral densities: A review. Annual Review of Financial Economics, 10, 329-359.
Gaffke, N., \& Rüschendorf, L. (1981). On a class of extremal problems in statistics. Optimization, 12(1), 123-135.
Galichon, A. (2018). Optimal transport methods in economics. Princeton: Princeton University Press.
Haus, U.-U. (2015). Bounding stochastic dependence, joint mixability of matrices, and multidimensional bottleneck assignment problems. Operations Research Letters, 43(1), 74-79.
Henry-Labordere, P. (2017). Model-free hedging: A martingale optimal transport viewpoint. Amsterdam: CRC Press.
Hobson, D., Laurence, P., \& Wang, T.-H. (2005). Static-arbitrage upper bounds for the prices of basket options. Quantitative Finance, 5(4), 329-342.
Hoeffding, W. (1940). Masstabinvariante korrelationstheorie. Schriften des Mathematischen Instituts und Instituts fur Angewandte Mathematik der Universitat Berlin, 5, 181-233.
Jackwerth, J. C., \& Rubinstein, M. (1996). Recovering probability distributions from option prices. Journal of Finance, 51(5), 1611-1631.
Jaynes, E. T. (2003). Probability theory: the logic of science. Cambridge University Press.
Jourdain, B., \& Sbai, M. (2012). Coupling index and stocks. Quantitative Finance, 12(5), 805-818.
Justel, A., Pea, D., \& Zamar, R. (1997). A multivariate Kolmogorov-Smirnov test of goodness of fit. Statistics and Probability Letters, 35(3), 251-259.
Lévy, P. (1926). Calcul des probabilités.
Loregian, A., Ballotta, L., Fusai, G., Perez, M. F. (2018). Estimation of multivariate asset models with jumps, forthcoming in Journal of Financial and Quantitative Analysis, pp. 1-60.
Martin, I. (2017). What is the expected return on the market? The Quarterly Journal of Economics, 132(1), 367-433.
Peyré, G., \& Cuturi, M. (2019). Computational optimal transport. Open source book.
Puccetti, G., \& Rüschendorf, L. (2012). Computation of sharp bounds on the distribution of a function of dependent risks. Journal of Computational and Applied Mathematics, 236(7), 1833-1840.
Puccetti, G., Wang, B., \& Wang, R. (2012). Advances in complete mixability. Journal of Applied Probability, 49(2), 430-440.
Ross, S. A. (1976). Options and efficiency. Quarterly Journal of Economics, 90(1), 75-89.
Rubinstein, M. (1994). Implied binomial trees. Journal of Finance, 49(3), 771-818.
Stutzer, M. (1996). A simple nonparametric approach to derivative security valuation. Journal of Finance, 51(5), 1633-1652.
Wang, B., \& Wang, R. (2011). The complete mixability and convex minimization problems with monotone marginal densities. Journal of Multivariate Analysis, 102(10), 1344-1360.
Wang, B., \& Wang, R. (2016). Joint mixability. Mathematics of Operations Research, 41(3), 808-826.

Publisher's Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.