# Unified Conditional Binary Martingale (UCBM): a maturity-aware model for binary prices

Gregory Sharma (working ideas) | with the help of ChatGPT

## Abstract

I present a single SDE framework for **any** binary (cash-or-nothing) price process $B_t\in(0,1)$ with fixed maturity $T$: elections, weather thresholds, in-play sports win probabilities, "happens-by-$T$" hazards, and financial digitals.

---

## 1. Setup and notation

* $T>0$: maturity (revelation time).
* $B_t\in(0,1)$: **forward** probability of the event (for financial digitals: arbitrage price divided by the $T$-bond $P(t,T)$).
* Probit gauge: $U_t:=\Phi^{-1}(B_t)$, so $B_t=\Phi(U_t)$. Here $\Phi,\phi$ are standard normal CDF/PDF.
* $W_t$: Brownian motion under the forward measure.
* Jump types $j=1,\dots,J$: counting processes $N^{(j)}_t$ with intensities $\lambda^{(j)}_t$; compensated martingales $M^{(j)}_t:=N^{(j)}_t-\int_0^t\lambda^{(j)}_s ds$.
* State $\mathcal S_t$: any observable drivers (score lead, weather state, etc.).

---

## 2. UCBM model

### 2.1 Dynamics in the probit gauge

$$
\boxed{
dU_t=\tfrac12\,\nu_t^2\,U_{t-}\,dt\;+\;\nu_t\,dW_t\;+\;\sum_{j=1}^J \zeta_j(t,\mathcal S_{t-})\,dM^{(j)}_t,\qquad B_t=\Phi(U_t).
}
\tag{UCBM}
$$

* $\nu_t\ge 0$: information clock (time/state-dependent).
* $\zeta_j(t,\mathcal S)$: jump size in $U$-space for event type $j$ (score, ruling, etc.).

### 2.2 Full Itô–Lévy dynamics of $B$

Applying Itô–Lévy to $B_t=\Phi(U_t)$ (using $\phi'(u)=-u\phi(u)$):

$$
\boxed{
dB_t=\underbrace{\nu_t\,\phi(U_{t-})\,dW_t}_{\text{diffusion}}
\;+\;\sum_{j=1}^J \underbrace{\big[\Phi(U_{t-}+\zeta_j)-\Phi(U_{t-})\big]}_{\Delta B^{(j)}_t}\,dM^{(j)}_t,
}
\tag{B-dyn}
$$

i.e., **zero predictable drift** under the pricing/forward measure.

**Instantaneous quadratic variation**

$$
\boxed{
d\langle B\rangle_t=\nu_t^2\,\phi\!\big(\Phi^{-1}(B_{t-})\big)^2\,dt
+\sum_{j=1}^J \big(\Delta B^{(j)}_t\big)^2\,\lambda^{(j)}_t\,dt.
}
\tag{QV}
$$

### 2.3 Maturity (revelation) condition

If the outcome is revealed at $T$ (so $B_T\in\{0,1\}$), enforce

$$
\boxed{\quad\int_0^T \nu_s^2\,ds=\infty\quad}
\tag{Revelation}
$$

(e.g., choose $\nu_t\sim c/\sqrt{T-t}$). Then $U_t\to\pm\infty$ and $B_t\to\{0,1\}$ as $t\uparrow T$.

---

## 3. Special cases

### 3.1 Taleb’s election model (continuous info, no jumps)

Taleb maps a shadow $X_t$ (OU-type) to a bounded estimator and prices a digital. In the probit gauge this corresponds to the deterministic clock

$$
\boxed{\ \nu_{\text{Taleb}}(t)=\frac{\sqrt{2}\,\sigma}{\sqrt{e^{2\sigma^2 (T-t)}-1}}
\sim \frac{1}{\sqrt{T-t}}\quad (t\uparrow T).\ }
\tag{Taleb-clock}
$$

Plugging $\nu_{\text{Taleb}}$ into (B-dyn) reproduces Taleb’s “uncertainty pulls $B$ toward 0.5” behavior.

### 3.2 Black–Scholes digital (cash-or-nothing call)

Under $Q$: $dS_t=rS_tdt+\sigma S_tdW_t$. The **forward** probability

$$
B(t,S)=\Phi(d_2),\qquad d_2=\frac{\ln(S/K)+(r-\tfrac12\sigma^2)\tau}{\sigma\sqrt{\tau}},\;\tau=T-t.
$$

Itô + BS PDE gives the **exact** SDE

$$
\boxed{\,dB_t=\frac{\phi(\Phi^{-1}(B_t))}{\sqrt{T-t}}\,dW_t^{Q},\,}
\tag{BS-dB}
$$

i.e., (UCBM) with **clock** $\nu_t=1/\sqrt{T-t}$, no jumps.
Interestingly, the **instantaneous variance of the probability**,
$\phi(\Phi^{-1}B_t)^2/(T-t)$, is independent of $\sigma$ and $r$.

### 3.3 In-play sports (jump layer + optional diffusion)

Let $D_t\in\mathbb Z$ be the lead; define

$$
b(t,d):=\mathbb P(D_T>0\mid D_t=d).
$$

Backward (Markov chain) equation:

$$
\boxed{
\partial_t b+\lambda^A(t,d)\,[b(t,d{+}1)-b(t,d)]
+\lambda^B(t,d)\,[b(t,d{-}1)-b(t,d)]=0,\ \ b(T,d)=\mathbf 1_{\{d>0\}}.
}
\tag{Sports-PDE}
$$

Price: $B_t=b(t,D_t)$. Embed score jumps into $U$-space:

$$
\zeta^A(t,d)=\Phi^{-1}\!\big(b(t,d{+}1)\big)-\Phi^{-1}\!\big(b(t,d)\big),\quad
\zeta^B(t,d)=\Phi^{-1}\!\big(b(t,d{-}1)\big)-\Phi^{-1}\!\big(b(t,d)\big),
$$

with intensities $\lambda^{A/B}(t,d)$. Optionally include a small $\nu_t$ for micro-info. Then (B-dyn) holds with these $\zeta$’s and $\lambda$’s.

**Skellam/normal approximation (time-only pace):**

$$
b(t,D_t)\approx \Phi\!\Big(z_t\Big),\quad
z_t:=\frac{D_t+\mu_A-\mu_B}{\sqrt{\mu_A+\mu_B}},\quad
\mu_i(t):=\int_t^T\lambda^i(u)du,
$$

giving $\Delta_\pm\approx \pm \phi(z_t)/\sqrt{\mu_A+\mu_B}$ and
$\mathrm{Var}(dB_t)\approx \phi(z_t)^2/(T-t)\,dt$.

### 3.4 “Happens-by-$T$?” hazards (pure jump)

Let event time $\tau$ have Cox intensity $\lambda_t$. While no event:

$$
B_t=1-\exp\!\Big(-\!\int_t^T\lambda_s ds\Big),
$$

and upon the event $B$ jumps to $1$. In martingale form:

$$
\boxed{\,dB_t=(1-B_{t-})\,dM_t,\qquad dM_t=dN_t-\lambda_t dt.}
$$

This is the UCBM jump-only limit ($\nu\equiv 0$).

---

## 4. Estimation & filtering

### 4.1 Always use **forward probabilities**

For tradable digitals, I work with $B_t=\Pi_t/P(t,T)$. Under the $T$-forward measure, $B_t$ is a martingale—matching (B-dyn).

### 4.2 Local-variance clock (nonparametric)

On bins without identified jumps,

$$
\Delta B_t\approx \mathcal N\!\Big(0,\;\nu_t^2\,\phi(\Phi^{-1} B_t)^2\,\Delta t\Big)
\ \Rightarrow\ 
\boxed{\ \widehat{\nu}_t^{\,2}=\frac{\mathrm{Var}(\Delta B_t)}{\phi(\Phi^{-1} B_t)^2\,\Delta t}\quad(\text{smooth}).\ }
\tag{LV-est}
$$

Jump-robust block estimator (bipower variation):

$$
BV=\tfrac{\pi}{2}\sum_{i\ge 2}|\Delta B_i||\Delta B_{i-1}|
\approx \int \nu_t^2\,\phi(\Phi^{-1}B_t)^2\,dt
\ \Rightarrow\
\widehat{\bar\nu}^{\,2}=\frac{BV}{\sum \phi(\Phi^{-1}B_{t_i})^2\,\Delta t}.
$$

### 4.3 Parametric clocks via time-change (Gaussian MLE)

For $\nu(t;\theta)$, define the clock

$$
\Lambda(t):=\int_0^t \nu(s;\theta)^2\,ds.
$$

Then

$$
dU_t=\tfrac12 U_{t-}\,d\Lambda(t)+dW^*_{\Lambda(t)}\quad
(\text{Brownian in clock time}),
$$

giving exact Gaussian transitions in $\Lambda$ and a clean MLE for $\theta$ (e.g., $\sigma$ in the Taleb clock).

### 4.4 Jumps: sizes and intensities

* Jump sizes in $U$: $\widehat{\zeta}_j\approx \Delta U=\Phi^{-1}(B_t)-\Phi^{-1}(B_{t-})$ at event type $j$, optionally conditioned on state $\mathcal S$.
* Intensities: fit $\lambda^{(j)}_t$ by (marked) Poisson regression on $(t,\mathcal S_t)$.

### 4.5 Filtering when state is latent/noisy

Hidden state $\Theta_t=(U_t,\{\log\lambda^{(j)}_t\},\ldots)$; observations:

* Price: $Y_t=\Phi(U_t)+\epsilon_t$ (microstructure noise).
* Event marks/times: update $\lambda^{(j)}$.

Case for some unscented / extended Kalman filter? Or particle filter (nonlinear measurement $\Phi$)?

---

## 5. Market making (Avellaneda–Stoikov for binaries)

Let inventory $q$, risk aversion $\gamma$, order-arrival decay $k$. Replace the mid variance in A–S by $\sigma_B^2(t)$ from (QV):

$$
\boxed{
\sigma_B^2(t)=\nu_t^2\phi(\Phi^{-1}B_t)^2+\sum_j (\Delta B^{(j)}_t)^2\lambda^{(j)}_t.
}
$$

Then the usual A–S forms translate to probabilities:

* **Reservation price**

$$
\boxed{\ r_t = B_t - q\,\gamma\,\sigma_B^2(t)\,(T-t).\ }
$$

* **Optimal half-spread (symmetric baseline)**

$$
\boxed{\ \delta_t^{*}\approx \frac{1}{k}+\frac12\,\gamma\,\sigma_B^2(t)\,(T-t).\ }
$$

Skew quotes when jump risk is one-sided (e.g., $\lambda^A\gg\lambda^B$) by shifting $r_t$ and widening on the exposed side.

---

## 6. Sports specifics (optional layer)

Given $\lambda^{A/B}(t,d)$ and lattice $b(t,d)$:

* **Center**: $B_t=b(t,D_t)$.
* **Jump sizes**: $\Delta_+(t)=b(t,D_{t-}{+}1)-b(t,D_{t-})$, $\Delta_-(t)=b(t,D_{t-}{-}1)-b(t,D_{t-})$.
* **$U$-jumps**: $\zeta^{A/B}=\Phi^{-1}(b(t,d\pm1))-\Phi^{-1}(b(t,d))$.
* **Conditional no-jump slope**: $\partial_t b=-\lambda^A\Delta_+-\lambda^B\Delta_-$.

These drop directly into (B-dyn) and (QV).

---

## 7. Simulation (two correct ways)

1. **Gauge simulation**: simulate $U$ via (UCBM) with maturity-consistent $\nu_t$ (e.g., $c/\sqrt{T-t}$ or $\nu_{\text{Taleb}}$) and state-dependent jumps; map $B_t=\Phi(U_t)$. Paths collapse to $\{0,1\}$ at $T$.

2. **Underlying-first**: simulate the context (e.g., GBM $S_t$, score process $D_t$, hazard), then compute $B_t$ from the closed form/lattice:

   * BS digital: $B=\Phi(d_2)$ ⇒ (BS-dB) holds.
   * Sports: $B=b(t,D_t)$ from the PDE/lattice.

*(Note: A toy diffusion $dB=\kappa \phi(\Phi^{-1}B)\,dW$ **without** a time-dependent clock does **not** enforce maturity.)*

---

## 8. Practical calibration checklist

1. Convert tradable digital price $\Pi_t$ to **forward probability** $B_t=\Pi_t/P(t,T)$.
2. Identify jump bins (scores/news).
3. **Clock $\nu_t$** from no-jump bins via (LV-est) (smoothed); optionally fit a parametric clock (Taleb/bridge) by time-change MLE.
4. **Jump sizes** $\widehat{\zeta}_j$ from $\Delta U$; **intensities** $\widehat{\lambda}^{(j)}$ by Poisson regression on state.
5. Compute **instantaneous risk** $\sigma_B^2(t)$ via (QV); plug into A–S for reservation price and spreads.
6. Validate: (i) martingale tests on $B$; (ii) realized vs model $d\langle B\rangle$ vs $B$; (iii) jump magnitudes vs lattice; (iv) out-of-sample Brier/log score; (v) MM backtest (PnL, inventory variance, fills).

---

## 9. What's new

* A single, maturity-aware SDE for binary prices where **boundedness + martingality** are guaranteed and **maturity** is enforced by the information clock.
* **Unification**: Taleb’s elections and Black–Scholes digitals are specific **clocks**; sports and hazards arise by adding calibrated **jump layers**.
* **Estimation/filters** and a direct **market-making plug-in** via $\sigma_B^2(t)$.

---

## 10. Appendix: Itô–Lévy with jumps

Let $dU_t=a_t dt + b_t dW_t + \sum_j \zeta_j(t)\,dN^{(j)}_t$ and $f\in C^2$. Then

$$
df(U_t)=f'(U_{t-})\,dU_t+\tfrac12 f''(U_{t-})\,b_t^2\,dt
+\sum_j\big(f(U_{t-}{+}\zeta_j)-f(U_{t-})-f'(U_{t-})\zeta_j\big)\,dN^{(j)}_t.
$$

With compensated $dM^{(j)}$ and $f=\Phi$,

$$
df=\Big[\phi(U_{t-})a_t+\tfrac12\phi'(U_{t-})b_t^2+\sum_j(\Delta\Phi_j-\phi(U_{t-})\zeta_j)\lambda^{(j)}_t\Big]dt
+\phi(U_{t-})b_t\,dW_t+\sum_j \Delta\Phi_j\,dM^{(j)}_t.
$$

Choosing $a_t=\tfrac12\,\nu_t^2 U_{t-}$ (and using compensated jumps) gives the driftless form (B-dyn).

---

### Notation glossary

* $\Phi,\phi$: standard normal CDF/PDF; $\Phi^{-1}$ probit.
* $U_t=\Phi^{-1}B_t$: probability gauge.
* $\nu_t$: information clock (controls continuous information flow).
* $\zeta_j$: jump size in $U$ for event type $j$; $\Delta B^{(j)}=\Phi(U_{-}{+}\zeta_j)-\Phi(U_{-})$.
* $\lambda^{(j)}_t$: jump intensity; $M^{(j)}$ compensated counting martingale.
* $b(t,d)$: sports lattice win-probability; $\Delta_\pm=b(t,d\pm1)-b(t,d)$.
