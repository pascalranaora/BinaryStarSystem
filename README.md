# 🌌 The Nakamoto VECM Model & Roche Lobe Binary Star System

> **A Stochastic-Relativistic Thermodynamic Projection Model for Bitcoin Adoption**

[![Dashboard Preview](https://img.shields.io/badge/Status-Active-brightgreen)](#)
[![Python](https://img.shields.io/badge/Backend-Python_3.9+-blue)](#)
[![JS](https://img.shields.io/badge/Frontend-Three.js_%7C_Chart.js-yellow)](#)
[![Math](https://img.shields.io/badge/Math-Stochastic_Calculus_%7C_VECM-purple)](#)

This repository contains a quantitative mathematical model and an interactive web dashboard that projects Bitcoin's price and market capitalization by treating the global macroeconomic economy as a **Binary Star System**.

Moving beyond standard Ordinary Least Squares (OLS) regressions, this v34 model utilizes a **Vector Error Correction Model (VECM)** combined with **Langevin Stochastic Differential Equations (SDE)** and **Chaos Theory** to prove that Bitcoin and Global Fiat Liquidity are *cointegrated*—bound together by gravitational mathematics over the long term.

## 🚀 Live Demo

[View Live Dashboard](https://pascalranaora.github.io/BinaryStarSystem/)

---

## 🧠 The Core Philosophy & Physics

The model rejects the premise that Bitcoin's price is purely random or driven solely by speculative noise. Instead, it posits that Bitcoin acts as a super-dense **White Dwarf** slowly siphoning capital (mass) from a highly inflationary, expanding **Red Giant** (Fiat M2 Supply).

This mass transfer is modeled using actual astrophysics and thermodynamics:

### I. Relativistic Volatility (Lorentz Factor)
$$\gamma_{Lorentz} = \exp\left(\frac{\min(VIX_t, 80) - 20}{40}\right)$$
Simulates spacetime compression. High VIX shrinks the 'distance' between assets, forcing tighter cointegration and violently increasing gravitational pull (correlations trending to 1 during market panics).

### II. Mass Transfer Velocity (Bernoulli Overflow)
$$\dot{M}_{Acc} = \left(\frac{\max(CPI_{yoy} - 2.0, 0)\cdot M2_{Vol}}{Rate_{FedFunds}}\right) \cdot \gamma_{Lorentz}$$
Calculates the velocity of liquidity escaping the expanding 'Red Giant' (M2), multiplied by the Lorentz Factor during volatile regimes.

### III. The Cointegrating Vector (Gravitational Equilibrium)
$$Logit(\rho)_t = \beta_0 + \beta_1 \ln(SF)_t + \beta_2 \ln(V)_t + \beta_3 \ln(H)_t + \beta_4 Density_{WD,t} + \beta_5 \dot{M}_{Acc,t} + \epsilon_t$$
Defined in the VECM framework, this vector represents the stable, long-run path where Bitcoin’s TAM absorption ($\rho$) is balanced against fundamental scarcity, internal heat/energy (Hashrate $H$), and network power.

### IV. Langevin SDE (Stochastic Orbital Accretion)
$$d(\dot{M}_{Acc}) = \mu(M_{ratio}, \gamma_{Lorentz}) dt + \sigma(\gamma_{Lorentz}) dW_t$$
Replaces deterministic Keplerian orbits. The accretion force follows a stochastic differential equation driven by dynamic mass ratios and random market entropy ($dW_t$).

### V. Phase Space Chaos (Lyapunov Exponent)
$$\lambda = \lim_{t \to \infty} \lim_{\delta Z_0 \to 0} \frac{1}{t} \ln \frac{|\delta Z(t)|}{|\delta Z_0|}$$
If the Largest Lyapunov Exponent ($\lambda > 0$), the binary orbit is in deterministic chaos. Initial minor divergences in the macro state lead to exponentially divergent short-term price paths.

---

## 📊 Dashboard Features

### 1. Interactive 3D Binary Star Simulation (Three.js)
A real-time, WebGL-rendered visual representation of the mass transfer. As you scrub through time, you can watch the White Dwarf (BTC) accrete mass from the Red Giant (M2) based on the calculated Bernoulli overflow equations.

### 2. VECM Cointegration & Monte Carlo Projection
A visualization plotting the actual historical price of Bitcoin against the **VECM Long-Term Gravitational Equilibrium**, projecting forward to the next halving epoch using a 5,000-path Langevin SDE simulation.

### 3. Actionable Z-Score Signal
A live mapping of the Relativistically Adjusted Error Correction Term ($ECT_{adj}$), providing a purely data-driven oscillator for generational accumulation (Extreme Compression < -2) and distribution (Roche Lobe Overflow > +2) zones.

### 4. Bilingual Support
The entire interface supports instant toggling between **English** and **French** (`i18n` integration).

---

## 📂 Repository Structure

* `index.html` : The main frontend dashboard (Tailwind CSS, Chart.js, Three.js, MathJax).
* `v34.py` : The backend Python quantitative engine. Dynamically fetches data (FRED API, Blockchain.info, Yahoo Finance), runs the Johansen/ADF cointegration tests, calculates Lyapunov exponents, executes the Langevin Monte Carlo projection, and outputs the JSON.
* `v33_dashboard_data.json` : The compiled historical and projected physics/price data consumed by the frontend.

---

## ⚙️ How to Run Locally

### 1. The Frontend Dashboard
Because the dashboard fetches a local JSON file, you must serve it over a local web server to avoid browser CORS errors.
```bash
python -m http.server 8000
```
Then navigate to `http://localhost:8000` in your browser.

### 2. The Python Engine
To run the model and generate a fresh projection / JSON data payload:

1. Install dependencies:
```bash
pip install pandas numpy statsmodels scikit-learn matplotlib yfinance tvdatafeed tqdm
```
2. Insert your FRED API key into `v34.py` (`FRED_API_KEY = 'your_key_here'`).
3. Run the engine:
```bash
python v34.py
```
This will output `Nakamoto_RocheLobe_VECM.pdf` (a high-resolution quant report) and update `v33_dashboard_data.json` for the web UI.

---

## ⚠️ Scientific Disclaimer

Past performance is not indicative of future results. The Nakamoto/Fiat Binary Star System model is a theoretical and scientific experiment designed to study macroeconomic thermodynamics and network effects. It is **strictly not financial or investment advice**. Cryptocurrencies and financial markets are highly volatile. Always conduct your own independent research.

