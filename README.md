```markdown
# 🌌 The Nakamoto V33 Resonant Master: Bitcoin-Fiat Binary Star System

> **A Thermodynamic & Statistical Projection Model for Bitcoin Adoption**

![Dashboard Preview](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Backend-Python_3.9+-blue) ![JS](https://img.shields.io/badge/Frontend-Three.js_|_Chart.js-yellow)

This repository contains the **Nakamoto V33 Resonant Master**, an interactive analytical dashboard and mathematical model that projects Bitcoin's price and market capitalization by treating the global economy as a **Binary Star System**. 

The model uses a thermodynamic analogy: as Central Banks expand the M2 money supply (the inflating *Red Giant*), capital seeks a dense, absolutely scarce store of value (Bitcoin, the *White Dwarf*). The resulting liquidity transfer is modeled as an accretion disk, mathematically driven by a generalized Metcalfe's Law and Scale-Free Network dynamics.

## 🚀 Live Demo
*(If you are hosting the HTML file on GitHub Pages, Vercel, or Netlify, put the link here: `[View Live Dashboard](https://pascalranaora.github.io/BinaryStarSystem/)`)*

---

## 🧠 The Core Philosophy & Mathematics

The V33 model rejects the idea that Bitcoin's price is purely speculative. Instead, it posits that Bitcoin is a mathematical consequence of fiat expansion. The model relies on the following core tensors (variables):

1. **The Scarcity Piston (`log_SF`):** Driven by the Nakamoto consensus and the 4-year Halving cycles (Stock-to-Flow).
2. **Metcalfe Network Force (`Network_Power`):** A proxy for network utility, measured by on-chain unique addresses and, recently, ETF trading velocity (The $V_{eff}$ tensor).
3. **Hashrate Premium (`H_Accumulated`):** The delayed accumulation and capitulation of miner energy.
4. **Liquidity Momentum (`Liq_Momentum`):** Driven by Federal Reserve balance sheet metrics (WALCL - WTREGEN).
5. **Global RORO State (`Global_RORO_State`):** An exogenous Risk-On / Risk-Off oscillator.

### The State Equation
The model caps Bitcoin's Total Addressable Market (TAM) at a multiple of the Global M2 Supply. The absorbed market share ($\rho$) is calculated using a Logistic Regression (Logit transformation) against the core tensors:

$$TAM = M2 \times 5.0$$
$$\rho = \frac{Cap_{BTC}}{TAM} \quad \Rightarrow \quad Logit(\rho) = \ln\left(\frac{\rho}{1 - \rho}\right)$$
$$Logit(\rho) = \beta_0 + \beta_1 \ln(SF) + \beta_2 H_{acc} + \beta_3 NetPower + ...$$

---

## 🛠️ Features

### 1. The Interactive 3D Viewscreen
A complete `Three.js` physics simulation of the Thermodynamic Transfer:
* **The Red Giant (M2):** Uses a custom WebGL Shader to simulate a boiling plasma surface that pulses (Stellar Flares) during 4-year liquidity injection cycles.
* **The White Dwarf (BTC):** Features a dynamic, spinning accretion disk that grows as Bitcoin's market share relative to M2 increases.
* **The Accretion Ray:** A volumetric, glowing plasma stream physically connecting the two bodies, representing the real-time transfer of global liquidity.

### 2. Principal Component Analysis (PCA) Chart
A `Chart.js` visualization plotting the actual historical price of Bitcoin against the **V33 Model Median Projection** (projecting forward to 2028). The projection utilizes a Log-Space Shift to ensure smooth, continuous forecasting from the present day.

### 3. Tensor Dynamics Chart
A visual breakdown of the underlying forces (the $\beta$-weighted parameters) driving the State Equation over time, including the Scarcity Piston, Metcalfe Force, and Global Liquidity.

### 4. Bilingual Support
The entire interface supports instant toggling between **English** and **French**.

---

## 📂 Repository Structure

* `index.html` : The main frontend dashboard (HTML/CSS/JS) containing the Tailwind UI, Chart.js graphs, and the injected Three.js 3D environment.
* `v33_dashboard_data.json` : The compiled historical and projected price data.
* `v33.py` (or similar name) : The backend Python engine. This script fetches data (FRED API, Blockchain.info, Yahoo Finance), performs the OLS regression audit, runs the 5000-path Monte Carlo GARCH projection, and outputs the JSON data.

---

## ⚙️ How to Run Locally

### 1. The Frontend Dashboard
Because the dashboard fetches a local JSON file (`v33_dashboard_data.json`), you cannot simply double-click the `index.html` file due to browser CORS (Cross-Origin Resource Sharing) security policies. You must serve it over a local web server.

If you have Python installed, open your terminal/command prompt in the project directory and run:
```bash
python -m http.server 8000
```
Then open your browser and navigate to `http://localhost:8000`.

### 2. Updating the Model (Running the Python Backend)
To generate fresh data for the dashboard based on today's market metrics:

1. Install the required Python dependencies:
   ```bash
   pip install pandas numpy statsmodels scikit-learn matplotlib yfinance tvdatafeed tqdm requests
   ```
2. Insert your FRED API Key into the Python script.
3. Run the script:
   ```bash
   python v33.py
   ```
This will run the physics audit, execute the Monte Carlo simulation, and automatically overwrite `v33_dashboard_data.json` with the new trajectory. Refresh your HTML dashboard to see the updated charts.

---

## 📊 Statistical Validation
The V33 OLS model currently operates with an $R^2$ of **0.975**. It categorically rejects the null hypothesis. The model demonstrates that Bitcoin's growth is not random noise, but a highly correlated response to the devaluation of the M2 supply governed by network effects.

*The Singularity is not a prediction. It is an equation.*

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
```

### Next Steps for you:
1. Copy this markdown text.
2. Go to your repository on GitHub.
3. Create a new file named `README.md` (or edit the existing one).
4. Paste the text.
5. *(Optional)* If you decide to host the HTML file on GitHub Pages so anyone can view it online, make sure to update the `[View Live Dashboard]` link in the template!# BinaryStarSystem
