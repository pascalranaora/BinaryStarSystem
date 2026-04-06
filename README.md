# 🌌 The Nakamoto VECM Model & Roche Lobe Binary Star System

> **A Thermodynamic & Statistical Projection Model for Bitcoin Adoption**

[![Dashboard Preview](https://img.shields.io/badge/Status-Active-brightgreen)](#)
[![Python](https://img.shields.io/badge/Backend-Python_3.9+-blue)](#)
[![JS](https://img.shields.io/badge/Frontend-Three.js_%7C_Chart.js-yellow)](#)

This repository contains an interactive analytical dashboard and a quantitative mathematical model that projects Bitcoin's price and market capitalization by treating the global macroeconomic economy as a **Binary Star System**.

Moving beyond standard (and often flawed) Ordinary Least Squares (OLS) regressions, this model utilizes a **Vector Error Correction Model (VECM)** to mathematically prove that Bitcoin and Global Fiat Liquidity are *cointegrated*—bound together by gravitational mathematics over the long term.

## 🚀 Live Demo

[View Live Dashboard](https://pascalranaora.github.io/BinaryStarSystem/)

---

## 🧠 The Core Philosophy & Physics

The model rejects the premise that Bitcoin's price is purely random or driven solely by speculative noise. Instead, it posits that Bitcoin acts as a super-dense **White Dwarf** slowly siphoning capital (mass) from a highly inflationary, expanding **Red Giant** (Fiat M2 Supply).

This mass transfer is modeled using actual fluid dynamics and thermodynamics:

1. **White Dwarf Density (`WD_Density`):** Calculated as `(Total Supply - Active Tx Vol) / Total Supply`. This represents the percentage of the network's mass that has crystallized into cold storage, increasing the network's gravitational pull.
2. **Roche Lobe Overflow (`Accretion_Force`):** Based on Bernoulli fluid dynamics. When Central Banks artificially suppress interest rates while inflation runs hot, fiat liquidity "overflows" its gravitational bounds and falls toward the denser asset.
3. **Scarcity Piston (`log_SF`):** The mathematically programmed core collapse (Halving) that periodically resets the baseline density.
4. **Metcalfe Velocity (`log_V`):** Network demand driven by unique on-chain entities.

### The Thermodynamic State Equations

Instead of projecting price to infinity, the model caps Bitcoin's Total Addressable Market (TAM) relative to the Global M2 Supply. The absorption rate ($\rho$) is governed by an Error Correction Term (ECT) that forces the asset back to its macroeconomic equilibrium:

$$Density_{WD} = \frac{Supply_{Total} - Vol_{Tx}}{Supply_{Total}}$$

$$\dot{M}_{Accretion} = \frac{(CPI_{yoy} - 2.0) \times M2_{Volume}}{Rate_{FedFunds}}$$

$$ECT_{Z-Score} = Logit(\rho) - \sum_{i=1}^{n} \beta_i X_{i}$$

When the $ECT_{Z-Score}$ exceeds +2, the system experiences "Severe Roche Lobe Overflow" (overvaluation) and must mean-revert downwards. When it drops below -2, the asset is undergoing extreme, undervalued compression.

---

## 🛠️ Features

### 1. The Interactive 3D Viewscreen
A complete `Three.js` physics simulation of the Thermodynamic Transfer:
* **The Red Giant (M2):** Uses a custom WebGL Shader to simulate boiling plasma that pulses during liquidity injection cycles.
* **The White Dwarf (BTC):** Features an accretion disk whose mass transfer velocity accelerates based on the actual historical data of the network.
* **The Eddington Limit:** The visual and mathematical cap (currently modeled at 10% of M2) that prevents the projection from resulting in infinite physical growth.

### 2. VECM Cointegration Chart
A visualization plotting the actual historical price of Bitcoin against the **VECM Long-Term Gravitational Equilibrium**, projecting forward 2 Years using a Monte Carlo GARCH simulation (5,000 paths).

### 3. Actionable Z-Score Signal
A live mapping of the Error Correction Term, providing a purely data-driven oscillator for generational buying (Green) and selling (Red) zones.

### 4. Bilingual Support
The entire interface supports instant toggling between **English** and **French**.

---

## 📂 Repository Structure

* `index.html` : The main frontend dashboard (HTML/CSS/JS) containing the Tailwind UI, Chart.js graphs, and the injected Three.js 3D environment.
* `v33_dashboard_data.json` : The compiled historical and projected price data (Output of the Python Engine).
* `v33.py` : The backend Python quantitative engine. This script dynamically fetches data (FRED API, Blockchain.info, Yahoo Finance), runs the Augmented Dickey-Fuller (ADF) cointegration tests, executes the 5000-path Monte Carlo GARCH projection, and outputs the JSON data.

---

## ⚙️ How to Run Locally

### 1. The Frontend Dashboard
Because the dashboard fetches a local JSON file, you must serve it over a local web server to avoid browser CORS errors.
Run this in your terminal:
```bash
python -m http.server 8000