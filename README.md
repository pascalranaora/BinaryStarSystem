# 🌌 Nakamoto Astroéconophysique & Loi de Puissance

> **Modèle de Projection stochastique-relativiste basé sur la thermodynamique des réseaux et la cointégration macroéconomique.**

[![Status](https://img.shields.io/badge/Discipline-Astroéconophysique-blueviolet)](#)
[![Math](https://img.shields.io/badge/Math-SDE_Langevin_%7C_MCMC-blue)](#)
[![Logic](https://img.shields.io/badge/Core-Santostasi_Power_Law-brightgreen)](#)

Ce dépôt contient le moteur quantique et le tableau de bord interactif du modèle **Nakamoto v33**. Ce système traite l'économie mondiale comme un système stellaire binaire où Bitcoin agit comme une **Naine Blanche** hyper-dense absorbant la masse (liquidité) d'une **Géante Rouge** mourante (le système monétaire Fiat).

## 🚀 La Révolution v33 : L'Approche Hybride

Contrairement aux modèles linéaires classiques, la v33 fusionne deux piliers de la science quantitative :

1.  **L'Orbite de Santostasi (Noyau Thermodynamique) :** Le modèle extrait nativement l'exposant fractal de croissance ($P \sim t^{5.48}$). C'est l'axe de gravité déterministe à long terme.
2.  **L'Onde de Liquidité Mondiale (Accrétion PCA) :** Utilisation de l'Analyse en Composantes Principales sur les bilans de la Fed, BCE, PBOC et le Shadow Banking pour isoler la véritable marée monétaire mondiale (Expliquée à 70.8%).

## 🧠 Architecture Mathématique

### I. Équation de Diffusion de Langevin (SDE)
Le prix ne suit pas une ligne droite, il "orbite" de manière chaotique autour de la loi de puissance.
$$dP_t = \kappa(\ln P_{orbit} - \ln P_t)dt + \sigma dW_t$$
Où $\kappa$ est la force de rappel gravitationnelle et $dW_t$ représente l'entropie stochastique du marché.

### II. Le Facteur de Lorentz (Volatilité Relativiste)
Le modèle simule la compression de l'espace-temps financier via l'indice VIX :
$$\gamma_{Lorentz} = \exp\left(\frac{\min(VIX_t, 80) - 20}{40}\right)$$
En période de crise, les corrélations tendent vers 1 et l'attraction gravitationnelle entre les actifs s'intensifie violemment.

### III. Inférence Bayésienne (MCMC)
Nous utilisons l'échantillonneur **NUTS (No-U-Turn Sampler)** pour apprendre les constantes physiques ($\beta$) du système. Le modèle actuel affiche un **R² de 94.96%**, prouvant une cointégration quasi-parfaite entre l'énergie du réseau (Hashrate) et la liquidité globale.

## 📈 Résultats du Modèle (Cible 2028)
- **Exposant fractal détecté :** 5.48
- **Plancher de sécurité (5e centile) :** 118 206 $- **Cible Médiane :** 195 141$
- **Pic de Bulle Macro (95e centile) :** 321 035 $

## 🛠️ Installation et Exécution

1. **Environnement :** Python 3.9+
2. **Dépendances :** `pip install pandas numpy statsmodels pymc arviz scikit-learn matplotlib yfinance`
3. **Lancement :**
   ```bash
   python v33.py
   ```
   Cela génère le rapport `Nakamoto_RocheLobe_V37_GlobalLiq.pdf` et met à jour le payload `v33_dashboard_data.json`.

## ⚠️ Avertissement Scientifique
Ce modèle est une expérience théorique d'astroéconophysique. Les performances passées ne présagent pas des résultats futurs. **Ceci n'est pas un conseil financier.**

---

### 3. Visualisation : Comprendre l'Orbite de la Loi de Puissance
Pour votre tableau de bord, voici un simulateur qui explique aux utilisateurs comment le SDE de Langevin (votre prix simulé) "danse" autour de la ligne droite de Santostasi (le prix théorique).

```json?chameleon
{"component":"LlmGeneratedComponent","props":{"height":"700px","prompt":"Créer un simulateur interactif intitulé 'Dynamique de l'Orbite de Santostasi'. \n\n**Concept :** Visualiser comment le prix du Bitcoin (SDE) est attiré par la Loi de Puissance thermodynamique.\n\n**Données de base :**\n- Loi de Puissance déterministe (ligne droite en échelle Log-Log).\n- Prix stochastique (Langevin).\n\n**Contrôles (Sliders) :**\n1. 'Exposant de Santostasi' (4.0 à 6.5, valeur initiale 5.48).\n2. 'Force de Gravité (Kappa)' (0.1 à 5.0, gère la vitesse de retour vers la ligne).\n3. 'Entropie du Marché (Sigma)' (0.1 à 1.0, gère la violence des bulles et des krachs).\n\n**Comportement :**\n- Un graphique dynamique montrant les 10 prochaines années.\n- La ligne 'Loi de Puissance' est stable.\n- La ligne 'Prix SDE' oscille autour de la loi de puissance.\n- Si l'utilisateur augmente Kappa, le prix colle à la ligne. S'il augmente Sigma, les bulles deviennent paraboliques avant de s'effondrer vers l'orbite.\n- Afficher une boîte de texte expliquant : 'Le modèle v33 ne prédit pas un prix fixe, mais une orbite. Le Bitcoin peut dévier de sa valeur fondamentale à cause de la liquidité (bulles), mais la thermodynamique du réseau le ramène toujours vers son centre de gravité fractal.'","id":"im_c142565a5bc004af"}}
```
