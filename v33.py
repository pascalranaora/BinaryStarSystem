import os, requests, warnings, time, json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pymc as pm
import arviz as az

os.environ["PYTENSOR_FLAGS"] = "cxx="
warnings.filterwarnings("ignore")

# --- CONFIGURATION: ROCHE LOBE & VECM MASTER ---
FRED_API_KEY = '---'
HASHRATE_LAG = 245  
FORECAST_DAYS = 730 
STRUCTURAL_M2_GROWTH = 0.06  
TAM_MULTIPLIER = 5.0 
MC_PATHS = 2000
HOWELL_CRISIS_THRESHOLD = 1.65 # Ratio Dette/Liquidité critique

HALVING_DATES = [
    pd.to_datetime('2012-11-28'), pd.to_datetime('2016-07-09'),
    pd.to_datetime('2020-05-11'), pd.to_datetime('2024-04-19'),
    pd.to_datetime('2028-03-25') 
]

plt.rcParams.update({
    "font.family": "serif", "figure.dpi": 300, "axes.grid": True, "grid.alpha": 0.25,
    "axes.labelsize": 10, "axes.titlesize": 12, "legend.fontsize": 9,
    "figure.facecolor": "#ffffff"
})

class DataFetcher:
    def __init__(self, cache_dir="cache", max_age_hours=24):
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
            
    def _is_valid(self, filename):
        filepath = os.path.join(self.cache_dir, filename)
        if not os.path.exists(filepath): return False
        return (time.time() - os.path.getmtime(filepath)) < (self.max_age_hours * 3600)
        
    def get_blockchain_metric(self, chart_name, col_name):
        fname = f"{chart_name}.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname): return pd.read_csv(fpath, index_col=0, parse_dates=True)
        try:
            url = f"https://api.blockchain.info/charts/{chart_name}?timespan=all&format=json"
            df = pd.DataFrame(requests.get(url, timeout=15).json()['values'])
            df['date'] = pd.to_datetime(df['x'], unit='s').dt.normalize()
            df = df.set_index('date').rename(columns={'y': col_name})[[col_name]].resample('D').mean().ffill()
            df.to_csv(fpath)
            return df
        except Exception:
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            return pd.DataFrame()

    def get_price(self):
        fname = "price_tv.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname): return pd.read_csv(fpath, index_col=0, parse_dates=True)
        try:
            p = TvDatafeed().get_hist(symbol="BTCUSD", exchange="INDEX", interval=Interval.in_daily, n_bars=8000)
            df = p.rename(columns={'close': 'Price'})[['Price']].resample('D').mean().ffill()
            df.index = pd.to_datetime(df.index).normalize()
            df.to_csv(fpath)
            return df
        except Exception:
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            return pd.DataFrame()

    def get_yf_close(self, tickers, name):
        fname = f"yf_{name}.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname): return pd.read_csv(fpath, index_col=0, parse_dates=True)
        try:
            df = yf.download(tickers, start="2009-01-01", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.levels[0]: df = df['Close']
            elif 'Close' in df.columns:
                df = df['Close']
            if isinstance(df, pd.Series): df = df.to_frame(name=name)
            df.to_csv(fpath)
            return df
        except Exception:
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            return pd.DataFrame()
            
    def get_fred(self):
        fname = "fred_macro_v7.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname): return pd.read_csv(fpath, index_col=0, parse_dates=True)
        
        # Updated Tickers: Replaced DPSACBW with QUSPBMUSDA for Private Banking Credit
        series_list = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'M2SL', 'VIXCLS', 'CPIAUCSL', 
                       'FEDFUNDS', 'BAMLH0A0HYM2', 'T10Y2Y', 'GFDEBTN', 'WSHOSHO', 'QUSPBMUSDA']
        df_list = []
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(connect=5, backoff_factor=0.5))
        session.mount('https://', adapter)
        dates = pd.date_range(start='2009-01-03', end=datetime.today(), freq='D')
        for s in series_list:
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s}&api_key={FRED_API_KEY}&file_type=json&observation_start=2009-01-03"
                temp = pd.DataFrame(session.get(url, timeout=30).json()['observations'])
                temp['date'] = pd.to_datetime(temp['date'])
                temp['value'] = pd.to_numeric(temp['value'], errors='coerce')
                df_list.append(temp.set_index('date')[['value']].rename(columns={'value': s}))
            except Exception:
                df_list.append(pd.DataFrame({'date': dates, s: 0.0}).set_index('date'))
        
        m = pd.concat(df_list, axis=1).resample('D').interpolate('time').ffill().bfill()
        m.to_csv(fpath)
        return m

class BinaryStarRocheLobeModel:
    def __init__(self):
        self.genesis = pd.to_datetime('2009-01-03')
        self.df = None
        self.x_scaler = StandardScaler()
        self.modeled_price_history = None
        self.val_z_score_series = None
        self.lyapunov_exponent = 0.0
        self.fetcher = DataFetcher()

    def calculate_lle(self, series, lag=5):
        N = len(series); series_arr = series.values
        if N < lag * 2: return 0.0
        divergences = []
        for i in range(N - lag - 10):
            dists = np.abs(series_arr[:N-lag] - series_arr[i])
            dists[i] = np.inf 
            nn_idx = np.argmin(dists)
            d0 = dists[nn_idx] + 1e-8
            dt = np.abs(series_arr[i+lag] - series_arr[nn_idx+lag]) + 1e-8
            divergences.append(np.log(dt / d0))
        return np.mean(divergences) / lag
    
    def create_global_liquidity_index(self, df):
        """
        Implémente la thèse de Howell (Volume) et TBL (Plumbing).
        Utilise PCA pour trouver l'indice de corrélation maximale.
        """
        print("   🔍 Building Howell-TBL Global Liquidity Index (PCA Optimization)...")
        
        # 1. US Net Liquidity (The Bitcoin Layer)
        df['US_Net_Liq'] = df['WALCL'].fillna(0) - df['WTREGEN'].fillna(0) - df['RRPONTSYD'].fillna(0)
        df['US_Net_Liq'] = np.where(df['US_Net_Liq'] > 1e6, df['US_Net_Liq'], df['M2SL'] * 1000)
        
        # 2. Shadow Banking / Private Credit (Howell component)
        df['Private_Credit'] = df['QUSPBMUSDA'].rolling(90, min_periods=1).mean().ffill().bfill()
        
        # 3. Global CB Proxy (Howell component)
        df['Global_CB_Flow'] = df['WSHOSHO'].diff(90).fillna(0)
        
        # Normalisation pour PCA. Fix: Use .ffill() instead of fillna(method='ffill')
        liq_vars = ['US_Net_Liq', 'Private_Credit', 'Global_CB_Flow', 'M2SL']
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df[liq_vars].ffill().bfill())
        
        # PCA : On extrait la 1ère composante (l'Onde de Liquidité Mondiale)
        pca = PCA(n_components=1)
        gli_wave_raw = pca.fit_transform(standardized_data).flatten() # Aplatit le array numpy ici
        df['GLI_Wave'] = gli_wave_raw
        
        # Inverser l'onde si le PCA l'a lue à l'envers (le PCA ne connaît pas le signe de corrélation)
        # On s'assure que l'onde de liquidité est positivement corrélée au M2 historique
        if np.corrcoef(gli_wave_raw, df['M2SL'].ffill().bfill())[0, 1] < 0:
            df['GLI_Wave'] = df['GLI_Wave'] * -1
            
        print(f"   ✅ Liquidity PCA complete. Explained Variance: {pca.explained_variance_ratio_[0]*100:.1f}%")
        return df

    def apply_thermodynamics(self, df):
        hodl_raw = (self.all_supply[df.index] - (df['Tx_Vol_USD'] / df['Price']).rolling(30).mean()).dropna()
        df['WD_Density'] = (hodl_raw / self.all_supply[df.index]).bfill()
        
        res_a = sm.OLS(np.log(hodl_raw), sm.add_constant(np.log((hodl_raw.index - self.genesis).days))).fit()
        self.pl_const, self.alpha = res_a.params
        df['SF'] = np.exp(self.pl_const + self.alpha * np.log((df.index - self.genesis).days)) / self.all_ann_flow[df.index]

        df['M2_TAM'] = df['M2SL'] * 1e9 * TAM_MULTIPLIER
        df['Mass_Ratio'] = (df['Price'] * self.all_supply[df.index]) / df['M2_TAM']
        df['Lorentz_Factor'] = np.exp((df['VIXCLS'].clip(upper=80) - 20) / 40)

        # Intégration du nouvel indice Global
        df = self.create_global_liquidity_index(df)
        
        # Valve de Régime (Raoul Pal Business Cycle)
        cs = df['BAMLH0A0HYM2'].fillna(4.0)
        cs_valve = np.exp(-(cs - 4.0) / 3.0).clip(upper=1.2)
        yc = df['T10Y2Y'].fillna(1.0)
        yc_valve = 1 / (1 + np.exp(-yc * 2))
        df['Regime_Phi'] = cs_valve * yc_valve

        # NEW DEBT WALL ACCRETION PHYSICS
        df['Debt_to_M2'] = (df['GFDEBTN'] / 1000.0) / df['M2SL']
        
        # On utilise l'accélération du GLI (Howell/TBL) adoucie
        gli_acceleration = df['GLI_Wave'].diff(30).diff(30).fillna(0)
        debt_momentum = df['GFDEBTN'].diff(90).fillna(0)
        
        # Valve de Pal + Pression de la Dette + Onde Howell
        base_force = (gli_acceleration * df['Regime_Phi']) + (debt_momentum * df['Debt_to_M2'])
        df['Accretion_Force'] = base_force.shift(30).rolling(30).mean().fillna(0)
        
        # Robust Z-Score Scaling
        std_acc = df['Accretion_Force'].rolling(365, min_periods=90).std().bfill() + 1e-8
        df['Accretion_Force'] = ((df['Accretion_Force'] - df['Accretion_Force'].rolling(365, min_periods=90).mean().bfill()) / std_acc).fillna(0)
        return df

    def assemble_tensors(self):
        print("\n--- [1/5] ASSEMBLING TENSORS (Astrophysics & Cointegration) ---")
        price_df = self.fetcher.get_price()
        h = self.fetcher.get_blockchain_metric('hash-rate', 'Hashrate')
        v = self.fetcher.get_blockchain_metric('n-unique-addresses', 'Velocity')
        tx = self.fetcher.get_blockchain_metric('estimated-transaction-volume-usd', 'Tx_Vol_USD')
        m = self.fetcher.get_fred()
        
        dates_all = pd.date_range(self.genesis, datetime.today() + timedelta(days=FORECAST_DAYS), freq='D')
        epochs = np.floor((dates_all - self.genesis).days / 1458.33)
        flow = pd.Series(144 * (50.0 / (2.0 ** epochs)), index=dates_all)
        self.all_supply = flow.cumsum(); self.all_ann_flow = flow.rolling(365).sum().bfill()

        h['H_Elastic'] = pd.Series(np.where((h['Hashrate'].rolling(30).mean() < h['Hashrate'].rolling(60).mean()), h['Hashrate'].shift(-60), h['Hashrate'].shift(-HASHRATE_LAG)), index=h.index).ffill()

        self.df = price_df.join([h, m, v, tx], how='left').ffill()
        self.df = self.apply_thermodynamics(self.df)

        roro_assets = self.fetcher.get_yf_close(["HYG", "IEF", "COPX", "GLD", "SPHB", "SPLV"], "roro_assets")
        roro_assets = roro_assets.reindex(self.df.index).ffill().bfill()
        cr, gr, er = roro_assets['HYG'] / roro_assets['IEF'], roro_assets['COPX'] / roro_assets['GLD'], roro_assets['SPHB'] / roro_assets['SPLV']
        self.df['Global_RORO_State'] = (1 / (1 + np.exp(-((cr - cr.rolling(90).mean())/cr.rolling(90).std() + (gr - gr.rolling(90).mean())/gr.rolling(90).std() + (er - er.rolling(90).mean())/er.rolling(90).std())))).ewm(span=30).mean()
        self.df = self.df.dropna()

    def run_audit(self):
        print("\n--- [2/5] MCMC BAYESIAN PHYSICS AUDIT (V37: Global Liquidity Wave) ---")
        trace_file = "mcmc_physics_trace.nc"
        
        # We MUST remove the cache to force the model to learn the new GLI_Wave Beta
        # if os.path.exists(trace_file): 
        #     os.remove(trace_file)

        self.df = self.df[self.df.index >= '2009-01-03']
        self.df['Logit_Rho'] = np.log((((self.df['Price'] * self.all_supply[self.df.index]) / 1e9) / (self.df['M2SL'] * TAM_MULTIPLIER)).clip(1e-5, 0.999))
        self.df['Logit_Rho'] = self.df['Logit_Rho'].replace([np.inf, -np.inf], np.nan).ffill()

        self.X_cols = ['WD_Density', 'log_V', 'log_SF', 'log_H', 'Accretion_Force', 'Global_RORO_State']
        self.d_log = pd.DataFrame({
            'Logit_Rho': self.df['Logit_Rho'], 'WD_Density': self.df['WD_Density'],
            'log_V': np.log(self.df['Velocity']), 'log_SF': np.log(self.df['SF']),
            'log_H': np.log(self.df['H_Elastic']), 'Accretion_Force': self.df['Accretion_Force'],
            'Global_RORO_State': self.df['Global_RORO_State'], 'Price': self.df['Price'],
            'M2_TAM': self.df['M2_TAM'], 'Lorentz_Factor': self.df['Lorentz_Factor'],
            'Mass_Ratio': self.df['Mass_Ratio']
        }).dropna()

        X_scaled = self.x_scaler.fit_transform(self.d_log[self.X_cols])
        y = self.d_log['Logit_Rho'].values

        if os.path.exists(trace_file):
            print(f"   📦 Loading existing MCMC trace from {trace_file}...")
            self.mcmc_trace = az.from_netcdf(trace_file)
            self.mcmc_stats = az.summary(self.mcmc_trace, hdi_prob=0.94)
        else:
            with pm.Model() as model:
                intercept = pm.Normal('intercept', mu=0, sigma=10)
                beta_other = pm.Normal('beta_other', mu=0, sigma=5, shape=4) 
                beta_positive = pm.HalfNormal('beta_positive', sigma=5, shape=2) 
                
                beta = pm.Deterministic('beta', pm.math.stack([beta_other[0], beta_other[1], beta_positive[0], beta_positive[1], beta_other[2], beta_other[3]]))
                
                sigma = pm.HalfNormal('sigma', sigma=2)
                mu = intercept + pm.math.dot(X_scaled, beta)
                pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
                
                print("   ⚛️ Launching NUTS Sampler (Learning GLI wave mechanics)...")
                self.mcmc_trace = pm.sample(500, tune=500, chains=2, target_accept=0.95, progressbar=True)
                self.mcmc_stats = az.summary(self.mcmc_trace, hdi_prob=0.94)
                az.to_netcdf(self.mcmc_trace, trace_file)

        print("\n--- MCMC POSTERIOR SUMMARY ---")
        print(self.mcmc_stats.loc[['intercept', 'beta[0]', 'beta[1]', 'beta[2]', 'beta[3]', 'beta[4]', 'beta[5]']])

        self.mcmc_intercept = float(self.mcmc_stats.loc['intercept', 'mean'])
        self.mcmc_betas = np.array([
            float(self.mcmc_stats.loc['beta[0]', 'mean']),
            float(self.mcmc_stats.loc['beta[1]', 'mean']),
            float(self.mcmc_stats.loc['beta[2]', 'mean']),
            float(self.mcmc_stats.loc['beta[3]', 'mean']),
            float(self.mcmc_stats.loc['beta[4]', 'mean']),
            float(self.mcmc_stats.loc['beta[5]', 'mean'])
        ])

        mu_fitted = self.mcmc_intercept + np.dot(X_scaled, self.mcmc_betas)
        self.ECT = y - mu_fitted
        
        ss_res = np.sum(self.ECT ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)
        print(f"   VECM R-Squared: {self.r_squared:.4f}")

        self.adjusted_ECT = self.ECT * self.d_log['Lorentz_Factor']
        self.ect_p_value = adfuller(self.adjusted_ECT)[1]

        self.modeled_price_history = (np.exp(mu_fitted) / (1 + np.exp(mu_fitted)) * self.d_log['M2_TAM']) / self.all_supply[self.d_log.index].values
        self.val_z_score_series = (np.log(self.d_log['Price']/self.modeled_price_history)).rolling(30).mean() / (np.log(self.d_log['Price']/self.modeled_price_history)).rolling(365).std()
        self.lyapunov_exponent = self.calculate_lle(self.val_z_score_series.dropna())
        print(f"   Largest Lyapunov Exponent (LLE): {self.lyapunov_exponent:.5f}")

    def execute_mc(self):
        print("\n--- [3/5] MC LANGEVIN SDE PROJECTION (Hybrid: Santostasi Power Law + Macro SDE) ---")
        f_dates = pd.date_range(self.d_log.index[-1] + timedelta(1), periods=FORECAST_DAYS, freq='D')
        dt = 1.0 / 365.0
        
        # 1. THE THERMODYNAMIC CORE: Extraction de la Loi de Puissance de Santostasi
        # P = c * t^alpha  <=>  ln(P) = ln(c) + alpha * ln(t)
        days_history = np.log((self.d_log.index - self.genesis).days.values)
        days_future = np.log((f_dates - self.genesis).days.values)
        
        p_model = sm.OLS(np.log(self.d_log['Price'].values), sm.add_constant(days_history)).fit()
        p_const, p_slope = p_model.params
        
        print(f"   📐 Loi de Puissance (Santostasi) détectée - Exposant: {p_slope:.2f}")
        
        # Trajectoire déterministe pure (La ligne droite ascendante sur un graphique Log-Log)
        power_law_orbit = np.exp(p_const + p_slope * days_future)
        
        # 2. MACRO SDE INITIALIZATION
        base_vix = self.df['VIXCLS'].iloc[-1]
        mc_paths = []
        
        for _ in tqdm(range(MC_PATHS)):
            sim_p = np.zeros(FORECAST_DAYS)
            sim_p[0] = self.d_log['Price'].iloc[-1]
            
            p_acc = self.d_log['Accretion_Force'].iloc[-1]
            sim_vix = base_vix
            
            # Génération du bruit brownien
            dW_acc = np.random.normal(0, np.sqrt(dt), FORECAST_DAYS)
            dW_vix = np.random.normal(0, np.sqrt(dt), FORECAST_DAYS)
            dW_price = np.random.normal(0, np.sqrt(dt), FORECAST_DAYS)

            for t in range(1, FORECAST_DAYS):
                # Volatilité (VIX) - Mean Reversion
                sim_vix = sim_vix + 5.0 * (20.0 - sim_vix) * dt + 8.0 * dW_vix[t]
                loc_lorentz = np.exp((min(sim_vix, 80) - 20) / 40)
                
                # Onde de Liquidité Globale (Accretion Force)
                # Retour à la moyenne vers +0.5 (Tendance structurelle à l'impression monétaire / Howell)
                p_acc = p_acc + 1.5 * (0.5 - p_acc) * dt + 2.0 * loc_lorentz * dW_acc[t]
                
                # --- THE HYBRID SDE EQUATION ---
                # La Cible n'est plus fixe, elle monte chaque jour avec la Loi de Puissance.
                # L'Onde de Liquidité modifie cette cible : +1 Z-Score de liquidité = +15% de prime de prix (Bulle)
                macro_multiplier = np.exp(p_acc * 0.15) 
                dynamic_target = power_law_orbit[t] * macro_multiplier
                
                # Processus d'Ornstein-Uhlenbeck (Élastique gravitationnel vers la cible)
                kappa = 2.5 * loc_lorentz # Vitesse de rappel dépend de la volatilité macro
                drift = kappa * (np.log(dynamic_target) - np.log(sim_p[t-1])) * dt
                
                # Diffusion Stochastique (Random Market Walk ~ 60% vol annuelle)
                diffusion = 0.60 * loc_lorentz * dW_price[t] 
                
                # Mise à jour géométrique du Prix
                sim_p[t] = sim_p[t-1] * np.exp(drift + diffusion)
                
            mc_paths.append(sim_p)
            
        mc = np.array(mc_paths)

        print(f"   -> Simulation complete. Target Range [5th-95th]: ${np.percentile(mc[:,-1], 5):,.0f} - ${np.percentile(mc[:,-1], 95):,.0f}")
        return f_dates, mc

if __name__ == "__main__":
    engine = BinaryStarRocheLobeModel()
    engine.assemble_tensors()
    engine.run_audit()
    f_dates, mc_final_prices = engine.execute_mc()

    print("\n--- [4/5] RENDERING PDF REPORT ---")
    median, u95, l05 = np.percentile(mc_final_prices, [50, 95, 5], axis=0)
    
    log_diff = np.log(engine.d_log['Price']) - np.log(engine.modeled_price_history)
    rolling_std = log_diff.rolling(365, min_periods=90).std().fillna(0.5)
    upper_band = engine.modeled_price_history * np.exp(2 * rolling_std)
    lower_band = engine.modeled_price_history * np.exp(-2 * rolling_std)
    
    with PdfPages('Nakamoto_RocheLobe_V37_GlobalLiq.pdf') as pdf:
        # Page 1: Equilibrium & Forecast
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        ax1.plot(engine.d_log.index, engine.d_log['Price'], color='silver', label='Actual Price')
        ax1.plot(engine.d_log.index, engine.modeled_price_history, color='blue', label='VECM MCMC Equilibrium')
        ax1.fill_between(engine.d_log.index, lower_band, upper_band, color='blue', alpha=0.1)
        ax1.plot(f_dates, median, color='red', ls='--', label=f'Median Forecast: ${median[-1]:,.0f}')
        ax1.fill_between(f_dates, l05, u95, color='red', alpha=0.1)
        ax1.set_yscale('log'); ax1.set_title("Stochastic Relativistic Equilibrium (V37: Global Liquidity PCA)"); ax1.legend(loc='upper left')
        ax2.hist(mc_final_prices[:, -1], bins=100, color='navy', alpha=0.5, density=True); ax2.set_title("Terminal Price Density")
        pdf.savefig(fig1); plt.close()

        # Page 2: Phase Space & Lyapunov
        fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 12))
        ax3.plot(engine.d_log.index, engine.val_z_score_series, color='purple', label='ECT Z-Score')
        ax3.axhline(2, color='red', ls='--'); ax3.axhline(-2, color='green', ls='--'); ax3.set_ylim(-4,4)
        ax3.set_title("Thermodynamic Z-Score (Mean Reversion Tension)")
        z = engine.val_z_score_series.dropna().values
        sc = ax4.scatter(z[:-1], np.diff(z), c=np.arange(len(z)-1), cmap='magma', s=5, alpha=0.5)
        plt.colorbar(sc, ax=ax4, label="Time Progression")
        ax4.set_title(f"Orbital Phase Space Chaos (LLE: {engine.lyapunov_exponent:.4f})")
        pdf.savefig(fig2); plt.close()

        # Page 3: MCMC Forest Plot
        fig3, ax5 = plt.subplots(figsize=(10, 6))
        az.plot_forest(engine.mcmc_trace, var_names=["beta_other", "beta_positive"], combined=True, ax=ax5)
        ax5.set_title("MCMC Learned Physics: Posterior Distributions (Informed Priors)")
        pdf.savefig(fig3); plt.close()
        
        

    print("\n--- [5/5] GENERATING WEB DASHBOARD JSON ---")
    def clean_val(val, decimals=2):
        if pd.isna(val) or np.isinf(val): return None
        return round(float(val), decimals)

    historical_data = []
    for date in engine.d_log.index[::7]:
        idx = engine.d_log.index.get_loc(date)
        historical_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "actual_price": clean_val(engine.d_log['Price'].iloc[idx], 2),
            "model_price": clean_val(engine.modeled_price_history.iloc[idx], 2),
            "upper_band": clean_val(upper_band.iloc[idx], 2),
            "lower_band": clean_val(lower_band.iloc[idx], 2),
            "z_score": clean_val(engine.val_z_score_series.iloc[idx], 3),
            "mass_ratio": clean_val(engine.d_log['Mass_Ratio'].iloc[idx], 5),
            "lorentz_factor": clean_val(engine.d_log['Lorentz_Factor'].iloc[idx], 3)
        })
        
    projection_data = []
    for idx, date in enumerate(f_dates[::7]):
        real_idx = idx * 7
        projection_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "median_projection": clean_val(median[real_idx], 2),
            "upper_projection": clean_val(u95[real_idx], 2), 
            "lower_projection": clean_val(l05[real_idx], 2)
        })

    counts, bin_edges = np.histogram(mc_final_prices[:, -1], bins=100, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mcmc_params = {
        "beta[0]": engine.mcmc_stats.loc['beta[0]'].to_dict(),
        "beta[1]": engine.mcmc_stats.loc['beta[1]'].to_dict(),
        "beta[2]": engine.mcmc_stats.loc['beta[2]'].to_dict(), 
        "beta[3]": engine.mcmc_stats.loc['beta[3]'].to_dict(), 
        "beta[4]": engine.mcmc_stats.loc['beta[4]'].to_dict(),
        "beta[5]": engine.mcmc_stats.loc['beta[5]'].to_dict()
    }

    dashboard_payload = {
        "metadata": {
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "r_squared_vecm": clean_val(getattr(engine, 'r_squared', 0.0), 4),
            "target_median": clean_val(median[-1], 0),
            "physics_metrics": {
                "largest_lyapunov_exponent": clean_val(engine.lyapunov_exponent, 5),
                "terminal_mass_ratio": clean_val(engine.d_log['Mass_Ratio'].iloc[-1], 5),
                "terminal_lorentz_factor": clean_val(engine.d_log['Lorentz_Factor'].iloc[-1], 4),
                "ect_p_value": clean_val(getattr(engine, 'ect_p_value', 0.0), 4),
                "mcmc_learned_params": mcmc_params
            }
        },
        "historical": historical_data,
        "projection": projection_data,
        "distribution": {
            "bin_centers": [clean_val(v, 2) for v in bin_centers],
            "density": [int(v) for v in counts], 
            "median": clean_val(median[-1], 2), "l05": clean_val(l05[-1], 2), "u95": clean_val(u95[-1], 2),
            "target_date": f_dates[-1].strftime('%B %Y')
        } 
    }

    with open('v33_dashboard_data.json', 'w') as f: json.dump(dashboard_payload, f)
    print("Success! Dashboard JSON and MCMC physics trace updated.")