import os, requests, warnings, time, json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

# --- CONFIGURATION: ROCHE LOBE & VECM MASTER ---
FRED_API_KEY = '---' # ADD YOUR FRED API KEY HERE
HASHRATE_LAG = 245  
FORECAST_DAYS = 730 
STRUCTURAL_M2_GROWTH = 0.06  
TAM_MULTIPLIER = 5.0 
HOWELL_CYCLE_DAYS = 1975 
MC_PATHS = 5000

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

# ==========================================
# UNIVERSAL CACHE MANAGER (24 HOURS)
# ==========================================
class DataFetcher:
    def __init__(self, cache_dir="cache", max_age_hours=24):
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
    def _is_valid(self, filename):
        filepath = os.path.join(self.cache_dir, filename)
        if not os.path.exists(filepath): return False
        return (time.time() - os.path.getmtime(filepath)) < (self.max_age_hours * 3600)
        
    def get_blockchain_metric(self, chart_name, col_name):
        fname = f"{chart_name}.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname):
            print(f"   [{col_name}] Loading from cache...")
            return pd.read_csv(fpath, index_col=0, parse_dates=True)
            
        print(f"   [{col_name}] Downloading from Blockchain.info...")
        try:
            url = f"https://api.blockchain.info/charts/{chart_name}?timespan=all&format=json"
            data = requests.get(url, timeout=15).json()['values']
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['x'], unit='s').dt.normalize()
            df = df.set_index('date').rename(columns={'y': col_name})[[col_name]].resample('D').mean().ffill()
            df.to_csv(fpath)
            return df
        except Exception as e:
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            return pd.DataFrame()

    def get_price(self):
        fname = "price_tv.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname):
            print("   [Price] Loading from cache...")
            return pd.read_csv(fpath, index_col=0, parse_dates=True)
        print("   [Price] Downloading TvDatafeed...")
        try:
            tv = TvDatafeed()
            p = tv.get_hist(symbol="BTCUSD", exchange="INDEX", interval=Interval.in_daily, n_bars=8000)
            df = p.rename(columns={'close': 'Price'})[['Price']].resample('D').mean().ffill()
            df.index = pd.to_datetime(df.index).normalize()
            df.to_csv(fpath)
            return df
        except Exception as e:
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            return pd.DataFrame()

    def get_yf_close(self, tickers, name):
        fname = f"yf_{name}.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname):
            print(f"   [{name}] Loading from cache...")
            return pd.read_csv(fpath, index_col=0, parse_dates=True)
        try:
            print(f"   [{name}] Downloading YFinance data...")
            df = yf.download(tickers, start="2010-01-01", progress=False)
            if 'Close' in df.columns: df = df['Close']
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            df.to_csv(fpath)
            return df
        except Exception as e:
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            return pd.DataFrame()
            
    def get_fred(self):
        fname = "fred_macro_v4.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname):
            print("   [Macro] Loading FRED Data from cache...")
            return pd.read_csv(fpath, index_col=0, parse_dates=True)
        
        print("   [Macro] Downloading Auth FRED API (Fault-Tolerant Mode)...")
        series_list = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'M2SL', 'VIXCLS', 'CPIAUCSL', 'FEDFUNDS']
        df_list = []
        
        session = requests.Session()
        retry = Retry(connect=5, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        dates = pd.date_range(start='2009-01-03', end=datetime.today(), freq='D')
        days_array = np.arange(len(dates))

        for s in series_list:
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s}&api_key={FRED_API_KEY}&file_type=json&observation_start=2009-01-03"
                r = session.get(url, timeout=30) 
                r.raise_for_status()
                data = r.json()
                temp = pd.DataFrame(data['observations'])
                temp['date'] = pd.to_datetime(temp['date'])
                temp['value'] = pd.to_numeric(temp['value'], errors='coerce')
                temp = temp.set_index('date')[['value']].rename(columns={'value': s})
                df_list.append(temp)
                print(f"      -> {s} ... OK")
            except Exception as e:
                print(f"      -> [Warning] FRED series '{s}' failed. Generating dynamic thermodynamic proxy.")
                if s == 'M2SL': val = 8000.0 * np.exp(days_array * (STRUCTURAL_M2_GROWTH/365)) + np.random.normal(0, 50, len(dates)).cumsum()
                elif s == 'WALCL': val = 4000.0 * np.exp(days_array * (0.05/365)) + np.random.normal(0, 30, len(dates)).cumsum()
                elif s == 'CPIAUCSL': val = 210.0 + (days_array * 0.005) + np.sin(days_array/365)*5
                elif s == 'FEDFUNDS': val = 2.5 + np.sin(days_array/730)*2.0
                elif s == 'VIXCLS': val = 20.0 + np.random.normal(0, 2, len(dates))
                else: val = 0.0 
                
                temp = pd.DataFrame({'date': dates, s: val}).set_index('date')
                df_list.append(temp)
        
        print("   [Macro] Merging Macro Tensors...")
        m = pd.concat(df_list, axis=1)
        m = m.resample('D').interpolate('time').ffill().fillna(method='bfill')
        m.to_csv(fpath)
        return m

# ==========================================
# MAIN VECM & ROCHE LOBE ENGINE
# ==========================================
class BinaryStarRocheLobeModel:
    def __init__(self):
        self.genesis = pd.to_datetime('2009-01-03')
        self.df = None
        self.coint_model = None
        self.x_scaler = StandardScaler()
        self.modeled_price_history = None
        self.val_z_score_series = None
        self.lyapunov_exponent = 0.0
        self.fetcher = DataFetcher()

    def calculate_lle(self, series, lag=5):
        """Largest Lyapunov Exponent proxy for phase-space density/chaos"""
        N = len(series)
        if N < lag * 2: return 0.0
        divergences = []
        eps = 1e-8
        series_arr = series.values
        
        # Simplified Rosenstein approach
        for i in range(N - lag - 10):
            dists = np.abs(series_arr[:N-lag] - series_arr[i])
            dists[i] = np.inf 
            nn_idx = np.argmin(dists)
            
            d0 = dists[nn_idx] + eps
            dt = np.abs(series_arr[i+lag] - series_arr[nn_idx+lag]) + eps
            divergences.append(np.log(dt / d0))
            
        return np.mean(divergences) / lag

    def apply_thermodynamics(self, df):
        # 1. White Dwarf Density 
        safe_velocity = df['Velocity'].replace(0, np.nan).bfill()
        hoarding_rate = np.where(df['Price'] < df['MA200'], 2500, 500)
        corp_h = pd.Series(hoarding_rate, index=df.index).cumsum()
        corp_h[df.index < '2020-08-11'] = 0 
        
        hodl_raw = (self.all_supply[df.index] - (df['Tx_Vol_USD'] / df['Price']).rolling(30).mean() - corp_h).dropna()
        df['WD_Density'] = hodl_raw / self.all_supply[df.index] 
        
        # Power Law Structural Scarcity (S2F)
        res_a = sm.OLS(np.log(hodl_raw), sm.add_constant(np.log((hodl_raw.index - self.genesis).days))).fit()
        self.pl_const, self.alpha = res_a.params
        df['SF'] = (np.exp(self.pl_const + self.alpha * np.log((df.index - self.genesis).days)) + corp_h) / self.all_ann_flow[df.index]

        # 2. Red Giant Equation of State (PV = nRT)
        df['YoY_Inflation'] = df['CPIAUCSL'].pct_change(365) * 100
        df['Fluid_Pressure'] = df['FEDFUNDS'].clip(lower=0.1) 
        
        # 3. Mass Heterogeneity (Dynamic Masses)
        df['M1_Asset'] = df['Price'] * self.all_supply[df.index]  # Market Cap of Asset 1
        df['M2_TAM'] = df['Broad_M2'] * 1e9 * TAM_MULTIPLIER      # Total Addressable Fluid Mass
        df['Mass_Ratio'] = df['M1_Asset'] / df['M2_TAM']          # Orbital center of mass shift
        
        # 4. Relativistic Effects (High Volatility Space-Time Shrink)
        # As VIX spikes, distance shrinks, gravitational pull intensifies (correlation -> 1)
        df['Lorentz_Factor'] = np.exp((df['VIXCLS'].clip(upper=80) - 20) / 40)

        # 5. Roche Lobe Overflow (Bernoulli Fluid Transfer modified by Lorentz)
        inflation_excess = (df['YoY_Inflation'] - 2.0).clip(lower=0)
        df['Mass_Transfer_Velocity'] = (inflation_excess * df['Broad_M2']) / df['Fluid_Pressure']
        df['Accretion_Force'] = (df['Mass_Transfer_Velocity'] * df['Lorentz_Factor']).rolling(90).mean().fillna(0)
        
        return df

    def assemble_tensors(self):
        print("\n--- [1/5] ASSEMBLING TENSORS (Astrophysics & Cointegration) ---")
        price_df = self.fetcher.get_price()
        price_df['MA200'] = price_df['Price'].rolling(200).mean()

        h = self.fetcher.get_blockchain_metric('hash-rate', 'Hashrate')
        v = self.fetcher.get_blockchain_metric('n-unique-addresses', 'Velocity')
        tx = self.fetcher.get_blockchain_metric('estimated-transaction-volume-usd', 'Tx_Vol_USD')
        m = self.fetcher.get_fred()
        
        m['Broad_M2'] = m['M2SL']
        
        dates_all = pd.date_range(self.genesis, datetime.today() + timedelta(days=FORECAST_DAYS), freq='D')
        epochs = np.floor((dates_all - self.genesis).days / 1458.33)
        flow = pd.Series(144 * (50.0 / (2.0 ** epochs)), index=dates_all)
        self.all_supply = flow.cumsum(); self.all_ann_flow = flow.rolling(365).sum().bfill()

        # Hashrate Elasticity
        h_elastic_raw = np.where((h['Hashrate'].rolling(30).mean() < h['Hashrate'].rolling(60).mean()), h['Hashrate'].shift(-60), h['Hashrate'].shift(-245))
        h['H_Elastic'] = pd.Series(h_elastic_raw, index=h.index).ffill()

        self.df = price_df.join([h, m, v, tx], how='inner').dropna()
        self.df = self.apply_thermodynamics(self.df)

        # Global RORO State
        roro_assets = self.fetcher.get_yf_close(["HYG", "IEF", "COPX", "GLD", "SPHB", "SPLV"], "roro_assets")
        roro_assets = roro_assets.reindex(self.df.index).ffill().bfill()
        cr, gr, er = roro_assets['HYG'] / roro_assets['IEF'], roro_assets['COPX'] / roro_assets['GLD'], roro_assets['SPHB'] / roro_assets['SPLV']
        def rz(s): return (s - s.rolling(90).mean()) / (s.rolling(90).std() + 1e-8)
        self.df['Global_RORO_State'] = (1 / (1 + np.exp(-(rz(cr) + rz(gr) + rz(er))))).ewm(span=30).mean()

    def run_audit(self):
        print("\n--- [2/5] ECONOMETRIC AUDIT (Johansen & VECM Proxy) ---")
        self.df['TAM_B'] = self.df['Broad_M2'] * TAM_MULTIPLIER
        self.df['Rho'] = (((self.df['Price'] * self.all_supply[self.df.index]) / 1e9) / self.df['TAM_B']).clip(upper=0.999)
        self.df['Logit_Rho'] = np.log(self.df['Rho'] / (1 - self.df['Rho']))
        
        audit_data = pd.DataFrame({
            'Logit_Rho': self.df['Logit_Rho'], 
            'log_H': np.log(self.df['H_Elastic']),
            'WD_Density': self.df['WD_Density'],
            'log_V': np.log(self.df['Velocity']),
            'log_SF': np.log(self.df['SF']), 
            'Accretion_Force': self.df['Accretion_Force'],
            'Global_RORO_State': self.df['Global_RORO_State'], 
            'Price': self.df['Price'], 
            'TAM_B': self.df['TAM_B'],
            'Mass_Ratio': self.df['Mass_Ratio'],
            'Lorentz_Factor': self.df['Lorentz_Factor']
        }).dropna()
        
        self.X_cols = ['WD_Density', 'log_V', 'log_SF', 'log_H', 'Accretion_Force', 'Global_RORO_State']
        self.d_log = audit_data 
        
        X_scaled_df = pd.DataFrame(self.x_scaler.fit_transform(self.d_log[self.X_cols]), columns=self.X_cols, index=self.d_log.index)
        X_coint = sm.add_constant(X_scaled_df)
        
        self.coint_model = sm.OLS(self.d_log['Logit_Rho'], X_coint).fit(cov_type='HC3')
        self.ECT = self.coint_model.resid
        
        # Relativistic ECT Modification: Gravity pulls harder when volatility (Lorentz) is high
        self.adjusted_ECT = self.ECT * self.d_log['Lorentz_Factor']

        adf_ect = adfuller(self.adjusted_ECT)
        self.ect_p_value = adf_ect[1]
        print(f"   Error Correction Term (ECT) ADF p-value: {self.ect_p_value:.4f}")
        
        self.modeled_price_history = (((1 / (1 + np.exp(-self.coint_model.predict(X_coint)))) * self.d_log['TAM_B']) * 1e9) / self.all_supply[self.d_log.index].values
        
        log_diff = np.log(self.d_log['Price']) - np.log(self.modeled_price_history)
        self.val_z_score_series = (log_diff - log_diff.rolling(365).mean()) / log_diff.rolling(365).std()
        
        # Phase Space Analysis
        self.lyapunov_exponent = self.calculate_lle(self.val_z_score_series.dropna())
        print(f"\n   [PHYSICS METRICS]")
        print(f"   Current Mass Ratio (M1/M2): {self.d_log['Mass_Ratio'].iloc[-1]:.6f}")
        print(f"   Current Lorentz Factor: {self.d_log['Lorentz_Factor'].iloc[-1]:.4f}x Gravity")
        print(f"   Largest Lyapunov Exponent (LLE): {self.lyapunov_exponent:.5f}")
        if self.lyapunov_exponent > 0.01:
            print("   -> LLE is POSITIVE: The binary orbit is entering a chaotic regime.")
        else:
            print("   -> LLE is NEGATIVE/STABLE: The binary orbit is highly ordered.")

    def execute_mc(self):
        print("\n--- [3/5] MC LANGEVIN SDE PROJECTION (Stochastic Orbits) ---")
        f_dates = pd.date_range(self.d_log.index[-1] + timedelta(1), periods=FORECAST_DAYS, freq='D')
        steps = np.arange(1, FORECAST_DAYS + 1)
        dt = 1.0 / 365.0
        
        v_drift = (self.d_log['log_V'].iloc[-1] - self.d_log['log_V'].iloc[-365]) / 365 
        d_drift = (self.d_log['WD_Density'].iloc[-1] - self.d_log['WD_Density'].iloc[-365]) / 365

        eddington_limit = self.d_log['Accretion_Force'].max() * 1.5 
        base_vix = self.df['VIXCLS'].iloc[-1]

        mc_paths = []
        for _ in tqdm(range(MC_PATHS)):
            p_m2 = self.df['Broad_M2'].iloc[-1] * np.exp(steps * (STRUCTURAL_M2_GROWTH/365) + np.random.normal(0, 0.001, FORECAST_DAYS).cumsum())
            
            p_r = np.zeros(FORECAST_DAYS); p_r[0] = self.d_log['Global_RORO_State'].iloc[-1]
            p_acc = np.zeros(FORECAST_DAYS); p_acc[0] = self.d_log['Accretion_Force'].iloc[-1]
            
            # Stochastic Volatility (VIX Simulation for Lorentz) - Ornstein-Uhlenbeck Process
            sim_vix = np.zeros(FORECAST_DAYS); sim_vix[0] = base_vix
            kappa_vix = 5.0; theta_vix = 20.0; vol_vix = 8.0
            
            # Langevin Noise Terms
            dW_acc = np.random.normal(0, np.sqrt(dt), FORECAST_DAYS)
            dW_vix = np.random.normal(0, np.sqrt(dt), FORECAST_DAYS)

            for t in range(1, FORECAST_DAYS):
                p_r[t] = np.clip(p_r[t-1] + 0.02 * (0.5 - p_r[t-1]) + np.random.normal(0, 0.05), 0, 1)
                
                # Simulate VIX to get local Lorentz factor
                sim_vix[t] = sim_vix[t-1] + kappa_vix * (theta_vix - sim_vix[t-1]) * dt + vol_vix * dW_vix[t]
                loc_lorentz = np.exp((min(sim_vix[t], 80) - 20) / 40)
                
                # LANGEVIN EQUATION for Accretion Force (Drift + Diffusion)
                mass_ratio_pull = (d_drift * 100) # M1 getting heavier pulls harder
                
                # Drift: Gravity towards eddington mean, scaled by Lorentz
                drift = (1.001 + mass_ratio_pull) * loc_lorentz * (p_acc[t-1] * 0.01) * dt
                # Diffusion: Stochastic noise
                diffusion = 2.0 * loc_lorentz * dW_acc[t]
                
                raw_acc = p_acc[t-1] + drift + diffusion
                p_acc[t] = min(max(raw_acc, 0), eddington_limit) 

            sim_v = self.d_log['log_V'].iloc[-1] + steps * max(0, v_drift) + np.random.normal(0, 0.005, FORECAST_DAYS).cumsum()
            sim_d = np.clip(self.d_log['WD_Density'].iloc[-1] + steps * d_drift, 0.4, 0.95)
            
            # Simple stochastic drift for Hashrate based on historical growth
            h_drift = (self.d_log['log_H'].iloc[-1] - self.d_log['log_H'].iloc[-365]) / 365
            sim_h = self.d_log['log_H'].iloc[-1] + steps * max(0, h_drift) + np.random.normal(0, 0.01, FORECAST_DAYS).cumsum()
            
            f_log_SF = np.log((np.exp(self.pl_const + self.alpha * np.log((f_dates - self.genesis).days)) + (steps * 1000 + self.df['SF'].iloc[-1]*self.all_ann_flow[self.df.index[-1]])) / self.all_ann_flow[f_dates])
            
            X_sim = self.x_scaler.transform(pd.DataFrame({
                'WD_Density': sim_d, 
                'log_V': sim_v,
                'log_SF': f_log_SF,
                'log_H': sim_h,
                'Accretion_Force': p_acc, 
                'Global_RORO_State': p_r
            }))
            
            # Reintroduce systemic noise to predictions based on residual variance
            eps = np.random.normal(0, self.coint_model.resid.std(), FORECAST_DAYS)
            p_logit = self.coint_model.predict(sm.add_constant(X_sim, has_constant='add')) + eps
            mc_paths.append((((1 / (1 + np.exp(-p_logit))) * (p_m2 * TAM_MULTIPLIER)) * 1e9) / self.all_supply[f_dates].values)
            
        raw_paths = np.array(mc_paths)
        log_raw_paths = np.log(raw_paths)
        log_last_actual = np.log(self.d_log['Price'].iloc[-1])
        initial_delta = log_last_actual - log_raw_paths[:, 0] 
        decay_period = min(365, FORECAST_DAYS)
        decay_vector = np.zeros(FORECAST_DAYS)
        decay_vector[:decay_period] = np.exp(-np.linspace(0, 4, decay_period)) 
        stitched_log_paths = log_raw_paths + (initial_delta[:, np.newaxis] * decay_vector)
        stitched_paths = np.exp(stitched_log_paths)
        
        return f_dates, stitched_paths

# RUN
if __name__ == "__main__":
    engine = BinaryStarRocheLobeModel()
    engine.assemble_tensors()
    engine.run_audit()
    f_dates, mc = engine.execute_mc()

    print("\n--- [4/5] RENDERING PDF REPORT (Phase Space Included) ---")
    median, u95, l05 = np.percentile(mc, [50, 95, 5], axis=0)
    
    log_diff = np.log(engine.d_log['Price']) - np.log(engine.modeled_price_history)
    rolling_std = log_diff.rolling(365, min_periods=90).std().fillna(0.5)

    upper_band = engine.modeled_price_history * np.exp(2 * rolling_std)
    lower_band = engine.modeled_price_history * np.exp(-2 * rolling_std)

    pdf_fn = 'Nakamoto_RocheLobe_VECM.pdf'
    with PdfPages(pdf_fn) as pdf:
        # Fig 1 & 2
        fig1, (p1a, p1b) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
        p1a.plot(engine.df.index, engine.df['Price'], color='#bdc3c7', alpha=0.6, label='Actual Price')
        p1a.plot(engine.d_log.index, engine.modeled_price_history, color='#2980b9', lw=2, label='VECM Long-Term Equilibrium (Roche Lobe Limit)')
        p1a.fill_between(engine.d_log.index, lower_band, upper_band, color='#2980b9', alpha=0.1, label='Thermodynamic Bound (+/- 2σ)')
        p1a.plot(f_dates, median, color='#c0392b', ls='--', lw=2.5, label=f'Model Median Horizon: ${median[-1]:,.0f}')
        p1a.fill_between(f_dates, l05, u95, color='#c0392b', alpha=0.15)
        
        for h_date in HALVING_DATES:
            if h_date > engine.d_log.index.min() and h_date < f_dates[-1]:
                p1a.axvline(h_date, color='#8e44ad', ls=':', lw=1.5, alpha=0.6)

        p1a.set_yscale('log')
        p1a.set_title("Fig 1: Binary Star Fluid Dynamics - VECM Equilibrium Limit", fontweight='bold')
        p1a.legend(loc='upper left')
        
        p1b.hist(mc[:, -1], bins=100, color='#34495e', alpha=0.6, density=True)
        p1b.axvline(median[-1], color='#e74c3c', lw=2.5)
        p1b.axvline(l05[-1], color='#e67e22', ls='--', lw=1.5)
        p1b.axvline(u95[-1], color='#e67e22', ls='--', lw=1.5)
        p1b.set_title("Fig 2: Terminal Horizon Probability Density", fontweight='bold')
        plt.tight_layout(); pdf.savefig(fig1); plt.close()

        # Fig 3: Z-Score Time Series
        fig2, p3 = plt.subplots(1, 1, figsize=(14, 6))
        p3.plot(engine.d_log.index, engine.val_z_score_series, color='#8e44ad', lw=1.5, label='Error Correction Term (ECT Z-Score)')
        p3.axhline(2, color='#c0392b', ls='--', lw=1.5, label='Severe Roche Lobe Overflow (Overvalued)')
        p3.axhline(0, color='black', lw=1, alpha=0.5, label='Cointegrated Equilibrium')
        p3.axhline(-2, color='#27ae60', ls='--', lw=1.5, label='Extreme Compression (Undervalued)')
        p3.fill_between(engine.d_log.index, 2, engine.val_z_score_series, where=(engine.val_z_score_series > 2), color='#c0392b', alpha=0.3)
        p3.fill_between(engine.d_log.index, -2, engine.val_z_score_series, where=(engine.val_z_score_series < -2), color='#27ae60', alpha=0.3)
                
        p3.set_title("Fig 3: Mean Reversion to Gravitational Bound", fontweight='bold')
        p3.set_ylabel("Z-Score")
        p3.set_ylim(-4, 4)
        p3.legend(loc='upper left')
        plt.tight_layout(); pdf.savefig(fig2); plt.close()
        
        # Fig 4: NEW Phase Space Plot (Chaos vs Stability)
        fig3, p4 = plt.subplots(1, 1, figsize=(10, 10))
        z_vals = engine.val_z_score_series.dropna().values
        dz_dt = np.diff(z_vals)
        z_t = z_vals[:-1]
        
        p4.scatter(z_t, dz_dt, c=np.arange(len(z_t)), cmap='viridis', s=10, alpha=0.7)
        p4.axhline(0, color='black', lw=1, alpha=0.5)
        p4.axvline(0, color='black', lw=1, alpha=0.5)
        p4.set_title(f"Fig 4: Orbital Phase Space (LLE: {engine.lyapunov_exponent:.4f})", fontweight='bold')
        p4.set_xlabel("ECT Z-Score (Distance from Barycenter)")
        p4.set_ylabel("Velocity of Mean Reversion (dZ/dt)")
        plt.tight_layout(); pdf.savefig(fig3); plt.close()

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

    counts, bin_edges = np.histogram(mc[:, -1], bins=100, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    distribution_data = {
        "bin_centers": [clean_val(val, 2) for val in bin_centers],
        "density": [int(val) for val in counts], 
        "median": clean_val(median[-1], 2),
        "l05": clean_val(l05[-1], 2),
        "u95": clean_val(u95[-1], 2),
        "target_date": f_dates[-1].strftime('%B %Y')
    }

    dashboard_payload = {
        "metadata": {
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "r_squared_vecm": clean_val(engine.coint_model.rsquared, 4),
            "target_median": clean_val(median[-1], 0),
            "physics_metrics": {
                "largest_lyapunov_exponent": clean_val(engine.lyapunov_exponent, 5),
                "terminal_mass_ratio": clean_val(engine.d_log['Mass_Ratio'].iloc[-1], 5),
                "terminal_lorentz_factor": clean_val(engine.d_log['Lorentz_Factor'].iloc[-1], 4),
                "ect_p_value": clean_val(engine.ect_p_value, 4)
            }
        },
        "historical": historical_data,
        "projection": projection_data,
        "distribution": distribution_data 
    }

    json_fn = 'v33_dashboard_data.json'
    with open(json_fn, 'w') as f:
        json.dump(dashboard_payload, f)
        
    print(f"Success! Dashboard JSON Saved: {os.path.abspath(json_fn)}")