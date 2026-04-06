import os, requests, warnings, time, json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- CONFIGURATION THE RESONANT MASTER ---
FRED_API_KEY = '---' # Ajouter sa clé API FRED ICI
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
    "axes.labelsize": 10, "axes.titlesize": 12, "legend.fontsize": 9
})

# ==========================================
# GESTIONNAIRE DE CACHE UNIVERSEL (24 HEURES)
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
            print(f"   [{col_name}] Chargement depuis le cache (<24h)...")
            return pd.read_csv(fpath, index_col=0, parse_dates=True)
            
        print(f"   [{col_name}] Téléchargement Blockchain.info...")
        try:
            url = f"https://api.blockchain.info/charts/{chart_name}?timespan=all&format=json"
            data = requests.get(url, timeout=15).json()['values']
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['x'], unit='s').dt.normalize()
            df = df.set_index('date').rename(columns={'y': col_name})[[col_name]].resample('D').mean().ffill()
            df.to_csv(fpath)
            return df
        except Exception as e:
            print(f"   [{col_name}] Erreur ({e}). Repli sur ancien cache...")
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            return pd.DataFrame()

    def get_price(self):
        fname = "price_tv.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname):
            print("   [Price] Chargement depuis le cache (<24h)...")
            return pd.read_csv(fpath, index_col=0, parse_dates=True)
        print("   [Price] Téléchargement TvDatafeed...")
        try:
            tv = TvDatafeed()
            p = tv.get_hist(symbol="BTCUSD", exchange="INDEX", interval=Interval.in_daily, n_bars=8000)
            df = p.rename(columns={'close': 'Price'})[['Price']].resample('D').mean().ffill()
            df.index = pd.to_datetime(df.index).normalize()
            df.to_csv(fpath)
            return df
        except Exception as e:
            print(f"   [Price] Erreur ({e}). Repli sur ancien cache...")
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            return pd.DataFrame()

    def get_yf_close(self, tickers, name):
        fname = f"yf_{name}.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname):
            print(f"   [{name}] Chargement depuis le cache (<24h)...")
            return pd.read_csv(fpath, index_col=0, parse_dates=True)
        print(f"   [{name}] Téléchargement YFinance...")
        try:
            df = yf.download(tickers, start="2010-01-01", progress=False)
            if 'Close' in df.columns: df = df['Close']
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            df.to_csv(fpath)
            return df
        except Exception as e:
            print(f"   [{name}] Erreur ({e}). Repli sur ancien cache...")
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            return pd.DataFrame()
            
    def get_ibit_vol(self):
        fname = "yf_ibit.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname):
            print("   [IBIT] Chargement depuis le cache (<24h)...")
            return pd.read_csv(fpath, index_col=0, parse_dates=True)
        print("   [IBIT] Téléchargement YFinance...")
        try:
            df = yf.download("IBIT", start="2024-01-11", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            vol = (df['Close'] * df['Volume']).resample('D').mean().ffill()
            vol = vol.to_frame(name='IBIT_Vol')
            vol.to_csv(fpath)
            return vol
        except Exception as e:
            print(f"   [IBIT] Erreur ({e}). Repli sur ancien cache...")
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            return pd.DataFrame()

    def get_fred(self):
        fname = "fred_macro.csv"
        fpath = os.path.join(self.cache_dir, fname)
        if self._is_valid(fname):
            print("   [Macro] Chargement FRED depuis le cache (<24h)...")
            return pd.read_csv(fpath, index_col=0, parse_dates=True)
        
        print("   [Macro] Téléchargement API FRED Authentifiée...")
        series_list = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'M2SL', 'VIXCLS']
        df_list = []
        try:
            for s in series_list:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s}&api_key={FRED_API_KEY}&file_type=json&observation_start=2009-01-03"
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                data = r.json()
                temp = pd.DataFrame(data['observations'])
                temp['date'] = pd.to_datetime(temp['date'])
                temp['value'] = pd.to_numeric(temp['value'], errors='coerce')
                temp = temp.set_index('date')[['value']].rename(columns={'value': s})
                df_list.append(temp)
            
            m = pd.concat(df_list, axis=1)
            m = m.resample('D').interpolate('time').ffill().fillna(0)
            m.to_csv(fpath)
            return m
        except Exception as e:
            print(f"   [Macro] Erreur FRED ({e}). Repli...")
            if os.path.exists(fpath): return pd.read_csv(fpath, index_col=0, parse_dates=True)
            
            dates = pd.date_range(start='2010-01-01', end=datetime.today(), freq='D')
            proxy_df = pd.DataFrame(index=dates)
            try:
                vix = yf.download("^VIX", start="2010-01-01", progress=False)
                if 'Close' in vix: vix = vix['Close']
                if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.droplevel(1)
                proxy_df['VIXCLS'] = vix.reindex(dates).ffill()
            except: proxy_df['VIXCLS'] = 20.0
            
            days_array = np.arange(len(dates))
            proxy_df['M2SL'] = 8000.0 * np.exp(days_array * (STRUCTURAL_M2_GROWTH/365)) 
            proxy_df['WALCL'] = proxy_df['M2SL'] * 0.4 
            proxy_df['WTREGEN'] = 0; proxy_df['RRPONTSYD'] = 0
            return proxy_df.fillna(method='bfill')

# ==========================================
# MOTEUR PRINCIPAL V33
# ==========================================
class NakamotoV33Singularity:
    def __init__(self):
        self.genesis = pd.to_datetime('2009-01-03')
        self.df = None
        self.model = None
        self.x_scaler = StandardScaler()
        self.modeled_price_history = None
        self.val_z_score_series = None
        self.fetcher = DataFetcher()

    def assemble_tensors(self):
        print("--- [1/5] ASSEMBLING TENSORS (V33 Resonant Logic) ---")
        price_df = self.fetcher.get_price()
        price_df['MA200'] = price_df['Price'].rolling(200).mean()

        h = self.fetcher.get_blockchain_metric('hash-rate', 'Hashrate')
        v = self.fetcher.get_blockchain_metric('n-unique-addresses', 'Velocity')
        tx = self.fetcher.get_blockchain_metric('estimated-transaction-volume-usd', 'Tx_Vol_USD')
        m = self.fetcher.get_fred()
        
        m['Net_Liquidity'] = (m['WALCL'] - m['WTREGEN'] - m['RRPONTSYD']).ewm(span=14).mean()
        m['Liq_Momentum'] = m['Net_Liquidity'] / m['Net_Liquidity'].rolling(90).mean()
        m['Broad_M2'] = m['M2SL']
        
        safe_haven = self.fetcher.get_yf_close(["GLD", "DX-Y.NYB"], "safehaven")
        m['Fiat_Defiance'] = (safe_haven['GLD'] / safe_haven['DX-Y.NYB']).ffill().reindex(m.index).ffill()
        m['Fiat_Defiance_Mom'] = m['Fiat_Defiance'] / m['Fiat_Defiance'].rolling(90).mean()
        
        ibit_vol = self.fetcher.get_ibit_vol()

        dates_all = pd.date_range(self.genesis, datetime.today() + timedelta(days=FORECAST_DAYS), freq='D')
        epochs = np.floor((dates_all - self.genesis).days / 1458.33)
        flow = pd.Series(144 * (50.0 / (2.0 ** epochs)), index=dates_all)
        self.all_supply = flow.cumsum(); self.all_ann_flow = flow.rolling(365).sum().bfill()

        # ==============================================================
        # CORRECTION DE LA TRONCATURE TEMPORELLE
        # ==============================================================
        h_elastic_raw = np.where((h['Hashrate'].rolling(30).mean() < h['Hashrate'].rolling(60).mean()), h['Hashrate'].shift(-60), h['Hashrate'].shift(-245))
        h['H_Elastic'] = pd.Series(h_elastic_raw, index=h.index).ffill()
        # ==============================================================

        self.df = price_df.join([h, m, v, tx], how='inner').dropna()
        
        realized_vol = self.df['Price'].pct_change().rolling(30).std().fillna(0) * np.sqrt(365)
        self.df['Gamma_Proxy'] = (realized_vol * self.df['VIXCLS']).ewm(span=14).mean()
        
        safe_velocity = self.df['Velocity'].replace(0, np.nan).bfill()
        baseline_tx_size = (self.df['Tx_Vol_USD'] / safe_velocity).rolling(90, min_periods=1).median().bfill().replace(0, 5000)
        self.df['V_eff'] = self.df['Velocity'] + (ibit_vol['IBIT_Vol'].reindex(self.df.index).fillna(0).rolling(14, min_periods=1).mean() / baseline_tx_size)
        
        roro_assets = self.fetcher.get_yf_close(["HYG", "IEF", "COPX", "GLD", "SPHB", "SPLV"], "roro_assets")
        roro_assets = roro_assets.reindex(self.df.index).ffill().bfill()
        
        cr, gr, er = roro_assets['HYG'] / roro_assets['IEF'], roro_assets['COPX'] / roro_assets['GLD'], roro_assets['SPHB'] / roro_assets['SPLV']
        def rz(s): return (s - s.rolling(90).mean()) / (s.rolling(90).std() + 1e-8)
        roro_composite = rz(cr) + rz(gr) + rz(er)
        self.df['Global_RORO_State'] = (1 / (1 + np.exp(-roro_composite))).ewm(span=30).mean()

        hoarding_rate = np.where(self.df['Price'] < self.df['MA200'], 2500, 500)
        corp_h = pd.Series(hoarding_rate, index=self.df.index).cumsum()
        corp_h[self.df.index < '2020-08-11'] = 0 
        
        hodl_raw = (self.all_supply[self.df.index] - (self.df['Tx_Vol_USD'] / self.df['Price']).rolling(30).mean() - corp_h).dropna()
        res_a = sm.OLS(np.log(hodl_raw), sm.add_constant(np.log((hodl_raw.index - self.genesis).days))).fit()
        self.pl_const, self.alpha = res_a.params
        self.df['SF'] = (np.exp(self.pl_const + self.alpha * np.log((self.df.index - self.genesis).days)) + corp_h) / self.all_ann_flow[self.df.index]

    def run_audit(self):
        print("--- [2/5] PURE PHYSICS AUDIT & VIF VALIDATION ---")
        self.df['TAM_B'] = self.df['Broad_M2'] * TAM_MULTIPLIER
        self.df['Rho'] = (((self.df['Price'] * self.all_supply[self.df.index]) / 1e9) / self.df['TAM_B']).clip(upper=0.999)
        self.df['Logit_Rho'] = np.log(self.df['Rho'] / (1 - self.df['Rho']))
        
        audit_data = pd.DataFrame({
            'Logit_Rho': self.df['Logit_Rho'], 'log_H': np.log(self.df['H_Elastic']),
            'log_SF': np.log(self.df['SF']), 'log_V_eff': np.log(self.df['V_eff']),
            'Liq_Momentum': self.df['Liq_Momentum'], 'Fiat_Defiance': self.df['Fiat_Defiance_Mom'], 
            'Gamma_Proxy': self.df['Gamma_Proxy'], 'Global_RORO_State': self.df['Global_RORO_State'], 
            'Price': self.df['Price'], 'TAM_B': self.df['TAM_B']
        }).dropna()
        
        self.df['Price_Momentum'] = self.df['Price'] / self.df['Price'].rolling(90).mean()
        audit_data['Price_Momentum'] = audit_data['Price'] / audit_data['Price'].rolling(90).mean()
        
        v_model = sm.OLS(np.log(audit_data['Price']), sm.add_constant(audit_data['log_V_eff'])).fit()
        self.metcalfe_beta = v_model.params['log_V_eff']
        self.v_model_params_const = v_model.params['const']
        audit_data['Network_Power'] = v_model.predict() 
        
        h_prem_res = sm.OLS(np.log(self.df['H_Elastic'].reindex(audit_data.index)), sm.add_constant(audit_data['log_SF'])).fit()
        audit_data['H_Accumulated'] = h_prem_res.resid.ewm(span=90).mean()
        self.h_premium_model = h_prem_res
        
        self.X_cols = ['log_SF', 'H_Accumulated', 'Network_Power', 'Liq_Momentum', 'Fiat_Defiance', 'Gamma_Proxy', 'Global_RORO_State']
        self.d_log = audit_data 
        
        X_scaled_df = pd.DataFrame(self.x_scaler.fit_transform(self.d_log[self.X_cols]), columns=self.X_cols, index=self.d_log.index)
        X_vif = sm.add_constant(X_scaled_df)
        self.model = sm.OLS(self.d_log['Logit_Rho'], X_vif).fit(cov_type='HC3')
        
        self.modeled_price_history = (((1 / (1 + np.exp(-self.model.predict(X_vif)))) * self.d_log['TAM_B']) * 1e9) / self.all_supply[self.d_log.index].values
        
        log_diff = np.log(self.d_log['Price']) - np.log(self.modeled_price_history)
        days_since_start = (self.d_log.index - self.genesis).days.values
        adaptive_window = np.maximum(90, 700 - (days_since_start / 10)).astype(int)
        
        val_z_score = []
        for i in range(len(log_diff)):
            w = adaptive_window[i]
            if i < w: val_z_score.append(0) 
            else:
                window_data = log_diff.iloc[i-w:i]
                val_z_score.append((log_diff.iloc[i] - window_data.mean()) / (window_data.std() + 1e-8))
        self.val_z_score_series = pd.Series(val_z_score, index=self.d_log.index)
        
        vif_data = pd.DataFrame({"Tenseur": X_vif.columns, "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]})
        print(vif_data.to_string(index=False))
        print(f"\nUnified Network Beta: {self.metcalfe_beta:.4f} | R2: {self.model.rsquared:.4f}")

    def execute_mc(self):
        print("\n--- [3/5] MC GARCH PROJECTION ---")
        f_dates = pd.date_range(self.d_log.index[-1] + timedelta(1), periods=FORECAST_DAYS, freq='D')
        steps = np.arange(1, FORECAST_DAYS + 1)
        
        h_drift = (np.log(self.df['H_Elastic']).iloc[-1] - np.log(self.df['H_Elastic']).iloc[-2920]) / 2920
        v_drift = (self.d_log['log_V_eff'].iloc[-1] - self.d_log['log_V_eff'].iloc[-365]) / 365 
        
        g_m = self.d_log['Gamma_Proxy'].mean(); f_m = self.d_log['Fiat_Defiance'].mean()

        omega = 0.00005; alpha_g = 0.15; beta_g = 0.80; mc_paths = []
        for _ in tqdm(range(MC_PATHS)):
            p_m2 = self.df['Broad_M2'].iloc[-1] * np.exp(steps * (STRUCTURAL_M2_GROWTH/365) + np.random.normal(0, 0.001, FORECAST_DAYS).cumsum())
            p_g = np.zeros(FORECAST_DAYS); p_l = np.zeros(FORECAST_DAYS); p_f = np.zeros(FORECAST_DAYS); p_r = np.zeros(FORECAST_DAYS)
            p_g[0], p_l[0], p_f[0], p_r[0] = self.d_log['Gamma_Proxy'].iloc[-1], self.d_log['Liq_Momentum'].iloc[-1], self.d_log['Fiat_Defiance'].iloc[-1], self.d_log['Global_RORO_State'].iloc[-1]
            p_ha = np.zeros(FORECAST_DAYS); p_ha[0] = self.d_log['H_Accumulated'].iloc[-1]
            sigma2 = np.zeros(FORECAST_DAYS); eps = np.zeros(FORECAST_DAYS); sigma2[0] = self.model.resid.std()**2; eps[0] = np.random.normal(0, np.sqrt(sigma2[0]))

            for t in range(1, FORECAST_DAYS):
                p_g[t] = max(0.01, p_g[t-1] + 0.10*(g_m - p_g[t-1]) + np.random.normal(0, 0.1))
                p_l[t] = p_l[t-1] + 0.05*(1.0 - p_l[t-1]) + np.random.normal(0, 0.01)
                p_f[t] = p_f[t-1] + 0.02*(f_m - p_f[t-1]) + np.random.normal(0, 0.02)
                p_r[t] = np.clip(p_r[t-1] + 0.02 * (0.5 - p_r[t-1]) + np.random.normal(0, 0.05), 0, 1)
                sigma2[t] = omega + alpha_g * eps[t-1]**2 + beta_g * sigma2[t-1]
                eps[t] = np.random.normal(0, np.sqrt(sigma2[t]))

            f_log_SF = np.log((np.exp(self.pl_const + self.alpha * np.log((f_dates - self.genesis).days)) + (steps * 1000 + self.df['SF'].iloc[-1]*self.all_ann_flow[self.df.index[-1]])) / self.all_ann_flow[f_dates])
            raw_h_p = (np.log(self.df['H_Elastic']).iloc[-1] + steps * h_drift) - self.h_premium_model.predict(sm.add_constant(f_log_SF))
            for t in range(1, FORECAST_DAYS): p_ha[t] = p_ha[t-1] * 0.98 + raw_h_p[t] * 0.02
            
            sim_net_power = self.v_model_params_const + self.metcalfe_beta * (self.d_log['log_V_eff'].iloc[-1] + steps * max(0, v_drift) + np.random.normal(0, 0.005, FORECAST_DAYS).cumsum())
            
            X_sim = self.x_scaler.transform(pd.DataFrame({
                'log_SF': f_log_SF, 'H_Accumulated': p_ha, 'Network_Power': sim_net_power,
                'Liq_Momentum': p_l, 'Fiat_Defiance': p_f, 
                'Gamma_Proxy': p_g, 'Global_RORO_State': p_r
            }))
            p_logit = self.model.predict(sm.add_constant(X_sim, has_constant='add')) + eps
            mc_paths.append((((1 / (1 + np.exp(-p_logit))) * (p_m2 * TAM_MULTIPLIER)) * 1e9) / self.all_supply[f_dates].values)
            
       # =========================================================================
        # LE VRAI FIX : LE RACCORDEMENT ADDITIF LOGARITHMIQUE (LOG-SPACE SHIFT)
        # =========================================================================
        raw_paths = np.array(mc_paths) # Shape: (5000, 730)
        
        # 1. Obtenir les prix en log
        log_raw_paths = np.log(raw_paths)
        log_last_actual = np.log(self.d_log['Price'].iloc[-1]) # Point de départ réel
        
        # 2. Calculer le Delta (l'écart) initial au jour 0 de la projection
        # CORRECTION ICI : [:, 0] prend le jour 0 des 5000 chemins (shape: 5000,)
        initial_delta = log_last_actual - log_raw_paths[:, 0] 
        
        # 3. Créer une fonction de "Decay" (dissipation) sur 1 an (365 jours)
        decay_period = min(365, FORECAST_DAYS)
        decay_vector = np.zeros(FORECAST_DAYS)
        decay_vector[:decay_period] = np.exp(-np.linspace(0, 4, decay_period)) 
        
        # 4. Appliquer ce Delta décroissant à toute la trajectoire théorique
        # initial_delta[:, np.newaxis] devient (5000, 1), multiplié par decay_vector (730,) 
        # donne bien une matrice d'ajustement (5000, 730) !
        stitched_log_paths = log_raw_paths + (initial_delta[:, np.newaxis] * decay_vector)
        
        # 5. Revenir en espace linéaire (Prix en $)
        stitched_paths = np.exp(stitched_log_paths)
        
        return f_dates, stitched_paths

# RUN
if __name__ == "__main__":
    engine = NakamotoV33Singularity()
    engine.assemble_tensors(); engine.run_audit()
    f_dates, mc = engine.execute_mc()

    print("\n--- [4/5] RENDERING PDF REPORT ---")
    median, u95, l05 = np.percentile(mc, [50, 95, 5], axis=0)
    X_hist_scaled = sm.add_constant(pd.DataFrame(engine.x_scaler.transform(engine.d_log[engine.X_cols]), columns=engine.X_cols, index=engine.d_log.index))
    hist_p_fitted = engine.modeled_price_history

    log_diff = np.log(engine.d_log['Price']) - np.log(hist_p_fitted)
    days_since_start = (engine.d_log.index - engine.genesis).days.values
    adaptive_window = np.maximum(90, 700 - (days_since_start / 10)).astype(int)
    rolling_std = pd.Series([log_diff.iloc[max(0, i-adaptive_window[i]):i].std() if i >= adaptive_window[i] else 0 for i in range(len(log_diff))], index=engine.d_log.index)

    upper_band = hist_p_fitted * np.exp(2 * rolling_std)
    lower_band = hist_p_fitted * np.exp(-2 * rolling_std)
    howell_t = (engine.d_log.index - pd.to_datetime('2020-03-23')).days.values
    howell_sine = 0.05 * np.sin(2 * np.pi * howell_t / HOWELL_CYCLE_DAYS) + 1.0 

    pdf_fn = 'Nakamoto_V33_Global_RORO.pdf'
    with PdfPages(pdf_fn) as pdf:
        # --- PAGE 1: ENVELOPE PRICE & DENSITY ---
        fig1, (p1a, p1b) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
        p1a.plot(engine.df.index, engine.df['Price'], color='#bdc3c7', alpha=0.6, label='Actual Price')
        p1a.plot(engine.d_log.index, hist_p_fitted, color='#2980b9', lw=2, label='V33 Fundamental State (RORO Powered)')
        p1a.fill_between(engine.d_log.index, lower_band, upper_band, color='#2980b9', alpha=0.1, label='Thermodynamic Envelope (+/- 2σ)')
        p1a.plot(f_dates, median, color='#c0392b', ls='--', lw=2.5, label=f'2028 Median: ${median[-1]:,.0f}')
        p1a.fill_between(f_dates, l05, u95, color='#c0392b', alpha=0.15)
        
        for h_date in HALVING_DATES:
            if h_date > engine.d_log.index.min() and h_date < f_dates[-1]:
                p1a.axvline(h_date, color='#8e44ad', ls=':', lw=1.5, alpha=0.6)
        p1a.text(HALVING_DATES[-1] - timedelta(days=200), 1e6, '2028 Halving Epoch', color='#8e44ad', rotation=90)

        p1a.set_yscale('log'); p1a.set_title("Fig 1: Nakamoto Accretion - V33 Global RORO Integration", fontweight='bold'); p1a.legend(loc='upper left')
        
        p1b.hist(mc[:, -1], bins=100, color='#34495e', alpha=0.6, density=True)
        p1b.axvline(median[-1], color='#e74c3c', lw=2.5)
        p1b.axvline(l05[-1], color='#e67e22', ls='--', lw=1.5, label=f'5th Pctl: ${l05[-1]:,.0f}')
        p1b.axvline(u95[-1], color='#e67e22', ls='--', lw=1.5, label=f'95th Pctl: ${u95[-1]:,.0f}')
        p1b.set_title("Fig 2: April 2028 Probability Density", fontweight='bold')
        plt.tight_layout(); pdf.savefig(fig1); plt.close()

        # --- PAGE 2: TENSORS & HOWELL ---
        fig2, (p2a, p2b) = plt.subplots(2, 1, figsize=(14, 14))
        
        p2a.plot(engine.d_log.index, engine.model.params['Network_Power'] * X_hist_scaled['Network_Power'], color='#8e44ad', label='Metcalfe Network Force')
        p2a.plot(engine.d_log.index, engine.model.params['log_SF'] * X_hist_scaled['log_SF'], color='#2980b9', label='Scarcity Piston')
        p2a.plot(engine.d_log.index, engine.model.params['Global_RORO_State'] * X_hist_scaled['Global_RORO_State'], color='#e67e22', alpha=0.8, label='Global RORO State (Project10X)')
        p2a.set_title("Fig 3: Beta-Weighted Structural Components (V33 Exogenous)", fontweight='bold')
        p2a.legend(loc='upper left')

        p2b.plot(engine.d_log.index, howell_sine, color='black', ls=':', lw=1, alpha=0.3, label='Howell Cycle (~65m)')
        p2b.plot(engine.d_log.index, engine.d_log['Liq_Momentum'], color='#27ae60', lw=1.5, alpha=0.8, label='Net Liquidity Momentum (90d)')
        p2bx = p2b.twinx()
        p2bx.plot(engine.d_log.index, engine.d_log['Price_Momentum'], color='#f39c12', lw=1.2, alpha=0.7, label='BTC Price Momentum (90d)')
        p2b.axhline(1.0, color='black', lw=1, alpha=0.5)
        p2b.set_title("Fig 4: Harmonic Resonance - Price Momentum vs Liquidity Momentum", fontweight='bold')
        p2b.set_ylabel("Liquidity Index"); p2bx.set_ylabel("BTC Price Index")
        lines, labels = p2b.get_legend_handles_labels(); lines2, labels2 = p2bx.get_legend_handles_labels()
        p2b.legend(lines + lines2, labels + labels2, loc='upper left', ncol=3)
        plt.tight_layout(); pdf.savefig(fig2); plt.close()
        
        # --- PAGE 3: ACTIONABLE SIGNAL (ADAPTIVE Z-SCORE) ---
        fig3, p3 = plt.subplots(1, 1, figsize=(14, 6))
        p3.plot(engine.d_log.index, engine.val_z_score_series, color='#8e44ad', lw=1.5, label='Adaptive Valuation Z-Score')
        p3.axhline(2, color='#c0392b', ls='--', lw=1.5, label='Extreme Overvaluation (Sell Zone)')
        p3.axhline(0, color='black', lw=1, alpha=0.5, label='Fair Value Equilibrium')
        p3.axhline(-2, color='#27ae60', ls='--', lw=1.5, label='Extreme Undervaluation (Buy Zone)')
        p3.fill_between(engine.d_log.index, 2, engine.val_z_score_series, where=(engine.val_z_score_series > 2), color='#c0392b', alpha=0.3)
        p3.fill_between(engine.d_log.index, -2, engine.val_z_score_series, where=(engine.val_z_score_series < -2), color='#27ae60', alpha=0.3)
        
        for h_date in HALVING_DATES:
            if h_date > engine.d_log.index.min() and h_date < engine.d_log.index.max():
                p3.axvline(h_date, color='#34495e', ls=':', lw=1.5, alpha=0.5)
                
        p3.set_title("Fig 5: V33 Adaptive Thermodynamic Divergence (Actionable Signal & Epochs)", fontweight='bold')
        p3.set_ylabel("Z-Score (Adaptive Window)")
        p3.set_ylim(-4, 4)
        p3.legend(loc='upper left')
        plt.tight_layout(); pdf.savefig(fig3); plt.close()


    # ==========================================
    # EXPORT DES DONNÉES EN JSON POUR LE DASHBOARD WEB
    # ==========================================
    print("\n--- [5/5] GENERATING WEB DASHBOARD JSON ---")
    
    # Fonction de nettoyage pour éviter les "NaN" qui font crasher le JSON
    def clean_val(val, decimals=2):
        if pd.isna(val) or np.isinf(val):
            return None
        return round(float(val), decimals)

    historical_data = []
    for date in engine.d_log.index[::7]:
        idx = engine.d_log.index.get_loc(date)
        historical_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "actual_price": clean_val(engine.d_log['Price'].iloc[idx], 2),
            "model_price": clean_val(hist_p_fitted[idx], 2),
            "upper_band": clean_val(upper_band[idx], 2),
            "lower_band": clean_val(lower_band[idx], 2),
            "z_score": clean_val(engine.val_z_score_series.iloc[idx], 3),
            "liq_momentum": clean_val(engine.d_log['Liq_Momentum'].iloc[idx], 3),
            "price_momentum": clean_val(engine.d_log['Price_Momentum'].iloc[idx], 3)
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

    # CALCUL DE LA DISTRIBUTION FINALE (FIGURE 2)
    final_prices = mc[:, -1] 
    # FIX 1 : On passe density=False pour avoir le compte réel (les fréquences) au lieu de 0.0000x
    counts, bin_edges = np.histogram(final_prices, bins=100, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    distribution_data = {
        "bin_centers": [clean_val(val, 2) for val in bin_centers],
        # FIX 2 : Plus besoin de 8 décimales, ce sont des entiers maintenant
        "density": [int(val) for val in counts], 
        "median": clean_val(median[-1], 2),
        "l05": clean_val(l05[-1], 2),
        "u95": clean_val(u95[-1], 2),
        "target_date": f_dates[-1].strftime('%B %Y')
    }

    dashboard_payload = {
        "metadata": {
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "r_squared": clean_val(engine.model.rsquared, 4),
            "target_median_2028": clean_val(median[-1], 0)
        },
        "historical": historical_data,
        "projection": projection_data,
        "distribution": distribution_data 
    }

    json_fn = 'v33_dashboard_data.json'
    with open(json_fn, 'w') as f:
        json.dump(dashboard_payload, f)
        
    print(f"Success! Dashboard JSON Saved: {os.path.abspath(json_fn)}")