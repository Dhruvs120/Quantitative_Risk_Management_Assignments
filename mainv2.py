import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from arch import arch_model
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

"""
Financial Quantitative Risk Management - Assignment 1
Value-at-Risk (VaR) and Expected Shortfall (ES) Analysis
"""

import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm


class PortfolioRiskAnalysis:
    def __init__(self):
        # Parameters
        self.confidence_level = 0.95  # For VaR and ES
        self.start_date = "2013-01-01"  # About 10 years of data
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Portfolio constituents
        self.stocks = {
            'AAPL': 'USD',    # Apple (US)
            'MSFT': 'USD',    # Microsoft (US)
            '000001.SS': 'CNY',  # SSE Composite Index (China)
            'AEX.AS': 'EUR',  # AEX Index (Netherlands)
            'BARC.L': 'GBP'   # Barclays (UK)
        }
        
        # Interest rate benchmarks
        self.interest_rates = {
            'USD': '^IRX',  # US Treasury Yield 13 Week
            'EUR': '^EIBOR3M',  # EURIBOR 3 Month
            'GBP': '^LIBOR3M',  # LIBOR 3 Month
            'CNY': None  # Will need to be sourced alternatively
        }
        
        # Exchange rates (all vs USD)
        self.exchange_rates = ['EURUSD=X', 'GBPUSD=X', 'CNYUSD=X']
        
        # Portfolio weights (to be defined)
        self.weights = {}
        self.data = None
        self.returns = None
        self.portfolio_returns = None
        
    def fetch_data(self):
        """Fetch all required data and synchronize it"""
        print("Fetching data...")
        
        # Fetch stock data
        stock_data = {}
        for ticker in self.stocks.keys():
            try:
                stock_data[ticker] = yf.download(ticker, start=self.start_date, end=self.end_date)['Adj Close']
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        # Fetch interest rate data
        ir_data = {}
        for currency, ticker in self.interest_rates.items():
            if ticker:
                try:
                    ir_data[currency] = yf.download(ticker, start=self.start_date, end=self.end_date)['Adj Close'] / 100  # Convert to decimal
                except Exception as e:
                    print(f"Error fetching interest rate data for {currency}: {e}")
        
        # Fetch exchange rate data
        fx_data = {}
        for fx_pair in self.exchange_rates:
            try:
                fx_data[fx_pair] = yf.download(fx_pair, start=self.start_date, end=self.end_date)['Adj Close']
            except Exception as e:
                print(f"Error fetching data for {fx_pair}: {e}")
        
        # Combine all data
        all_data = pd.DataFrame()
        
        # Add stock prices
        for ticker, data in stock_data.items():
            all_data[f'{ticker}_price'] = data
        
        # Add interest rates
        for currency, data in ir_data.items():
            if isinstance(data, pd.Series):
                all_data[f'{currency}_rate'] = data
        
        # Add exchange rates
        for fx_pair, data in fx_data.items():
            currency = fx_pair.split('=')[0].replace('USD', '')
            all_data[f'{currency}_USD'] = data
            
        # Handle missing data
        self.data = self.synchronize_data(all_data)
        
        # Calculate returns
        self.calculate_returns()
        
        print("Data fetched and synchronized successfully.")
        return self.data
        
    def synchronize_data(self, data):
        """Synchronize data by handling missing values"""
        # Forward fill limited missing values (for single missing days)
        data_filled = data.fillna(method='ffill', limit=3)
        
        # Remove dates with too many missing values
        missing_threshold = len(data.columns) * 0.1  # If more than 10% of columns have missing data
        rows_to_keep = data_filled.isnull().sum(axis=1) <= missing_threshold
        synchronized_data = data_filled[rows_to_keep]
        
        # Final forward fill if any missing values remain
        synchronized_data = synchronized_data.fillna(method='ffill')
        
        print(f"Original data shape: {data.shape}, Synchronized data shape: {synchronized_data.shape}")
        return synchronized_data
        
    def calculate_returns(self):
        """Calculate returns for all assets and the portfolio"""
        # Calculate returns for all assets
        returns = pd.DataFrame()
        
        # Stock returns
        for ticker in self.stocks.keys():
            returns[f'{ticker}_return'] = self.data[f'{ticker}_price'].pct_change()
        
        # Interest rate changes (absolute changes)
        for currency in self.interest_rates.keys():
            if f'{currency}_rate' in self.data.columns:
                returns[f'{currency}_rate_change'] = self.data[f'{currency}_rate'].diff()
        
        # Exchange rate returns
        for fx_pair in self.exchange_rates:
            currency = fx_pair.split('=')[0].replace('USD', '')
            if f'{currency}_USD' in self.data.columns:
                returns[f'{currency}_USD_return'] = self.data[f'{currency}_USD'].pct_change()
        
        # Remove first row with NaNs
        self.returns = returns.dropna()
        
        return self.returns
        
    def set_portfolio_weights(self, weights=None):
        """Set portfolio weights for all assets"""
        if weights is None:
            # Default weights: equal weights to stocks, smaller weight to interest/forex
            assets = list(self.stocks.keys()) + [c for c in self.interest_rates.keys() if f'{c}_rate' in self.data.columns] + self.exchange_rates
            equal_weight = 1.0 / len(assets)
            weights = {asset: equal_weight for asset in assets}
        
        self.weights = weights
        print(f"Portfolio weights set: {self.weights}")
        
        # Calculate portfolio returns
        self.calculate_portfolio_returns()
        
    def calculate_portfolio_returns(self):
        """Calculate portfolio returns based on weights"""
        if self.weights is None or not self.weights:
            raise ValueError("Portfolio weights not set")
            
        # Initialize portfolio returns
        self.portfolio_returns = pd.Series(0, index=self.returns.index)
        
        # Add weighted stock returns
        for ticker, currency in self.stocks.items():
            if ticker in self.weights:
                stock_return = self.returns[f'{ticker}_return']
                
                # Convert to USD if not already in USD
                if currency != 'USD' and f'{currency}_USD_return' in self.returns.columns:
                    fx_return = self.returns[f'{currency}_USD_return']
                    # Total return in USD = (1+r_stock)*(1+r_fx) - 1
                    total_return = (1 + stock_return) * (1 + fx_return) - 1
                    self.portfolio_returns += self.weights[ticker] * total_return
                else:
                    self.portfolio_returns += self.weights[ticker] * stock_return
        
        # Add interest rate component (simplified)
        for currency, rate_ticker in self.interest_rates.items():
            if currency in self.weights and f'{currency}_rate' in self.data.columns:
                self.portfolio_returns += self.weights[currency] * self.returns[f'{currency}_rate_change']
        
        # Add direct forex exposure if any
        for fx_pair in self.exchange_rates:
            currency = fx_pair.split('=')[0].replace('USD', '')
            if fx_pair in self.weights and f'{currency}_USD_return' in self.returns.columns:
                self.portfolio_returns += self.weights[fx_pair] * self.returns[f'{currency}_USD_return']
        
        return self.portfolio_returns

    def plot_portfolio_returns(self):
        """Plot portfolio returns histogram and time series"""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated yet")
            
        plt.figure(figsize=(15, 10))
        
        # Plot time series
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_returns.index, self.portfolio_returns.values)
        plt.title('Portfolio Returns Time Series')
        plt.ylabel('Return')
        
        # Plot histogram
        plt.subplot(2, 1, 2)
        plt.hist(self.portfolio_returns.values, bins=50, density=True, alpha=0.7)
        plt.title('Portfolio Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        
        # Add normal distribution for comparison
        x = np.linspace(min(self.portfolio_returns), max(self.portfolio_returns), 100)
        mean = np.mean(self.portfolio_returns)
        std = np.std(self.portfolio_returns)
        plt.plot(x, stats.norm.pdf(x, mean, std), 'r', linewidth=2)
        
        plt.tight_layout()
        plt.show()

    # VaR and ES calculation methods
    def var_parametric_normal(self, confidence_level=None, window=None):
        """Calculate VaR using parametric method with normal distribution"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        returns = self.portfolio_returns
        if window is not None:
            returns = returns.iloc[-window:]
            
        mean = returns.mean()
        std = returns.std()
        var = -mean + std * stats.norm.ppf(1 - confidence_level)
        
        return var
        
    def es_parametric_normal(self, confidence_level=None, window=None):
        """Calculate ES using parametric method with normal distribution"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        returns = self.portfolio_returns
        if window is not None:
            returns = returns.iloc[-window:]
            
        mean = returns.mean()
        std = returns.std()
        
        # ES for normal = E[X | X < -VaR]
        var = self.var_parametric_normal(confidence_level, window)
        z_alpha = stats.norm.ppf(1 - confidence_level)
        es = -mean + std * stats.norm.pdf(z_alpha) / (1 - confidence_level)
        
        return es

    def var_parametric_t(self, df, confidence_level=None, window=None):
        """Calculate VaR using parametric method with Student-t distribution"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        returns = self.portfolio_returns
        if window is not None:
            returns = returns.iloc[-window:]
            
        mean = returns.mean()
        std = returns.std()
        
        # Scale factor for t-distribution
        scale_factor = np.sqrt((df - 2) / df)
        t_quantile = stats.t.ppf(1 - confidence_level, df)
        
        # VaR adjusted for t-distribution
        var = -mean + std * t_quantile / scale_factor
        
        return var
        
    def es_parametric_t(self, df, confidence_level=None, window=None):
        """Calculate ES using parametric method with Student-t distribution"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        returns = self.portfolio_returns
        if window is not None:
            returns = returns.iloc[-window:]
            
        mean = returns.mean()
        std = returns.std()
        
        # Scale factor for t-distribution
        scale_factor = np.sqrt((df - 2) / df)
        t_quantile = stats.t.ppf(1 - confidence_level, df)
        
        # ES for t-distribution
        var = self.var_parametric_t(df, confidence_level, window)
        constant = df + t_quantile**2
        es = -mean + std * ((df + t_quantile**2) / (df - 1)) * stats.t.pdf(t_quantile, df) / (1 - confidence_level) / scale_factor
        
        return es

    def var_historical(self, confidence_level=None, window=None):
        """Calculate VaR using historical simulation"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        returns = self.portfolio_returns
        if window is not None:
            returns = returns.iloc[-window:]
            
        var = -np.percentile(returns, 100 * (1 - confidence_level))
        return var
        
    def es_historical(self, confidence_level=None, window=None):
        """Calculate ES using historical simulation"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        returns = self.portfolio_returns
        if window is not None:
            returns = returns.iloc[-window:]
            
        var = self.var_historical(confidence_level, window)
        es = -returns[returns < -var].mean()
        return es

    def var_garch_ccc(self, confidence_level=None, forecast_horizon=1):
        """Calculate VaR using GARCH(1,1) with Constant Conditional Correlation"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        # Fit GARCH(1,1) model to each asset return
        volatilities = {}
        for col in self.returns.columns:
            try:
                model = arch_model(self.returns[col], mean='Constant', vol='GARCH', p=1, q=1)
                result = model.fit(disp='off')
                forecast = result.forecast(horizon=forecast_horizon)
                volatilities[col] = np.sqrt(forecast.variance.iloc[-1, 0])
            except Exception as e:
                print(f"Error in GARCH fitting for {col}: {e}")
                volatilities[col] = self.returns[col].std()
        
        # Calculate correlation matrix
        corr_matrix = self.returns.corr()
        
        # Construct the covariance matrix using volatilities and correlations
        cov_matrix = pd.DataFrame(index=self.returns.columns, columns=self.returns.columns)
        for i in self.returns.columns:
            for j in self.returns.columns:
                cov_matrix.loc[i, j] = volatilities[i] * volatilities[j] * corr_matrix.loc[i, j]
        
        # Calculate portfolio variance
        portfolio_var = 0
        for i in self.returns.columns:
            for j in self.returns.columns:
                if i in self.weights and j in self.weights:
                    portfolio_var += self.weights[i] * self.weights[j] * cov_matrix.loc[i, j]
        
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Calculate VaR
        portfolio_mean = sum(self.weights.get(col, 0) * self.returns[col].mean() for col in self.returns.columns)
        var = -portfolio_mean + portfolio_vol * stats.norm.ppf(1 - confidence_level) * np.sqrt(forecast_horizon)
        
        return var

    def var_filtered_historical(self, confidence_level=None, lambda_ewma=0.94):
        """Calculate VaR using Filtered Historical Simulation with EWMA"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        # Calculate standardized returns for each risk factor using EWMA volatility
        standardized_returns = pd.DataFrame(index=self.returns.index, columns=self.returns.columns)
        
        for col in self.returns.columns:
            # Calculate EWMA volatility
            returns = self.returns[col].dropna()
            var = returns.var()  # Initial variance
            vol_ewma = [np.sqrt(var)]
            
            for i in range(1, len(returns)):
                var = lambda_ewma * var + (1 - lambda_ewma) * returns.iloc[i-1]**2
                vol_ewma.append(np.sqrt(var))
                
            vol_series = pd.Series(vol_ewma, index=returns.index)
            
            # Standardize returns
            standardized_returns[col] = returns / vol_series
        
        # Calculate current EWMA volatility for each factor
        current_vol = {}
        for col in self.returns.columns:
            returns = self.returns[col].dropna()
            var = lambda_ewma * returns.iloc[-1]**2 + (1 - lambda_ewma) * returns.var()
            current_vol[col] = np.sqrt(var)
        
        # Generate simulated returns by rescaling standardized returns
        simulated_returns = pd.DataFrame(index=standardized_returns.index, columns=standardized_returns.columns)
        for col in standardized_returns.columns:
            simulated_returns[col] = standardized_returns[col] * current_vol[col]
        
        # Calculate simulated portfolio returns
        simulated_portfolio_returns = pd.Series(0, index=simulated_returns.index)
        for col in simulated_returns.columns:
            if col.split('_')[0] in self.weights:
                weight = self.weights[col.split('_')[0]]
                simulated_portfolio_returns += weight * simulated_returns[col]
        
        # Calculate VaR
        var = -np.percentile(simulated_portfolio_returns, 100 * (1 - confidence_level))
        return var

    def backtest_var(self, method='normal', **kwargs):
        """Backtest VaR using specified method"""
        # Define time windows for backtesting
        window_size = 252  # One year of trading days
        step_size = 63     # Quarterly updates (approximately)
        
        results = []
        dates = []
        violations = []
        
        for i in range(window_size, len(self.portfolio_returns), step_size):
            train_returns = self.portfolio_returns.iloc[i - window_size:i]
            test_returns = self.portfolio_returns.iloc[i:min(i + step_size, len(self.portfolio_returns))]
            
            # Calculate VaR based on the specified method
            if method == 'normal':
                var = self.var_parametric_normal(window=window_size, **kwargs)
            elif method == 't':
                df = kwargs.get('df', 5)
                var = self.var_parametric_t(df=df, window=window_size, **kwargs)
            elif method == 'historical':
                var = self.var_historical(window=window_size, **kwargs)
            elif method == 'garch':
                var = self.var_garch_ccc(**kwargs)
            elif method == 'filtered':
                var = self.var_filtered_historical(**kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Track violations
            violation_count = sum(test_returns < -var)
            expected_violations = len(test_returns) * (1 - self.confidence_level)
            
            results.append({
                'date': test_returns.index[0],
                'var': var,
                'violations': violation_count,
                'expected_violations': expected_violations,
                'violation_ratio': violation_count / expected_violations if expected_violations > 0 else np.nan,
                'actual_returns': test_returns
            })
            
            dates.append(test_returns.index[0])
            violations.extend([1 if r < -var else 0 for r in test_returns])
        
        return pd.DataFrame(results), violations
        
    def backtest_es(self, method='normal', **kwargs):
        """Backtest ES using specified method"""
        # Define time windows for backtesting
        window_size = 252  # One year of trading days
        step_size = 63     # Quarterly updates (approximately)
        
        results = []
        
        for i in range(window_size, len(self.portfolio_returns), step_size):
            train_returns = self.portfolio_returns.iloc[i - window_size:i]
            test_returns = self.portfolio_returns.iloc[i:min(i + step_size, len(self.portfolio_returns))]
            
            # Calculate ES based on the specified method
            if method == 'normal':
                var = self.var_parametric_normal(window=window_size, **kwargs)
                es = self.es_parametric_normal(window=window_size, **kwargs)
            elif method == 't':
                df = kwargs.get('df', 5)
                var = self.var_parametric_t(df=df, window=window_size, **kwargs)
                es = self.es_parametric_t(df=df, window=window_size, **kwargs)
            elif method == 'historical':
                var = self.var_historical(window=window_size, **kwargs)
                es = self.es_historical(window=window_size, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Actual shortfalls
            shortfalls = -test_returns[test_returns < -var]
            avg_shortfall = shortfalls.mean() if len(shortfalls) > 0 else np.nan
            
            results.append({
                'date': test_returns.index[0],
                'var': var,
                'es': es,
                'actual_avg_shortfall': avg_shortfall,
                'es_ratio': avg_shortfall / es if not np.isnan(avg_shortfall) and es > 0 else np.nan
            })
        
        return pd.DataFrame(results)

    def analyze_var_violations(self, violations):
        """Analyze VaR violations for independence"""
        violations_array = np.array(violations)
        
        # Plot violations
        plt.figure(figsize=(12, 6))
        plt.plot(violations_array, marker='o', linestyle='')
        plt.title('VaR Violations')
        plt.ylabel('Violation (1=Yes)')
        plt.grid(True)
        plt.show()
        
        # Calculate autocorrelation to test for independence
        acf_values = acf(violations_array, nlags=10)
        
        # Plot autocorrelation
        plt.figure(figsize=(12, 6))
        plot_acf(violations_array, lags=10)
        plt.title('Autocorrelation of VaR Violations')
        plt.grid(True)
        plt.show()
        
        # Simple clustering analysis
        runs = 1
        for i in range(1, len(violations_array)):
            if violations_array[i] != violations_array[i-1]:
                runs += 1
        
        print(f"Total violations: {sum(violations_array)}")
        print(f"Expected number of runs for independent violations: {2 * np.mean(violations_array) * len(violations_array) * (1 - np.mean(violations_array)) + 1}")
        print(f"Actual number of runs: {runs}")
        
        return acf_values, runs

    def multi_day_var(self, method='normal', days=5, **kwargs):
        """Calculate multi-day VaR using specified method"""
        # For most methods, we can use the square-root-of-time rule
        if method == 'normal':
            one_day_var = self.var_parametric_normal(**kwargs)
        elif method == 't':
            df = kwargs.get('df', 5)
            one_day_var = self.var_parametric_t(df=df, **kwargs)
        elif method == 'historical':
            one_day_var = self.var_historical(**kwargs)
        elif method == 'garch':
            # For GARCH, use the model's forecasting capabilities
            return self.var_garch_ccc(forecast_horizon=days, **kwargs)
        elif method == 'filtered':
            one_day_var = self.var_filtered_historical(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Apply square-root-of-time rule (assuming no autocorrelation in returns)
        multi_day_var = one_day_var * np.sqrt(days)
        
        return multi_day_var

    def stress_test(self, scenarios=None):
        """Perform stress testing on the portfolio using historical or hypothetical scenarios"""
        if scenarios is None:
            # Default scenarios
            scenarios = {
                "2008 Financial Crisis": {
                    "stocks": -0.40,  # 40% drop in stocks
                    "rates": 0.02,    # 200 bps increase in rates
                    "fx": -0.20       # 20% depreciation against USD
                },
                "2020 COVID Crash": {
                    "stocks": -0.30,  # 30% drop in stocks
                    "rates": -0.01,   # 100 bps decrease in rates
                    "fx": -0.10       # 10% depreciation against USD
                },
                "Inflation Shock": {
                    "stocks": -0.15,  # 15% drop in stocks
                    "rates": 0.03,    # 300 bps increase in rates
                    "fx": -0.15       # 15% depreciation against USD
                }
            }
        
        results = {}
        
        for name, scenario in scenarios.items():
            # Initialize portfolio impact
            portfolio_impact = 0
            
            # Apply scenario to stocks
            for ticker, currency in self.stocks.items():
                if ticker in self.weights:
                    stock_impact = scenario["stocks"] * self.weights[ticker]
                    portfolio_impact += stock_impact
            
            # Apply scenario to interest rates
            for currency, rate_ticker in self.interest_rates.items():
                if currency in self.weights and f'{currency}_rate' in self.data.columns:
                    # Simple approximation of interest rate impact (modified duration approach)
                    # Assuming a duration of 5 years for the interest rate component
                    duration = 5
                    ir_impact = -duration * scenario["rates"] * self.weights[currency]
                    portfolio_impact += ir_impact
            
            # Apply scenario to exchange rates
            for fx_pair in self.exchange_rates:
                currency = fx_pair.split('=')[0].replace('USD', '')
                if fx_pair in self.weights and f'{currency}_USD_return' in self.returns.columns:
                    fx_impact = scenario["fx"] * self.weights[fx_pair]
                    portfolio_impact += fx_impact
            
            results[name] = portfolio_impact
        
        return results

    def generate_qq_plot(self, df_values=[3, 4, 5, 6]):
        """Generate QQ plots for normal and t-distributions with various degrees of freedom"""
        returns = self.portfolio_returns
        
        plt.figure(figsize=(15, 10))
        
        # Normal QQ plot
        plt.subplot(len(df_values) // 2 + 1, 2, 1)
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title("Normal QQ Plot")
        
        # T-distribution QQ plots for each df
        for i, df in enumerate(df_values, 1):
            plt.subplot(len(df_values) // 2 + 1, 2, i + 1)
            stats.probplot(returns, dist=lambda x: stats.t.ppf(x, df=df), plot=plt)
            plt.title(f"Student-t QQ Plot (df={df})")
        
        plt.tight_layout()
        plt.show()

    def run_full_analysis(self):
        """Run a complete analysis workflow"""
        # 1. Fetch and prepare data
        self.fetch_data()
        
        # 2. Set portfolio weights (default or custom)
        self.set_portfolio_weights()
        
        # 3. Plot portfolio returns
        self.plot_portfolio_returns()
        
        # 4. Generate QQ plot to determine best fitting distribution
        self.generate_qq_plot()
        
        # 5. Calculate VaR and ES using various methods
        methods = ['normal', 't', 'historical', 'garch', 'filtered']
        var_results = {}
        es_results = {}
        
        for method in methods:
            if method == 'normal':
                var_results[method] = self.var_parametric_normal()
                es_results[method] = self.es_parametric_normal()
            elif method == 't':
                # Try different degrees of freedom
                for df in [3, 4, 5, 6]:
                    var_results[f'{method}_{df}'] = self.var_parametric_t(df=df)
                    es_results[f'{method}_{df}'] = self.es_parametric_t(df=df)
            elif method == 'historical':
                var_results[method] = self.var_historical()
                es_results[method] = self.es_historical()
            elif method == 'garch':
                var_results[method] = self.var_garch_ccc()
                # No simple ES calculation for GARCH
            elif method == 'filtered':
                var_results[method] = self.var_filtered_historical()
                # No simple ES calculation for filtered HS
        
        print("1-Day VaR Results:")
        for method, value in var_results.items():
            print(f"{method}: {value:.6f}")
            
        print("\n1-Day ES Results:")
        for method, value in es_results.items():
            print(f"{method}: {value:.6f}")
        
        # 6. Backtest VaR and ES
        backtest_results = {}
        for method in ['normal', 't', 'historical']:
            kwargs = {'df': 5} if method == 't' else {}
            backtest_var, violations = self.backtest_var(method=method, **kwargs)
            backtest_es = self.backtest_es(method=method, **kwargs)
            
            backtest_results[method] = {
                'var_results': backtest_var,
                'violations': violations,
                'es_results': backtest_es
            }
        
        # 7. Analyze violations
        for method, results in backtest_results.items():
            print(f"\nAnalyzing violations for {method} method:")
            self.analyze_var_violations(results['violations'])
        
        # 8. Calculate multi-day VaR
        for days in [5, 10]:
            print(f"\n{days}-Day VaR Results:")
            for method in methods:
                if method == 't':
                    var = self.multi_day_var(method=method, days=days, df=5)
                    print(f"{method}_5: {var:.6f}")
                else:
                    var = self.multi_day_var(method=method, days=days)
                    print(f"{method}: {var:.6f}")
        
        # 9. Perform stress testing
        stress_results = self.stress_test()
        print("\nStress Test Results:")
        for scenario, impact in stress_results.items():
            print(f"{scenario}: {impact*100:.2f}%")
        
        return {
            'var_results': var_results,
            'es_results': es_results,
            'backtest_results': backtest_results,
            'stress_results': stress_results
        }


if __name__ == "__main__":
    risk_analysis = PortfolioRiskAnalysis()
    results = risk_analysis.run_full_analysis()