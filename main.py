import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import seaborn as sns
from arch import arch_model
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioRiskAnalyzer:
    def __init__(self):
        self.data = None
        self.returns = None
        self.portfolio_returns = None
        self.weights = None
        self.assets = None
        self.currencies = None
        self.risk_free_rates = None
        
    def fetch_data(self, assets, currencies, start_date, end_date):
        """
        Fetch historical data for assets and currencies
        
        Parameters:
        - assets: Dictionary with ticker symbols as keys and currency as values
        - currencies: List of currency pairs to fetch (relative to base currency)
        - start_date: Start date for data collection
        - end_date: End date for data collection
        """
        self.assets = assets
        self.currencies = currencies
        
        # Fetch stock/index data
        stock_data = {}
        for ticker, currency in assets.items():
            try:
                stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        # Fetch currency data
        currency_data = {}
        for curr in currencies:
            try:
                currency_data[curr] = yf.download(f"{curr}=X", start=start_date, end=end_date)['Adj Close']
            except Exception as e:
                print(f"Error fetching data for currency {curr}: {e}")
        
        # Fetch interest rate data (e.g., LIBOR or EURIBOR)
        # Note: As YFinance doesn't provide direct access to LIBOR/EURIBOR,
        # you may need to import this data separately from another source
        
        # Combine all data into one DataFrame
        all_data = pd.DataFrame()
        
        # Add stock data
        for ticker in stock_data:
            all_data[ticker] = stock_data[ticker]
            
        # Add currency data
        for curr in currency_data:
            all_data[f"{curr}_FX"] = currency_data[curr]
        
        self.data = all_data
        
        return self.data
    
    def clean_and_synchronize_data(self, fill_method='ffill', max_gap=5):
        """
        Synchronize and clean the data by handling missing values
        
        Parameters:
        - fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate')
        - max_gap: Maximum number of consecutive missing values to fill
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        # Check for missing values
        missing_before = self.data.isnull().sum()
        
        if missing_before.sum() > 0:
            print(f"Missing values before cleaning:\n{missing_before}")
            
            # Fill small gaps using the specified method
            if fill_method == 'ffill':
                self.data = self.data.fillna(method='ffill', limit=max_gap)
                self.data = self.data.fillna(method='bfill', limit=max_gap)
            elif fill_method == 'interpolate':
                self.data = self.data.interpolate(method='linear', limit=max_gap)
                self.data = self.data.fillna(method='ffill')  # Handle edges
                self.data = self.data.fillna(method='bfill')
                
            # Remove dates with still missing values
            self.data = self.data.dropna()
            
            missing_after = self.data.isnull().sum()
            print(f"Missing values after cleaning:\n{missing_after}")
        
        return self.data
    
    def calculate_returns(self, method='log'):
        """
        Calculate returns from price data
        
        Parameters:
        - method: 'log' for log returns, 'simple' for simple returns
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        if method == 'log':
            self.returns = np.log(self.data / self.data.shift(1)).dropna()
        else:
            self.returns = (self.data / self.data.shift(1) - 1).dropna()
            
        return self.returns
    
    def set_portfolio_weights(self, weights_dict):
        """
        Set portfolio weights
        
        Parameters:
        - weights_dict: Dictionary with asset tickers as keys and weights as values
        """
        self.weights = weights_dict
        weights_sum = sum(self.weights.values())
        
        if not np.isclose(weights_sum, 1.0):
            print(f"Warning: Weights sum to {weights_sum}, not 1.0. Normalizing weights.")
            for key in self.weights:
                self.weights[key] = self.weights[key] / weights_sum
                
        return self.weights
    
    def calculate_portfolio_returns(self):
        """
        Calculate portfolio returns based on individual asset returns and weights
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Call calculate_returns() first.")
            
        if self.weights is None:
            raise ValueError("Portfolio weights not set. Call set_portfolio_weights() first.")
        
        # Initialize portfolio returns with zeros
        self.portfolio_returns = pd.Series(0, index=self.returns.index)
        
        # Calculate weighted returns
        for asset, weight in self.weights.items():
            if asset in self.returns.columns:
                self.portfolio_returns += self.returns[asset] * weight
            else:
                print(f"Warning: Asset {asset} not found in returns data.")
                
        return self.portfolio_returns
    
    def calculate_var_normal(self, confidence_level=0.95, horizon=1):
        """
        Calculate Value at Risk using the Variance-Covariance method (Normal distribution)
        
        Parameters:
        - confidence_level: Confidence level for VaR (e.g., 0.95 for 95% confidence)
        - horizon: Time horizon in days
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Call calculate_portfolio_returns() first.")
        
        # Calculate mean and standard deviation of portfolio returns
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()
        
        # Calculate VaR
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(mu * horizon + z_score * sigma * np.sqrt(horizon))
        
        return var
    
    def calculate_var_t(self, confidence_level=0.95, horizon=1, df=5):
        """
        Calculate Value at Risk using the Variance-Covariance method (Student's t-distribution)
        
        Parameters:
        - confidence_level: Confidence level for VaR (e.g., 0.95 for 95% confidence)
        - horizon: Time horizon in days
        - df: Degrees of freedom for t-distribution
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Call calculate_portfolio_returns() first.")
        
        # Calculate mean and standard deviation of portfolio returns
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()
        
        # Calculate VaR using t-distribution
        t_score = stats.t.ppf(1 - confidence_level, df)
        var = -(mu * horizon + t_score * sigma * np.sqrt((df - 2) / df) * np.sqrt(horizon))
        
        return var
    
    def calculate_var_historical(self, confidence_level=0.95, horizon=1, overlapping=False):
        """
        Calculate Value at Risk using Historical Simulation
        
        Parameters:
        - confidence_level: Confidence level for VaR (e.g., 0.95 for 95% confidence)
        - horizon: Time horizon in days
        - overlapping: Whether to use overlapping periods for multi-day returns
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Call calculate_portfolio_returns() first.")
        
        if horizon == 1:
            # For 1-day horizon, use daily returns directly
            returns = self.portfolio_returns
        else:
            if overlapping:
                # For multi-day horizon with overlapping periods
                returns = pd.Series(
                    [self.portfolio_returns.iloc[i:i+horizon].sum() for i in range(len(self.portfolio_returns) - horizon + 1)],
                    index=self.portfolio_returns.index[horizon-1:]
                )
            else:
                # For multi-day horizon with non-overlapping periods
                returns = pd.Series(
                    [self.portfolio_returns.iloc[i:i+horizon].sum() for i in range(0, len(self.portfolio_returns) - horizon + 1, horizon)],
                    index=self.portfolio_returns.index[horizon-1::horizon]
                )
        
        # Calculate the VaR as the negative of the percentile
        var = -np.percentile(returns, 100 * (1 - confidence_level))
        
        return var
    
    def calculate_es_historical(self, confidence_level=0.95, horizon=1, overlapping=False):
        """
        Calculate Expected Shortfall using Historical Simulation
        
        Parameters:
        - confidence_level: Confidence level for ES (e.g., 0.95 for 95% confidence)
        - horizon: Time horizon in days
        - overlapping: Whether to use overlapping periods for multi-day returns
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Call calculate_portfolio_returns() first.")
        
        var = self.calculate_var_historical(confidence_level, horizon, overlapping)
        
        if horizon == 1:
            returns = self.portfolio_returns
        else:
            if overlapping:
                returns = pd.Series(
                    [self.portfolio_returns.iloc[i:i+horizon].sum() for i in range(len(self.portfolio_returns) - horizon + 1)],
                    index=self.portfolio_returns.index[horizon-1:]
                )
            else:
                returns = pd.Series(
                    [self.portfolio_returns.iloc[i:i+horizon].sum() for i in range(0, len(self.portfolio_returns) - horizon + 1, horizon)],
                    index=self.portfolio_returns.index[horizon-1::horizon]
                )
                
        # Calculate ES as the mean of returns below VaR
        es = -returns[returns < -var].mean()
        
        return es
    
    def calculate_var_garch_ccc(self, confidence_level=0.95, horizon=1, forecast_days=1):
        """
        Calculate Value at Risk using GARCH(1,1) with Constant Conditional Correlation
        
        Parameters:
        - confidence_level: Confidence level for VaR (e.g., 0.95 for 95% confidence)
        - horizon: Time horizon in days
        - forecast_days: Number of days to forecast
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Call calculate_portfolio_returns() first.")
        
        # Fit GARCH(1,1) model to portfolio returns
        garch_model = arch_model(self.portfolio_returns.dropna(), mean='Zero', vol='GARCH', p=1, q=1)
        garch_result = garch_model.fit(disp='off')
        
        # Forecast volatility
        forecast = garch_result.forecast(horizon=forecast_days)
        forecasted_var = forecast.variance.iloc[-1].values[0]
        
        # Calculate VaR using forecasted volatility
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -z_score * np.sqrt(forecasted_var) * np.sqrt(horizon)
        
        return var
    
    def calculate_var_filtered_historical(self, confidence_level=0.95, horizon=1, lambda_ewma=0.94):
        """
        Calculate Value at Risk using Filtered Historical Simulation with EWMA
        
        Parameters:
        - confidence_level: Confidence level for VaR (e.g., 0.95 for 95% confidence)
        - horizon: Time horizon in days
        - lambda_ewma: Decay factor for EWMA
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Call calculate_portfolio_returns() first.")
        
        # Calculate EWMA volatility
        returns = self.portfolio_returns
        vol = [returns.iloc[0]**2]  # Initialize with first squared return
        
        for i in range(1, len(returns)):
            vol.append(lambda_ewma * vol[i-1] + (1 - lambda_ewma) * returns.iloc[i-1]**2)
        
        vol = np.sqrt(pd.Series(vol, index=returns.index))
        
        # Calculate standardized returns
        std_returns = returns / vol
        
        # Current volatility (most recent)
        current_vol = vol.iloc[-1]
        
        # Calculate VaR based on quantile of standardized returns and current volatility
        var = -np.percentile(std_returns, 100 * (1 - confidence_level)) * current_vol * np.sqrt(horizon)
        
        return var
    
    def backtest_var(self, var_method, confidence_level=0.95, window_size=252, plot=True):
        """
        Backtest VaR predictions against actual returns
        
        Parameters:
        - var_method: Method to calculate VaR ('normal', 't', 'historical', 'garch_ccc', 'filtered_historical')
        - confidence_level: Confidence level for VaR
        - window_size: Rolling window size for VaR calculation (in days)
        - plot: Whether to plot the results
        
        Returns:
        - Dictionary with backtest results
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Call calculate_portfolio_returns() first.")
            
        returns = self.portfolio_returns
        var_predictions = []
        violations = []
        
        # Select VaR calculation method
        if var_method == 'normal':
            var_func = self.calculate_var_normal
        elif var_method == 't':
            var_func = self.calculate_var_t
        elif var_method == 'historical':
            var_func = self.calculate_var_historical
        elif var_method == 'garch_ccc':
            var_func = self.calculate_var_garch_ccc
        elif var_method == 'filtered_historical':
            var_func = self.calculate_var_filtered_historical
        else:
            raise ValueError(f"Unknown VaR method: {var_method}")
            
        # Rolling window VaR calculation and violation check
        for i in range(window_size, len(returns)):
            train_returns = returns.iloc[i-window_size:i]
            
            # Create a temporary object for the window
            temp_analyzer = PortfolioRiskAnalyzer()
            temp_analyzer.portfolio_returns = train_returns
            
            # Calculate VaR for next day
            var = var_func(confidence_level=confidence_level)
            var_predictions.append(var)
            
            # Check for violation
            actual_return = returns.iloc[i]
            violation = 1 if actual_return < -var else 0
            violations.append(violation)
            
        # Convert to Series
        var_predictions = pd.Series(var_predictions, index=returns.index[window_size:])
        violations = pd.Series(violations, index=returns.index[window_size:])
        
        # Calculate violation ratio
        expected_violations = 1 - confidence_level
        observed_violations = violations.mean()
        
        # Group by year for annual violation statistics
        violations_by_year = violations.groupby(violations.index.year).agg(['count', 'sum'])
        violations_by_year['ratio'] = violations_by_year['sum'] / violations_by_year['count']
        violations_by_year['expected'] = expected_violations * violations_by_year['count']
        
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Plot portfolio returns and negative VaR
            plt.subplot(2, 1, 1)
            plt.plot(returns.loc[var_predictions.index], label='Portfolio Returns', alpha=0.7)
            plt.plot(-var_predictions, 'r--', label=f'{confidence_level*100}% VaR ({var_method})', linewidth=1.5)
            plt.title(f'Portfolio Returns vs VaR ({var_method} method)')
            plt.legend()
            plt.grid(True)
            
            # Plot violations
            plt.subplot(2, 1, 2)
            plt.bar(violations.index, violations, width=2, label='VaR Violations', color='red', alpha=0.7)
            plt.title('VaR Violations')
            plt.tight_layout()
            plt.show()
            
            # Plot annual violation statistics
            plt.figure(figsize=(10, 6))
            plt.bar(violations_by_year.index, violations_by_year['ratio'], alpha=0.7, label='Observed Violation Ratio')
            plt.axhline(y=expected_violations, color='r', linestyle='--', label=f'Expected ({expected_violations:.2%})')
            plt.title(f'Annual VaR Violation Ratios ({var_method} method)')
            plt.xlabel('Year')
            plt.ylabel('Violation Ratio')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        results = {
            'var_predictions': var_predictions,
            'violations': violations,
            'expected_violations': expected_violations,
            'observed_violations': observed_violations,
            'violations_by_year': violations_by_year
        }
        
        return results
    
    def backtest_es(self, es_method, confidence_level=0.95, window_size=252, plot=True):
        """
        Backtest Expected Shortfall predictions against actual returns
        
        Parameters:
        - es_method: Method to calculate ES ('historical')
        - confidence_level: Confidence level for ES
        - window_size: Rolling window size for ES calculation (in days)
        - plot: Whether to plot the results
        
        Returns:
        - Dictionary with backtest results
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Call calculate_portfolio_returns() first.")
            
        returns = self.portfolio_returns
        es_predictions = []
        actual_shortfalls = []
        var_predictions = []
        violations = []
        
        # Rolling window ES calculation
        for i in range(window_size, len(returns)):
            train_returns = returns.iloc[i-window_size:i]
            
            # Create a temporary object for the window
            temp_analyzer = PortfolioRiskAnalyzer()
            temp_analyzer.portfolio_returns = train_returns
            
            # Calculate VaR and ES for next day
            if es_method == 'historical':
                var = temp_analyzer.calculate_var_historical(confidence_level)
                es = temp_analyzer.calculate_es_historical(confidence_level)
            else:
                raise ValueError(f"Unknown ES method: {es_method}")
                
            var_predictions.append(var)
            es_predictions.append(es)
            
            # Check for violation
            actual_return = returns.iloc[i]
            violation = 1 if actual_return < -var else 0
            violations.append(violation)
            
            # Calculate actual shortfall when violation occurs
            if violation:
                actual_shortfalls.append(-actual_return)
        
        # Convert to Series
        var_predictions = pd.Series(var_predictions, index=returns.index[window_size:])
        es_predictions = pd.Series(es_predictions, index=returns.index[window_size:])
        violations = pd.Series(violations, index=returns.index[window_size:])
        
        # Calculate average ES vs. average actual shortfall
        average_es = es_predictions.mean()
        if len(actual_shortfalls) > 0:
            average_actual_shortfall = np.mean(actual_shortfalls)
        else:
            average_actual_shortfall = np.nan
        
        # Group by year for annual ES statistics
        es_by_year = pd.DataFrame({
            'ES': es_predictions
        })
        es_by_year['Year'] = es_by_year.index.year
        es_by_year = es_by_year.groupby('Year')['ES'].mean()
        
        # Group violations by year to calculate actual shortfall
        violations_series = pd.Series(violations, index=returns.index[window_size:])
        violation_indices = violations_series[violations_series == 1].index
        
        actual_shortfall_by_year = {}
        if len(violation_indices) > 0:
            shortfall_df = pd.DataFrame({
                'Shortfall': -returns.loc[violation_indices],
                'Year': violation_indices.year
            })
            actual_shortfall_by_year = shortfall_df.groupby('Year')['Shortfall'].mean()
        
        if plot and len(actual_shortfalls) > 0:
            plt.figure(figsize=(12, 8))
            
            # Plot ES predictions and actual shortfalls
            plt.subplot(2, 1, 1)
            plt.plot(es_predictions, 'b-', label=f'{confidence_level*100}% ES ({es_method})', linewidth=1.5)
            
            # Plot shortfall points at violation dates
            shortfall_df = pd.DataFrame({
                'Date': violation_indices,
                'Shortfall': actual_shortfalls
            }).set_index('Date')
            
            plt.scatter(shortfall_df.index, shortfall_df['Shortfall'], color='r', marker='x', label='Actual Shortfall')
            
            plt.title(f'Expected Shortfall vs Actual Shortfalls ({es_method} method)')
            plt.legend()
            plt.grid(True)
            
            # Plot annual statistics
            plt.subplot(2, 1, 2)
            years = sorted(set(es_by_year.index) | set(actual_shortfall_by_year.index if actual_shortfall_by_year else []))
            
            width = 0.35
            x = np.arange(len(years))
            
            es_values = [es_by_year.get(year, np.nan) for year in years]
            as_values = [actual_shortfall_by_year.get(year, np.nan) for year in years]
            
            plt.bar(x - width/2, es_values, width, label='Average ES')
            plt.bar(x + width/2, as_values, width, label='Average Actual Shortfall')
            
            plt.xlabel('Year')
            plt.ylabel('Value')
            plt.title('Annual ES vs Actual Shortfall')
            plt.xticks(x, years)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        results = {
            'es_predictions': es_predictions,
            'var_predictions': var_predictions,
            'violations': violations,
            'actual_shortfalls': actual_shortfalls if len(actual_shortfalls) > 0 else None,
            'average_es': average_es,
            'average_actual_shortfall': average_actual_shortfall,
            'es_by_year': es_by_year,
            'actual_shortfall_by_year': actual_shortfall_by_year if len(actual_shortfalls) > 0 else None
        }
        
        return results
    
    def plot_qq(self):
        """
        Create QQ-plot of portfolio returns against normal and t-distributions
        to determine the best-fitting degrees of freedom
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Call calculate_portfolio_returns() first.")
        
        returns = self.portfolio_returns
        
        plt.figure(figsize=(15, 10))
        
        # QQ-plot against normal distribution
        plt.subplot(2, 3, 1)
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title("QQ Plot (Normal Distribution)")
        
        # QQ-plots against t-distribution with different degrees of freedom
        dfs = [3, 4, 5, 6]
        for i, df in enumerate(dfs):
            plt.subplot(2, 3, i+2)
            
            # Standardize returns
            std_returns = (returns - returns.mean()) / returns.std()
            
            # Theoretical quantiles from t-distribution
            n = len(std_returns)
            p = np.arange(1, n+1) / (n+1)
            theoretical_quantiles = stats.t.ppf(p, df)
            
            # Empirical quantiles (sorted returns)
            empirical_quantiles = sorted(std_returns)
            
            # Plot
            plt.scatter(theoretical_quantiles, empirical_quantiles)
            plt.plot(theoretical_quantiles, theoretical_quantiles, 'r-')
            plt.title(f"QQ Plot (t-distribution, df={df})")
            plt.xlabel("Theoretical Quantiles")
            plt.ylabel("Empirical Quantiles")
        
        plt.tight_layout()
        plt.show()
        
    def stress_test(self, scenarios, confidence_level=0.95):
        """
        Perform stress testing on the portfolio
        
        Parameters:
        - scenarios: Dictionary of scenarios with asset/factor names as keys and percentage changes as values
        - confidence_level: Confidence level for VaR calculation
        
        Returns:
        - Dictionary with stress test results
        """
        if self.data is None or self.weights is None:
            raise ValueError("Data and weights must be set before stress testing.")
        
        # Get the most recent data point
        latest_data = self.data.iloc[-1].copy()
        
        results = {}
        
        for scenario_name, changes in scenarios.items():
            # Apply changes to the latest data
            stressed_data = latest_data.copy()
            
            for asset, change_pct in changes.items():
                if asset in stressed_data:
                    stressed_data[asset] *= (1 + change_pct)
            
            # Calculate portfolio value before and after stress
            original_value = sum(latest_data[asset] * self.weights.get(asset, 0) for asset in self.weights if asset in latest_data)
            stressed_value = sum(stressed_data[asset] * self.weights.get(asset, 0) for asset in self.weights if asset in stressed_data)
            
            # Calculate impact
            absolute_impact = stressed_value - original_value
            percentage_impact = absolute_impact / original_value * 100
            
            # Store results
            results[scenario_name] = {
                'original_value': original_value,
                'stressed_value': stressed_value,
                'absolute_impact': absolute_impact,
                'percentage_impact': percentage_impact
            }
            
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    risk_analyzer = PortfolioRiskAnalyzer()
    
    # Define assets and currencies
    assets = {
        "AAPL": "USD",  # Apple in USD
        "MSFT": "USD",  # Microsoft in USD
        "ASML.AS": "EUR",  # ASML in EUR
        "BMW.DE": "EUR",  # BMW in EUR
        "7203.T": "JPY"   # Toyota in JPY
    }
    
    currencies = ["EURUSD", "JPYUSD"]  # Exchange rates to USD (base currency)
    
    # Fetch data for the past 8 years
    start_date = (datetime.now() - timedelta(days=365*8)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = risk_analyzer.fetch_data(assets, currencies, start_date, end_date)
    print(f"Fetched data shape: {data.shape}")
    
    # Clean and synchronize data
    data = risk_analyzer.clean_and_synchronize_data()
    print(f"Cleaned data shape: {data.shape}")
    
    # Calculate returns
    returns = risk_analyzer.calculate_returns()
    print(f"Returns data shape: {returns.shape}")
    
    # Set portfolio weights
    weights = {
        "AAPL": 0.20,
        "MSFT": 0.20,
        "ASML.AS": 0.20,
        "BMW.DE": 0.20,
        "7203.T": 0.20,
    }
    risk_analyzer.set_portfolio_weights(weights)
    
    # Calculate portfolio returns
    portfolio_returns = risk_analyzer.calculate_portfolio_returns()
    
    # Plot portfolio returns distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(portfolio_returns, kde=True)
    plt.title("Portfolio Returns Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    
    # Create QQ-plot to find best-fitting degrees of freedom
    risk_analyzer.plot_qq()
    
    # Calculate VaR using different methods
    confidence_level = 0.99  # 99% confidence level
    
    var_normal = risk_analyzer.calculate_var_normal(confidence_level)
    var_t = risk_analyzer.calculate_var_t(confidence_level, df=5)  # Adjust df based on QQ-plot
    var_historical = risk_analyzer.calculate_var_historical(confidence_level)
    var_garch = risk_analyzer.calculate_var_garch_ccc(confidence_level)
    var_filtered = risk_analyzer.calculate_var_filtered_historical(confidence_level)
    
    print(f"\nVaR estimates at {confidence_level*100}% confidence level:")
    print(f"Normal VaR: {var_normal:.6f}")
    print(f"Student-t VaR (df=5): {var_t:.6f}")
    print(f"Historical VaR: {var_historical:.6f}")
    print(f"GARCH(1,1) VaR: {var_garch:.6f}")
    print(f"Filtered Historical VaR: {var_filtered:.6f}")
    
    # Calculate ES using historical method
    es_historical = risk_analyzer.calculate_es_historical(confidence_level)
    print(f"\nES estimate at {confidence_level*100}% confidence level:")
    print(f"Historical ES: {es_historical:.6f}")
    
    # Backtest VaR predictions
    print("\nBacktesting VaR...")
    var_backtest_normal = risk_analyzer.backtest_var('normal', confidence_level=0.99, window_size=252)
    var_backtest_t = risk_analyzer.backtest_var('t', confidence_level=0.99, window_size=252)
    var_backtest_historical = risk_analyzer.backtest_var('historical', confidence_level=0.99, window_size=252)
    
    # Multi-day VaR
    print("\nMulti-day VaR analysis:")
    var_1day = risk_analyzer.calculate_var_historical(confidence_level, horizon=1)
    var_5day = risk_analyzer.calculate_var_historical(confidence_level, horizon=5, overlapping=False)
    var_10day = risk_analyzer.calculate_var_historical(confidence_level, horizon=10, overlapping=False)
    
    # Square-root-of-time rule
    var_5day_sqrt = var_1day * np.sqrt(5)
    var_10day_sqrt = var_1day * np.sqrt(10)
    
    print(f"1-day Historical VaR: {var_1day:.6f}")
    print(f"5-day Historical VaR: {var_5day:.6f}")
    print(f"5-day VaR (sqrt rule): {var_5day_sqrt:.6f}")
    print(f"10-day Historical VaR: {var_10day:.6f}")
    print(f"10-day VaR (sqrt rule): {var_10day_sqrt:.6f}")
    
    # Compare
    print(f"5-day ratio (actual/sqrt): {var_5day/var_5day_sqrt:.4f}")
    print(f"10-day ratio (actual/sqrt): {var_10day/var_10day_sqrt:.4f}")
    
    # Stress testing
    print("\nStress testing:")
    
    scenarios = {
        "Equity +20%": {"AAPL": 0.20, "MSFT": 0.20, "ASML.AS": 0.20, "BMW.DE": 0.20, "7203.T": 0.20},
        "Equity -20%": {"AAPL": -0.20, "MSFT": -0.20, "ASML.AS": -0.20, "BMW.DE": -0.20, "7203.T": -0.20},
        "Equity -40%": {"AAPL": -0.40, "MSFT": -0.40, "ASML.AS": -0.40, "BMW.DE": -0.40, "7203.T": -0.40},
        "EUR +10%": {"EURUSD_FX": 0.10},
        "EUR -10%": {"EURUSD_FX": -0.10},
        "JPY +20%": {"JPYUSD_FX": 0.20},
        "JPY -20%": {"JPYUSD_FX": -0.20}
    }
    
    stress_results = risk_analyzer.stress_test(scenarios)
    
    for scenario, result in stress_results.items():
        print(f"{scenario}:")
        print(f"  Original value: {result['original_value']:.2f}")
        print(f"  Stressed value: {result['stressed_value']:.2f}")
        print(f"  Impact: {result['absolute_impact']:.2f} ({result['percentage_impact']:.2f}%)")