"""
Deep Reinforcement Learning for Portfolio Optimization

This implementation creates a framework for optimizing stock portfolio allocation
using Deep Reinforcement Learning (DRL) techniques. The framework processes
financial data including historical stock prices and technical indicators to train
an intelligent agent capable of making data-driven investment decisions while
balancing risk and return.

Based on the research paper: "Optimizing Stock Portfolio Allocation with Deep Reinforcement Learning"
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import yfinance as yf
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import ta  # Technical analysis library
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
np.random.seed(42)

class PortfolioOptimizationEnv(gym.Env):
    """
    A custom Gym environment for portfolio optimization using deep reinforcement learning.
    
    This environment simulates the dynamic of stock portfolio management, including:
    - Market observations (prices, technical indicators)
    - Portfolio rebalancing actions
    - Reward calculation based on risk-adjusted returns
    """
    
    def __init__(self, df, stocks, initial_amount=10000, window_size=30, transaction_cost_pct=0.001):
        """
        Initialize the portfolio optimization environment.
        
        Args:
            df (pd.DataFrame): DataFrame containing historical stock prices and indicators
            stocks (list): List of stock symbols in the portfolio
            initial_amount (float): Initial capital for investment
            window_size (int): Number of previous days to consider for state representation
            transaction_cost_pct (float): Transaction cost percentage
        """
        super(PortfolioOptimizationEnv, self).__init__()
        
        self.df = df
        self.stocks = stocks
        self.num_stocks = len(stocks)
        self.initial_amount = initial_amount
        self.current_amount = initial_amount
        self.window_size = window_size
        self.transaction_cost_pct = transaction_cost_pct
        
        # Current position in the dataset
        self.current_step = window_size
        
        # Action space: Portfolio weights for each stock
        # Each action represents a weight allocation for a stock (between 0 and 1)
        # Weights will be normalized to sum to 1
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_stocks,), dtype=np.float32
        )
        
        # State space: Historical price data + technical indicators + current portfolio allocation
        # For each stock we have price data and technical indicators for window_size days
        # Plus current portfolio weights
        num_features_per_stock = len([col for col in df.columns if stocks[0] in col])
        state_dim = window_size * num_features_per_stock * self.num_stocks + self.num_stocks
    
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Initialize portfolio
        self.portfolio_value = initial_amount
        self.portfolio_weights = np.zeros(self.num_stocks)
        self.portfolio_history = [initial_amount]
        self.benchmark_history = [initial_amount]
        
        # Initialize the dates
        self.dates = df.index.unique()
        self.current_date = None
        
    def reset(self, seed=None):
        """
        Reset the environment to an initial state.
        
        Returns:
            observation (np.array): Initial state observation
        """
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.current_amount = self.initial_amount
        self.portfolio_value = self.initial_amount
        self.portfolio_weights = np.zeros(self.num_stocks)
        self.portfolio_history = [self.initial_amount]
        self.benchmark_history = [self.initial_amount]
        
        if self.current_step < len(self.df.index.unique()):
            self.current_date = self.dates[self.current_step]
        
        # Return initial observation
        observation = self._get_observation()
        print(f"Observation shape: {observation.shape}, Expected: {self.observation_space.shape}")
        return observation, {}
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (np.array): Portfolio weights for each stock
            
        Returns:
            observation (np.array): Next state observation
            reward (float): Reward for the action
            done (bool): Whether the episode is done
            truncated (bool): Whether the episode was truncated
            info (dict): Additional information
        """
        # Normalize action to ensure weights sum to 1
        action = np.clip(action, 0, 1)
        sum_action = np.sum(action)
        if sum_action > 0:
            action = action / sum_action
        
        # Calculate transaction costs
        cost = self._calculate_transaction_cost(action)
        
        # Save previous portfolio value for reward calculation
        prev_portfolio_value = self.portfolio_value
        
        # Update current step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.dates) - 1
        
        # If not done, update the portfolio value based on the price changes
        if not done:
            self.current_date = self.dates[self.current_step]
            
            # Calculate new portfolio value based on price changes and weights
            self.portfolio_value = self._update_portfolio_value(action)
            self.portfolio_value -= cost
            
            # Update portfolio history
            self.portfolio_history.append(self.portfolio_value)
            
            # Update benchmark history (equal weight allocation)
            benchmark_action = np.ones(self.num_stocks) / self.num_stocks
            benchmark_value = self._update_portfolio_value(benchmark_action)
            self.benchmark_history.append(benchmark_value)
            
            # Calculate reward
            reward = self._calculate_reward(prev_portfolio_value, cost)
            
            # Get new observation
            observation = self._get_observation()
            
            # Update portfolio weights
            self.portfolio_weights = action
            
            truncated = False
            info = {
                'portfolio_value': self.portfolio_value,
                'transaction_cost': cost,
                'portfolio_weights': action
            }
            
            return observation, reward, done, truncated, info
        
        truncated = False
        info = {
            'portfolio_value': self.portfolio_value,
            'final_return': (self.portfolio_value / self.initial_amount) - 1,
            'portfolio_history': self.portfolio_history,
            'benchmark_history': self.benchmark_history
        }
        
        # If done, return the last observation
        observation = self._get_observation()
        reward = self._calculate_reward(prev_portfolio_value, cost)
        
        return observation, reward, done, truncated, info
    
    def _get_observation(self):
        """
        Create state representation for the current time step.
    
        Returns:
            observation (np.array): State observation with fixed shape
        """
        # Extract window of historical data
        start_idx = self.df.index.get_indexer([self.dates[self.current_step - self.window_size]])[0]
        end_idx = self.df.index.get_indexer([self.dates[self.current_step]])[0]
    
        window_data = self.df.iloc[start_idx:end_idx+1]
    
        # Reshape data for observation
        obs_list = []
    
        # Add historical data for each stock
        for stock in self.stocks:
            stock_data = window_data[[col for col in window_data.columns if stock in col]]
            obs_list.append(stock_data.values.flatten())
    
        # Add current portfolio weights
        obs_list.append(self.portfolio_weights)
    
        # Combine all features into a single vector
        observation = np.concatenate(obs_list)
    
        # IMPORTANT: Force the observation to have the right shape
        # If it's too large, truncate it
        if observation.shape[0] > self.observation_space.shape[0]:
            # print(f"Truncating observation from {observation.shape[0]} to {self.observation_space.shape[0]}")
            observation = observation[:self.observation_space.shape[0]]
    
        # If it's too small, pad it with zeros
        elif observation.shape[0] < self.observation_space.shape[0]:
            print(f"Padding observation from {observation.shape[0]} to {self.observation_space.shape[0]}")
            padding = np.zeros(self.observation_space.shape[0] - observation.shape[0])
            observation = np.concatenate([observation, padding])
    
        return observation.astype(np.float32)
 
    def _calculate_transaction_cost(self, new_weights):
        """
        Calculate transaction costs for portfolio rebalancing.
        
        Args:
            new_weights (np.array): New portfolio weights
            
        Returns:
            cost (float): Transaction cost
        """
        # Calculate the absolute difference in weights
        weight_diff = np.abs(new_weights - self.portfolio_weights)
        
        # Calculate transaction cost
        cost = np.sum(weight_diff) * self.transaction_cost_pct * self.portfolio_value
        
        return cost
    
    def _update_portfolio_value(self, weights):
        """
        Update portfolio value based on price changes and weights.
        
        Args:
            weights (np.array): Portfolio weights
            
        Returns:
            new_value (float): Updated portfolio value
        """
        # Get price data for the current and previous day
        current_day_idx = self.df.index.get_indexer([self.dates[self.current_step]])[0]
        prev_day_idx = self.df.index.get_indexer([self.dates[self.current_step - 1]])[0]
        
        # Extract close prices
        current_prices = []
        prev_prices = []
        
        for stock in self.stocks:
            # Find the column with closing price for this stock
            close_col = [col for col in self.df.columns if stock in col and 'close' in col.lower()][0]
            current_prices.append(self.df.iloc[current_day_idx][close_col])
            prev_prices.append(self.df.iloc[prev_day_idx][close_col])
        
        # Convert to numpy arrays
        current_prices = np.array(current_prices)
        prev_prices = np.array(prev_prices)
        
        # Calculate price changes
        price_change_ratio = current_prices / prev_prices
        
        # Calculate portfolio return
        weighted_return = np.sum(weights * price_change_ratio)
        
        # Calculate new portfolio value
        new_value = self.portfolio_value * weighted_return
        
        return new_value
    

    def _calculate_reward(self, prev_value, cost):
        """
        Calculate reward based on portfolio return and risk.

        Args:
            prev_value (float): Previous portfolio value
            cost (float): Transaction cost

        Returns:
            reward (float): Calculated reward
        """
        # ? Avoid divide-by-zero
        if prev_value <= 1e-8:
            prev_value = 1e-8
        if self.portfolio_value <= 1e-8:
            self.portfolio_value = 1e-8

        # Calculate return
        portfolio_return = (self.portfolio_value - prev_value) / prev_value

        # Penalize transaction costs
        reward = portfolio_return - (cost / self.portfolio_value)

        # Add Sharpe ratio component if we have enough history
        if len(self.portfolio_history) > 20:
            portfolio_values = np.array(self.portfolio_history[-21:])
        
            # ? Sanity check: all values must be > 0
            if np.any(portfolio_values <= 1e-8):
                portfolio_values = np.clip(portfolio_values, 1e-8, None)
        
            returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
        
            # ? Check for clean std and mean values
            if not np.isnan(returns).any() and np.std(returns) > 0:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
                reward += 0.1 * sharpe

        # ? Final check
        if np.isnan(reward):
            reward = 0.0

        return reward

    def render(self):
        """
        Render the environment state (for visualization).
        """
        pass


class PortfolioOptimizer:
    """
    Main class for handling the portfolio optimization process using DRL.
    """
    
    def __init__(self, stock_symbols, start_date, end_date, initial_capital=10000, 
                 window_size=30, test_ratio=0.2, algorithm='ppo'):
        """
        Initialize the portfolio optimizer.
        
        Args:
            stock_symbols (list): List of stock symbols to include in the portfolio
            start_date (str): Start date for data collection (YYYY-MM-DD)
            end_date (str): End date for data collection (YYYY-MM-DD)
            initial_capital (float): Initial investment capital
            window_size (int): Window size for historical data in state representation
            test_ratio (float): Ratio of data to use for testing
            algorithm (str): DRL algorithm to use ('ppo', 'a2c', 'dqn', 'sac')
        """
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.window_size = window_size
        self.test_ratio = test_ratio
        self.algorithm = algorithm.lower()
        
        # Validate algorithm
        valid_algorithms = ['ppo', 'a2c', 'dqn', 'sac']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")
        
        # Create directories for models and results
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def fetch_data(self):
        """
        Fetch historical stock data and preprocess for DRL.
        
        Returns:
            df (pd.DataFrame): Processed DataFrame with stock data and indicators
        """
        print(f"Fetching data for {len(self.stock_symbols)} stocks from {self.start_date} to {self.end_date}...")
        
        # Download data using yfinance
        data = yf.download(self.stock_symbols, start=self.start_date, end=self.end_date)
        
        # Rename columns to include stock symbol
        columns = []
        for stock in self.stock_symbols:
            for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
                columns.append(f"{stock}_{feature}")
        
        # Reorganize the data
        df = pd.DataFrame(index=data.index)
        
        for stock in self.stock_symbols:
            df[f"{stock}_Open"] = data['Open'][stock]
            df[f"{stock}_High"] = data['High'][stock]
            df[f"{stock}_Low"] = data['Low'][stock]
            df[f"{stock}_Close"] = data['Close'][stock]
            df[f"{stock}_Volume"] = data['Volume'][stock]
            
            # Add technical indicators for each stock
            close_series = df[f"{stock}_Close"]
            high_series = df[f"{stock}_High"]
            low_series = df[f"{stock}_Low"]
            volume_series = df[f"{stock}_Volume"]
            
            # Price-based indicators
            df[f"{stock}_SMA_15"] = ta.trend.sma_indicator(close_series, window=15)
            df[f"{stock}_SMA_50"] = ta.trend.sma_indicator(close_series, window=50)
            df[f"{stock}_EMA_15"] = ta.trend.ema_indicator(close_series, window=15)
            
            # Volatility indicators
            df[f"{stock}_BollingerB_20"] = ta.volatility.bollinger_hband_indicator(close_series, window=20)
            df[f"{stock}_ATR_14"] = ta.volatility.average_true_range(high_series, low_series, close_series, window=14)
            
            # Momentum indicators
            df[f"{stock}_RSI_14"] = ta.momentum.rsi(close_series, window=14)
            df[f"{stock}_MACD"] = ta.trend.macd_diff(close_series, window_slow=26, window_fast=12, window_sign=9)
            
            # Volume indicators
            df[f"{stock}_OBV"] = ta.volume.on_balance_volume(close_series, volume_series)
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        print(f"Processed data shape: {df.shape}")
        return df
    
    def split_data(self, df):
        """
        Split data into training and testing sets.
        
        Args:
            df (pd.DataFrame): Full dataset
            
        Returns:
            train_df (pd.DataFrame): Training dataset
            test_df (pd.DataFrame): Testing dataset
        """
        # Calculate split point
        split_idx = int(len(df) * (1 - self.test_ratio))
        
        # Split data
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"Training data: {train_df.shape}, Testing data: {test_df.shape}")
        return train_df, test_df
    
    def create_env(self, df):
        """
        Create a portfolio optimization environment.
        
        Args:
            df (pd.DataFrame): Dataset to use in the environment
            
        Returns:
            env (gym.Env): Gym environment for portfolio optimization
        """
        # Create environment
        env = PortfolioOptimizationEnv(
            df=df,
            stocks=self.stock_symbols,
            initial_amount=self.initial_capital,
            window_size=self.window_size
        )
        
        # Wrap environment for compatibility with stable-baselines3
        env = DummyVecEnv([lambda: env])
        
        return env
    
    def train_model(self, env, total_timesteps=100000):
        """
        Train a DRL model for portfolio optimization.
        
        Args:
            env (gym.Env): Training environment
            total_timesteps (int): Total number of training steps
            
        Returns:
            model: Trained DRL model
        """
        print(f"Training {self.algorithm.upper()} model for {total_timesteps} timesteps...")
        
        # Set up model based on selected algorithm
        if self.algorithm == 'ppo':
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log="./tensorboard/",
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
        elif self.algorithm == 'a2c':
            model = A2C(
                "MlpPolicy", 
                env, 
                verbose=1,
                tensorboard_log="./tensorboard/",
                learning_rate=0.0007,
                n_steps=5,
                gamma=0.99,
                ent_coef=0.01
            )
        elif self.algorithm == 'dqn':
            model = DQN(
                "MlpPolicy", 
                env, 
                verbose=1,
                tensorboard_log="./tensorboard/",
                learning_rate=0.0001,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=64,
                gamma=0.99,
                target_update_interval=1000
            )
        elif self.algorithm == 'sac':
            model = SAC(
                "MlpPolicy", 
                env, 
                verbose=1,
                tensorboard_log="./tensorboard/",
                learning_rate=0.0003,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=64,
                gamma=0.99,
                ent_coef='auto'
            )
        
        # Set up evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path='./models/',
            log_path='./logs/',
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
        # Save the final model
        model.save(f"models/{self.algorithm}_portfolio_model")
        
        return model
    
    def evaluate_model(self, model, env, n_episodes=1):
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained DRL model
            env (gym.Env): Test environment
            n_episodes (int): Number of episodes to evaluate
            
        Returns:
            results (dict): Evaluation results
        """
        print("Evaluating model performance...")
        
        # Run evaluation
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
        
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Get detailed performance metrics
        # obs, _ = env.reset()
        obs = env.reset()
        
        done = False
        portfolio_values = []

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
    
            info = info[0]  # Unwrap from DummyVecEnv list
            portfolio_values.append(info['portfolio_value'])

            if done:
                final_portfolio_value = info['portfolio_value']
                final_return = info['final_return']
                portfolio_history = info['portfolio_history']
                benchmark_history = info['benchmark_history']
        
        # Calculate more performance metrics
        portfolio_returns = np.diff(portfolio_history) / portfolio_history[:-1]
        benchmark_returns = np.diff(benchmark_history) / benchmark_history[:-1]
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-6) * np.sqrt(252)
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(portfolio_history)
        drawdowns = (peak - portfolio_history) / peak
        max_drawdown = np.max(drawdowns)
        
        # Benchmark comparison
        benchmark_final_return = (benchmark_history[-1] / benchmark_history[0]) - 1
        outperformance = final_return - benchmark_final_return
        
        # Store results
        results = {
            'final_portfolio_value': final_portfolio_value,
            'total_return': final_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_return': benchmark_final_return,
            'outperformance': outperformance,
            'portfolio_history': portfolio_history,
            'benchmark_history': benchmark_history
        }
        
        # Plot performance
        self.plot_performance(portfolio_history, benchmark_history)
        
        return results
    
    def plot_performance(self, portfolio_history, benchmark_history):
        """
        Plot portfolio performance compared to benchmark.
        
        Args:
            portfolio_history (list): Portfolio value history
            benchmark_history (list): Benchmark value history
        """
        plt.figure(figsize=(14, 7))
        
        # Convert to percentage returns
        portfolio_returns = [(value / portfolio_history[0] - 1) * 100 for value in portfolio_history]
        benchmark_returns = [(value / benchmark_history[0] - 1) * 100 for value in benchmark_history]
        
        # Plot
        plt.plot(portfolio_returns, label=f'DRL Portfolio ({self.algorithm.upper()})')
        plt.plot(benchmark_returns, label='Equal Weight Benchmark', linestyle='--')
        
        plt.title('Portfolio Performance Comparison')
        plt.xlabel('Trading Days')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig(f'results/performance_{self.algorithm}.png')
        plt.close()
    
    def run_optimization(self, total_timesteps=100000):
        """
        Run the complete portfolio optimization process.
        
        Args:
            total_timesteps (int): Total training timesteps
            
        Returns:
            results (dict): Optimization results
        """
        # Fetch and preprocess data
        data = self.fetch_data()
        
        # Split data into train and test sets
        train_data, test_data = self.split_data(data)
        
        # Create training environment
        train_env = self.create_env(train_data)
        
        # Train model
        model = self.train_model(train_env, total_timesteps)
        
        # Create test environment
        test_env = self.create_env(test_data)
        
        # Evaluate model
        results = self.evaluate_model(model, test_env)
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """
        Print a summary of optimization results.
        
        Args:
            results (dict): Optimization results
        """
        print("\n" + "=" * 50)
        print("PORTFOLIO OPTIMIZATION RESULTS")
        print("=" * 50)
        print(f"Algorithm: {self.algorithm.upper()}")
        print(f"Stocks: {', '.join(self.stock_symbols)}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print("-" * 50)
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:.2f}")
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Benchmark Return: {results['benchmark_return']*100:.2f}%")
        print(f"Outperformance: {results['outperformance']*100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print(f"Maximum Drawdown: {results['max_drawdown']*100:.2f}%")
        print("=" * 50)
        print(f"Performance chart saved to: results/performance_{self.algorithm}.png")
        print("=" * 50)


def main():
    """
    Main function to demonstrate the portfolio optimization framework.
    """
    # Define optimization parameters
    stock_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS']
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    initial_capital = 10000
    window_size = 30
    algorithm = 'a2c'  # Choose from 'ppo', 'a2c', 'dqn', 'sac'
    
    # Create and run optimizer
    optimizer = PortfolioOptimizer(
        stock_symbols=stock_symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        window_size=window_size,
        algorithm=algorithm
    )
    
    # Run optimization process
    # results = optimizer.run_optimization(total_timesteps=50000)  # Reduced for demonstration
    results = optimizer.run_optimization(total_timesteps=10000)    
    return results


if __name__ == "__main__":
    main()
