# Stock Portfolio Optimization using Deep Reinforcement Learning

This project implements a framework for optimizing stock portfolio allocation using Deep Reinforcement Learning (DRL) techniques. The system processes financial data including historical stock prices and technical indicators to train an intelligent agent capable of making data-driven investment decisions while balancing risk and return.

## Features

- **Data Acquisition**: Automatically fetch historical stock data from Yahoo Finance
- **Technical Analysis**: Calculate various technical indicators (SMA, EMA, RSI, MACD, etc.)
- **DRL Environment**: Custom Gymnasium environment for portfolio management simulation
- **Multiple Algorithms**: Support for various DRL algorithms (PPO, A2C, DQN, SAC)
- **Performance Metrics**: Comprehensive evaluation including Sharpe ratio, maximum drawdown, benchmark comparison
- **Visualization**: Performance charts comparing against an equal-weight benchmark

## Installation

### Option 1: Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-portfolio-optimization.git
   cd stock-portfolio-optimization
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t portfolio-optimization .
   ```

2. Run the container:
   ```bash
   docker run -it -v $(pwd)/results:/app/results portfolio-optimization
   ```

## Usage

### Basic Usage

Run the main script to start the optimization process:

```bash
python portfolio_optimization.py
```

### Customization

Edit the `main()` function in `portfolio_optimization.py` to customize:

```python
# Define optimization parameters
stock_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS']  # Example for Indian stocks
start_date = '2018-01-01'
end_date = '2023-01-01'
initial_capital = 10000
window_size = 30
algorithm = 'ppo'  # Choose from 'ppo', 'a2c', 'dqn', 'sac'
```

For Indian stocks, add the `.NS` suffix for stocks listed on the National Stock Exchange (e.g., `RELIANCE.NS`).

## Architecture

The project consists of two main classes:

1. **PortfolioOptimizationEnv**: A custom Gymnasium environment that simulates portfolio management with:
   - Market observations (prices, technical indicators)
   - Portfolio rebalancing actions
   - Reward calculation based on risk-adjusted returns

2. **PortfolioOptimizer**: The main class for handling the optimization process:
   - Data fetching and preprocessing
   - Environment creation
   - Model training and evaluation
   - Results visualization

## Performance Metrics

The system evaluates portfolio performance using several metrics:

- **Total Return**: Overall portfolio gain/loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Benchmark Comparison**: Performance compared to equal-weight allocation

## Files

- `portfolio_optimization.py`: Main Python script
- `requirements.txt`: Required Python packages
- `Dockerfile`: Docker configuration for containerized execution

## Output

The program creates several directories:

- `models/`: Saved DRL models
- `results/`: Performance charts and metrics
- `logs/` and `tensorboard/`: Training logs for monitoring

## Dependencies

- numpy, pandas: Data manipulation
- gymnasium: Reinforcement learning environment
- matplotlib: Visualization
- yfinance: Stock data acquisition
- stable-baselines3: DRL algorithm implementations
- ta: Technical analysis indicators

## Disclaimer

This project is for educational purposes only. The strategies implemented here are not guaranteed to provide profit in real-world trading scenarios. Always consult with a financial advisor before making investment decisions.

## License

[MIT License](LICENSE)
