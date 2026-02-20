"""
Coffee Option Pricing Analysis
Using Cost of Carry Model, Black-Scholes Model, and Monte Carlo Simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import pandas as pd


# Class to hold market data and parameters
class CoffeeMarketData:
    def __init__(self, spot_price, risk_free_rate, storage_cost,
                 time_to_maturity, strike_price, volatility):
        self.St = spot_price 
        self.r = risk_free_rate 
        self.d = storage_cost 
        self.T = time_to_maturity  
        self.X = strike_price 
        self.sigma = volatility 
        
    def __repr__(self):
        return f"""
Market Data Summary:
- Spot Price (St): ${self.St:.2f}/lb
- Risk-free Rate (r): {self.r*100:.2f}%
- Storage Cost (d): {self.d*100:.2f}%
- Time to Maturity (T): {self.T:.2f} years
- Strike Price (X): ${self.X:.2f}
- Volatility (σ): {self.sigma*100:.2f}%
"""


# Cost of Carry Model for forward price calculation and analysis
class CostOfCarryModel:
    """
    The cost of carry model determines the forward price of a commodity.
    Forward Price = S * e^((r-d)*T)
    where:
    - S: Spot price
    - r: Risk-free rate
    - d: Storage/carry cost
    - T: Time to maturity
    """
    
    def __init__(self, market_data):
        self.data = market_data
        
    def forward_price(self):
        """Calculate forward price using cost of carry model"""
        F = self.data.St * np.exp((self.data.r - self.data.d) * self.data.T)
        return F
    
    def cost_of_carry(self):
        """Calculate the cost of carry rate"""
        b = self.data.r - self.data.d
        return b
    
    def analyze(self):
        """Comprehensive analysis of cost of carry"""
        F = self.forward_price()
        b = self.cost_of_carry()
        carry_cost_total = self.data.St * (self.data.d * self.data.T)
        discount_effect = self.data.St * (self.data.r * self.data.T)
        
        print("=" * 60)
        print("COST OF CARRY MODEL ANALYSIS")
        print("=" * 60)
        print(f"Current Spot Price: ${self.data.St:.4f}/lb")
        print(f"Forward Price (F): ${F:.4f}/lb")
        print(f"Price Change: ${F - self.data.St:.4f} ({((F/self.data.St)-1)*100:.2f}%)")
        print(f"\nCost of Carry Rate (b): {b*100:.2f}%")
        print(f"Total Storage Cost Impact: ${carry_cost_total:.4f}")
        print(f"Total Time Value of Money: ${discount_effect:.4f}")
        print(f"Net Effect: ${(discount_effect - carry_cost_total):.4f}")
        print()
        
        return {
            'forward_price': F,
            'cost_of_carry_rate': b,
            'price_change': F - self.data.St,
            'storage_cost_total': carry_cost_total
        }


# Black Scholes Model for option pricing and Greeks calculation
class BlackScholesModel:
    """
    Black-Scholes pricing model for commodity options.
    Modified for commodities using the cost of carry approach.
    
    Call Price: S*e^(-d*T)*N(d1) - X*e^(-r*T)*N(d2)
    Put Price: X*e^(-r*T)*N(-d2) - S*e^(-d*T)*N(-d1)
    
    where:
    d1 = [ln(S/X) + (r-d+σ²/2)*T] / (σ*√T)
    d2 = d1 - σ*√T
    """
    
    def __init__(self, market_data):
        self.data = market_data
        
    def _calculate_d1_d2(self):
        """Calculate d1 and d2 parameters"""
        numerator = (np.log(self.data.St / self.data.X) + 
                    (self.data.r - self.data.d + 0.5 * self.data.sigma ** 2) * self.data.T)
        denominator = self.data.sigma * np.sqrt(self.data.T)
        d1 = numerator / denominator
        d2 = d1 - self.data.sigma * np.sqrt(self.data.T)
        return d1, d2
    
    def call_price(self):
        """Calculate call option price"""
        d1, d2 = self._calculate_d1_d2()
        call = (self.data.St * np.exp(-self.data.d * self.data.T) * norm.cdf(d1) - 
                self.data.X * np.exp(-self.data.r * self.data.T) * norm.cdf(d2))
        return call
    
    def put_price(self):
        """Calculate put option price"""
        d1, d2 = self._calculate_d1_d2()
        put = (self.data.X * np.exp(-self.data.r * self.data.T) * norm.cdf(-d2) - 
               self.data.St * np.exp(-self.data.d * self.data.T) * norm.cdf(-d1))
        return put
    
    def greeks(self):
        """Calculate option Greeks (sensitivity measures)"""
        d1, d2 = self._calculate_d1_d2()
        sqrt_T = np.sqrt(self.data.T)
        
        # Delta: rate of change of option price with respect to spot price
        delta_call = np.exp(-self.data.d * self.data.T) * norm.cdf(d1)
        delta_put = -np.exp(-self.data.d * self.data.T) * norm.cdf(-d1)
        
        # Gamma: rate of change of delta
        gamma = (np.exp(-self.data.d * self.data.T) * norm.pdf(d1) / 
                (self.data.St * self.data.sigma * sqrt_T))
        
        # Vega: sensitivity to volatility
        vega = self.data.St * np.exp(-self.data.d * self.data.T) * norm.pdf(d1) * sqrt_T / 100
        
        # Theta: time decay (per day)
        theta_call = (-self.data.St * np.exp(-self.data.d * self.data.T) * norm.pdf(d1) * self.data.sigma / (2 * sqrt_T) +
                     self.data.d * self.data.St * np.exp(-self.data.d * self.data.T) * norm.cdf(d1) -
                     self.data.r * self.data.X * np.exp(-self.data.r * self.data.T) * norm.cdf(d2)) / 365
        
        theta_put = (-self.data.St * np.exp(-self.data.d * self.data.T) * norm.pdf(d1) * self.data.sigma / (2 * sqrt_T) -
                    self.data.d * self.data.St * np.exp(-self.data.d * self.data.T) * norm.cdf(-d1) +
                    self.data.r * self.data.X * np.exp(-self.data.r * self.data.T) * norm.cdf(-d2)) / 365
        
        # Rho: sensitivity to interest rates
        rho_call = self.data.X * self.data.T * np.exp(-self.data.r * self.data.T) * norm.cdf(d2) / 100
        rho_put = -self.data.X * self.data.T * np.exp(-self.data.r * self.data.T) * norm.cdf(-d2) / 100
        
        return {
            'delta_call': delta_call,
            'delta_put': delta_put,
            'gamma': gamma,
            'vega': vega,
            'theta_call': theta_call,
            'theta_put': theta_put,
            'rho_call': rho_call,
            'rho_put': rho_put
        }
    
    def analyze(self):
        """Comprehensive Black-Scholes analysis"""
        call = self.call_price()
        put = self.put_price()
        greeks = self.greeks()
        
        print("=" * 60)
        print("BLACK-SCHOLES MODEL ANALYSIS")
        print("=" * 60)
        print(f"\nOption Prices:")
        print(f"  Call Option Price: ${call:.4f}/lb")
        print(f"  Put Option Price: ${put:.4f}/lb")
        print(f"  Call-Put Parity Check (C-P): ${call-put:.4f}")
        
        intrinsic_call = max(self.data.St - self.data.X, 0)
        intrinsic_put = max(self.data.X - self.data.St, 0)
        time_value_call = call - intrinsic_call
        time_value_put = put - intrinsic_put
        
        print(f"\nIntrinsic Values:")
        print(f"  Call Intrinsic: ${intrinsic_call:.4f} (Time Value: ${time_value_call:.4f})")
        print(f"  Put Intrinsic: ${intrinsic_put:.4f} (Time Value: ${time_value_put:.4f})")
        
        print(f"\nGreeks (Sensitivity Analysis):")
        print(f"  Delta Call: {greeks['delta_call']:.4f} (price change per $1 spot move)")
        print(f"  Delta Put: {greeks['delta_put']:.4f}")
        print(f"  Gamma: {greeks['gamma']:.6f} (delta change per $1 spot move)")
        print(f"  Vega: ${greeks['vega']:.4f} (price change per 1% volatility increase)")
        print(f"  Theta Call: ${greeks['theta_call']:.6f}/day (time decay)")
        print(f"  Theta Put: ${greeks['theta_put']:.6f}/day")
        print(f"  Rho Call: ${greeks['rho_call']:.6f} (price change per 1% rate increase)")
        print(f"  Rho Put: ${greeks['rho_put']:.6f}")
        print()
        
        return {
            'call_price': call,
            'put_price': put,
            'greeks': greeks,
            'intrinsic_call': intrinsic_call,
            'intrinsic_put': intrinsic_put
        }


# Monte Carlo Simulation for option pricing and risk analysis
class MonteCarloSimulation:
    """
    Monte Carlo simulation for option pricing using geometric Brownian motion.
    Spot Price Evolution: S(t) = S0 * exp((μ - σ²/2)*t + σ*√t*Z)
    where Z ~ N(0,1)
    """
    
    def __init__(self, market_data, num_simulations=100000, num_steps=252):
        self.data = market_data
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.dt = market_data.T / num_steps
        self.mu = market_data.r - market_data.d  # drift rate
        
    def _generate_paths(self):
        """Generate Monte Carlo price paths"""
        np.random.seed(42)  # For reproducibility
        
        # Initialize price matrix
        paths = np.zeros((self.num_steps + 1, self.num_simulations))
        paths[0] = self.data.St
        
        # Generate random returns
        for t in range(1, self.num_steps + 1):
            Z = np.random.standard_normal(self.num_simulations)
            paths[t] = paths[t-1] * np.exp(
                (self.mu - 0.5 * self.data.sigma ** 2) * self.dt +
                self.data.sigma * np.sqrt(self.dt) * Z
            )
        
        return paths
    
    def simulate(self):
        """Run Monte Carlo simulation"""
        paths = self._generate_paths()
        final_prices = paths[-1]
        
        # Calculate option payoffs
        call_payoffs = np.maximum(final_prices - self.data.X, 0)
        put_payoffs = np.maximum(self.data.X - final_prices, 0)
        
        # Discount to present value
        discount_factor = np.exp(-self.data.r * self.data.T)
        call_price_mc = np.mean(call_payoffs) * discount_factor
        put_price_mc = np.mean(put_payoffs) * discount_factor
        
        # Standard error
        call_std_error = np.std(call_payoffs) / np.sqrt(self.num_simulations) * discount_factor
        put_std_error = np.std(put_payoffs) / np.sqrt(self.num_simulations) * discount_factor
        
        return {
            'paths': paths,
            'final_prices': final_prices,
            'call_price': call_price_mc,
            'put_price': put_price_mc,
            'call_std_error': call_std_error,
            'put_std_error': put_std_error,
            'call_payoffs': call_payoffs,
            'put_payoffs': put_payoffs
        }
    
    def analyze(self):
        results = self.simulate()
        
        print("=" * 60)
        print("MONTE CARLO SIMULATION ANALYSIS")
        print(f"Simulations: {self.num_simulations:,} | Steps: {self.num_steps}")
        print("=" * 60)
        
        final_prices = results['final_prices']
        print(f"\nFinal Spot Price Distribution:")
        print(f"  Mean: ${np.mean(final_prices):.4f}")
        print(f"  Median: ${np.median(final_prices):.4f}")
        print(f"  Std Dev: ${np.std(final_prices):.4f}")
        print(f"  Min: ${np.min(final_prices):.4f}")
        print(f"  Max: ${np.max(final_prices):.4f}")
        print(f"  5th Percentile: ${np.percentile(final_prices, 5):.4f}")
        print(f"  95th Percentile: ${np.percentile(final_prices, 95):.4f}")
        
        print(f"\nOption Prices (MC):")
        print(f"  Call Price: ${results['call_price']:.4f} ± ${results['call_std_error']:.4f}")
        print(f"  Put Price: ${results['put_price']:.4f} ± ${results['put_std_error']:.4f}")
        
        print(f"\nProbability Analysis:")
        prob_in_the_money_call = np.sum(final_prices > self.data.X) / len(final_prices) * 100
        prob_in_the_money_put = np.sum(final_prices < self.data.X) / len(final_prices) * 100
        print(f"  Probability Call ITM: {prob_in_the_money_call:.2f}%")
        print(f"  Probability Put ITM: {prob_in_the_money_put:.2f}%")
        print()
        
        return results


# Scenario Analysis to evaluate impact of supply/demand shocks, weather events, and geopolitical risks on option pricing
class ScenarioAnalysis:
    """
    Analyze impact of supply, demand, weather, and geopolitical factors
    on option pricing through volatility and spot price adjustments.
    """
    
    def __init__(self, base_market_data):
        self.base_data = base_market_data
        
    def analyze_scenarios(self):
        """Analyze different market scenarios"""
        scenarios = {
            'Base Case': {
                'spot': self.base_data.St,
                'volatility': self.base_data.sigma,
                'description': 'Current market conditions'
            },
            'Supply Shock (Drought)': {
                'spot': self.base_data.St * 1.15,  # +15% spot price
                'volatility': self.base_data.sigma * 1.5,  # +50% volatility
                'description': 'Severe weather reduces supply'
            },
            'Supply Surplus': {
                'spot': self.base_data.St * 0.85,  # -15% spot price
                'volatility': self.base_data.sigma * 0.8,  # -20% volatility
                'description': 'Bumper crop increases supply'
            },
            'Geopolitical Crisis': {
                'spot': self.base_data.St * 1.25,  # +25% spot price
                'volatility': self.base_data.sigma * 2.0,  # +100% volatility
                'description': 'Conflict disrupts production/trade'
            },
            'Strong Demand': {
                'spot': self.base_data.St * 1.10,  # +10% spot price
                'volatility': self.base_data.sigma * 1.3,  # +30% volatility
                'description': 'Increased global demand'
            },
            'Weak Demand': {
                'spot': self.base_data.St * 0.90,  # -10% spot price
                'volatility': self.base_data.sigma * 0.9,  # -10% volatility
                'description': 'Economic slowdown reduces demand'
            }
        }
        
        print("=" * 80)
        print("SCENARIO ANALYSIS - IMPACT ON OPTION PRICES")
        print("=" * 80)
        
        results_df = []
        
        for scenario_name, scenario_params in scenarios.items():
            # Create market data for scenario
            scenario_data = CoffeeMarketData(
                spot_price=scenario_params['spot'],
                risk_free_rate=self.base_data.r,
                storage_cost=self.base_data.d,
                time_to_maturity=self.base_data.T,
                strike_price=self.base_data.X,
                volatility=scenario_params['volatility']
            )
            
            # Calculate prices
            bs = BlackScholesModel(scenario_data)
            call = bs.call_price()
            put = bs.put_price()
            
            # Calculate changes from base case
            base_bs = BlackScholesModel(self.base_data)
            base_call = base_bs.call_price()
            base_put = base_bs.put_price()
            
            call_change_pct = ((call - base_call) / base_call * 100) if base_call != 0 else 0
            put_change_pct = ((put - base_put) / base_put * 100) if base_put != 0 else 0
            
            results_df.append({
                'Scenario': scenario_name,
                'Spot Price': f"${scenario_params['spot']:.2f}",
                'Volatility': f"{scenario_params['volatility']*100:.1f}%",
                'Call Price': f"${call:.4f}",
                'Call Change': f"{call_change_pct:+.1f}%",
                'Put Price': f"${put:.4f}",
                'Put Change': f"{put_change_pct:+.1f}%",
                'Description': scenario_params['description']
            })
            
            print(f"\n{scenario_name}")
            print(f"  Description: {scenario_params['description']}")
            print(f"  Spot Price: ${scenario_params['spot']:.4f} ({((scenario_params['spot']/self.base_data.St)-1)*100:+.1f}%)")
            print(f"  Volatility: {scenario_params['volatility']*100:.1f}% ({((scenario_params['volatility']/self.base_data.sigma)-1)*100:+.1f}%)")
            print(f"  Call Price: ${call:.4f} ({call_change_pct:+.1f}%)")
            print(f"  Put Price: ${put:.4f} ({put_change_pct:+.1f}%)")
        
        print()
        df = pd.DataFrame(results_df)
        print("\nSummary Table:")
        print(df.to_string(index=False))
        print()
        
        return results_df


# Data Visualization of Monte Carlo paths, price distributions, and model comparisons
def visualize_results(market_data, mc_results, bs_results):
    """Create visualizations of pricing analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Coffee Option Pricing Analysis', fontsize=16, fontweight='bold')
    
    # 1. Monte Carlo Price Paths
    ax = axes[0, 0]
    paths = mc_results['paths']
    time_steps = np.linspace(0, market_data.T, paths.shape[0])
    
    # Plot sample paths
    for i in range(0, min(100, mc_results['paths'].shape[1]), 1):
        ax.plot(time_steps, paths[:, i], alpha=0.1, color='blue')
    
    # Plot mean and percentiles
    mean_path = np.mean(paths, axis=1)
    p5 = np.percentile(paths, 5, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    
    ax.plot(time_steps, mean_path, color='red', linewidth=2, label='Mean Path')
    ax.plot(time_steps, p5, color='orange', linewidth=2, linestyle='--', label='5th Percentile')
    ax.plot(time_steps, p95, color='green', linewidth=2, linestyle='--', label='95th Percentile')
    ax.axhline(y=market_data.X, color='black', linestyle=':', linewidth=2, label=f'Strike: ${market_data.X}')
    
    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Price ($/lb)')
    ax.set_title('Monte Carlo Price Paths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Final Price Distribution
    ax = axes[0, 1]
    final_prices = mc_results['final_prices']
    ax.hist(final_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(market_data.St, color='red', linestyle='--', linewidth=2, label=f'Current Spot: ${market_data.St}')
    ax.axvline(market_data.X, color='black', linestyle=':', linewidth=2, label=f'Strike: ${market_data.X}')
    ax.axvline(np.mean(final_prices), color='green', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(final_prices):.2f}')
    
    ax.set_xlabel('Price ($/lb)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Final Prices')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Greeks Sensitivity
    ax = axes[1, 0]
    greeks = bs_results['greeks']
    greek_names = ['Delta Call', 'Delta Put', 'Gamma', 'Vega']
    greek_values = [greeks['delta_call'], greeks['delta_put'], greeks['gamma']*100, greeks['vega']]
    colors = ['green', 'red', 'blue', 'orange']
    
    ax.bar(greek_names, greek_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Value')
    ax.set_title("Option Greeks (Sensitivity Measures)")
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Model Comparison
    ax = axes[1, 1]
    bs_call = bs_results['call_price']
    bs_put = bs_results['put_price']
    mc_call = mc_results['call_price']
    mc_put = mc_results['put_price']
    
    models = ['Black-Scholes\nCall', 'Black-Scholes\nPut', 'Monte Carlo\nCall', 'Monte Carlo\nPut']
    prices = [bs_call, bs_put, mc_call, mc_put]
    errors = [0, 0, mc_results['call_std_error'], mc_results['put_std_error']]
    colors_comp = ['green', 'red', 'green', 'red']
    
    bars = ax.bar(models, prices, color=colors_comp, alpha=0.7, edgecolor='black', yerr=errors, capsize=5)
    ax.set_ylabel('Price ($/lb)')
    ax.set_title('Model Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('coffee_option_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'coffee_option_analysis.png'")
    plt.show()
