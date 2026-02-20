import Commodity_fair_price_calc



# Main Analysis Function
def main():
    print("\n" + "="*80)
    print("COFFEE OPTION PRICING ANALYSIS")
    print("="*80 + "\n")
    
    # Initialize market data
    market_data = Commodity_fair_price_calc.CoffeeMarketData(
        spot_price=1.20,
        risk_free_rate=0.02,
        storage_cost=0.01,
        time_to_maturity=0.5,
        strike_price=1.25,
        volatility=0.25
    )
    
    print(market_data)
    
    # 1. Cost of Carry Model
    print("\n" + "="*80)
    coc = Commodity_fair_price_calc.CostOfCarryModel(market_data)
    coc_results = coc.analyze()
    
    # 2. Black-Scholes Model
    print("="*80)
    bs = Commodity_fair_price_calc.BlackScholesModel(market_data)
    bs_results = bs.analyze()
    
    # 3. Monte Carlo Simulation
    print("="*80)
    mc = Commodity_fair_price_calc.MonteCarloSimulation(market_data, num_simulations=100000, num_steps=252)
    mc_results = mc.analyze()
    
    # 4. Scenario Analysis
    print("="*80)
    scenario = Commodity_fair_price_calc.ScenarioAnalysis(market_data)
    scenario_results = scenario.analyze_scenarios()
    
    # 5. Visualization
    print("="*80)
    Commodity_fair_price_calc.visualize_results(market_data, mc_results, bs_results)
    
    # 6. Summary and Recommendations
    print("="*80)
    print("FINAL PRICING SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    print(f"\nFair Value Estimate (Consensus):")
    print(f"  Call Option: ${(bs_results['call_price'] + mc_results['call_price'])/2:.4f}/lb")
    print(f"  Put Option: ${(bs_results['put_price'] + mc_results['put_price'])/2:.4f}/lb")
    
    print(f"\nKey Risk Factors:")
    print(f"  1. Volatility Risk: High at {market_data.sigma*100:.0f}%")
    print(f"     - Recommendation: Monitor weather forecasts regularly")
    print(f"  2. Geopolitical Risk: Supplies concentrated in few regions")
    print(f"     - Recommendation: Hedge with put options for downside protection")
    print(f"  3. Storage Costs: {market_data.d*100:.0f}% annually affects forward prices")
    print(f"     - Recommendation: Optimize storage logistics")
    print(f"  4. Time Decay: Short-dated options lose ${bs_results['greeks']['theta_call']:.6f}/day")
    print(f"     - Recommendation: Roll positions before expiration")
    
    print(f"\nTrading Guidance:")
    print(f"  - Current spot is BELOW strike price (${market_data.St} < ${market_data.X})")
    print(f"  - Call options are OUT-OF-THE-MONEY")
    print(f"  - Put options are IN-THE-MONEY")
    print(f"  - Market expects {coc_results['cost_of_carry_rate']*100:.2f}% annual growth")
    
    print("\n" + "="*80)
    print("Analysis complete. Chart saved as 'coffee_option_analysis.png'")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()