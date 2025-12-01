# Market-risk-engine
# Market-Risk-Engine 📈

## Overview  
Market-Risk-Engine is a comprehensive, Python-based framework to measure and manage **market risk** for equity and fixed-income portfolios. It implements several standard risk-modeling techniques — including historical, parametric, and Monte Carlo Value-at-Risk (VaR), Conditional VaR (CVaR / Expected Shortfall), GARCH-based volatility forecasting, stress testing, and backtesting (Kupiec & Christoffersen) — and packages them into a cohesive, end-to-end risk-analysis engine.  

This project reflects how a bank or asset manager might build a risk-management tool internally, combining quantitative rigor with regulatory-compliance awareness (e.g., Basel III / ICAAP).  

---

## Features  

- **Historical VaR** — Non-parametric VaR calculation using historical P&L/returns distribution  
- **Parametric VaR (Variance–Covariance)** — Assuming normal returns, using covariance matrix of asset returns  
- **Monte Carlo VaR** — Simulate portfolio returns under random scenarios  
- **Conditional VaR / Expected Shortfall** — Tail-risk metrics at various confidence levels (95%, 99%)  
- **GARCH / Volatility Forecasting Module** — Estimate time-varying volatility for improved risk estimates under volatile markets  
- **Backtesting Suite**  
  - Kupiec Test (POF — Probability of Failure)  
  - Christoffersen Test (for independence of exceptions)  
- **Regulatory-style Capital Multiplier Logic (“Traffic Light” Framework)** — Tags model performance zones (Green / Yellow / Red) and applies “Increased Capital Multiplier” under Yellow Zone to reflect elevated risk capital requirements  
- **Stress Testing & Scenario Analysis** — Simulate extreme market conditions to gauge portfolio resilience  
- **Portfolio-level Risk Reporting** — Aggregated risk metrics (VaR, CVaR, stress losses) ready for analysis or dashboarding  

---

## Usage / Getting Started  

### Prerequisites  
- Python 3.8+  
- pandas, numpy, scipy, statsmodels, arch, matplotlib / seaborn / plotly (see `requirements.txt`)  

### Installation & Setup  
```bash
git clone https://github.com/phanimusunuri1234/Market-risk-engine.git  
cd Market-risk-engine  
pip install -r requirements.txt  
