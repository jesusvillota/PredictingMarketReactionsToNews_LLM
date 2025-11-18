# Options Trades Outlier Detection

This project focuses on analyzing options trades data to identify outliers and unusual trading patterns. The goal is to leverage market data and derived features to detect trades that deviate significantly from expected behavior, which can be useful for risk management, trading strategy refinement, and market surveillance.

## Purpose

The primary purpose of this project is to analyze a large dataset of options trades and underlying asset market data enriched with SpiderRock analytics. By engineering relevant features and applying statistical and machine learning techniques, the project aims to detect outliers in the options trading data.

## Variables Analyzed

The dataset includes a comprehensive set of variables related to options contracts, underlying assets, market data, and SpiderRock analytics. Key variables include:

- **Option Identifiers:** Variables defining the option contract such as `okey_ts`, `okey_at`, `okey_tk`, `okey_yr`, `okey_mn`, `okey_dy`, `okey_xx`, `okey_cp`.
- **Underlying Asset Identifiers:** Variables identifying the underlying asset like `undSecKey_at`, `undSecKey_ts`, `undSecKey_tk`, `undSecKey_yr`, `undSecKey_mn`, `undSecKey_dy`, `undSecType`.
- **Market Data:** Includes timestamps, bid and ask prices and sizes, cumulative sizes, exchange information, and trading session details.
- **SpiderRock Analytics:** A set of Greeks and volatility measures such as `bidIV`, `askIV`, `srPrc`, `srVol`, `de`, `ga`, `th`, `ve`, `rh`, `ph`, `vo`, `va`, `deDecay`, `sdiv`, `ddiv`, `rate`, `years`, and `atmVol`.

## Analysis Approach

The analysis involves the following steps:

1. **Data Loading:** Load the large options trades dataset efficiently using tools capable of handling big data formats.

2. **Feature Engineering:** Create new features that capture important aspects of the options market, such as mid-price, bid-ask spread, moneyness, price deviations from theoretical values, volatility differences, and time to expiration.

3. **Outlier Detection:** Apply statistical methods and machine learning models to identify trades that exhibit unusual characteristics compared to the typical market behavior.

4. **Interpretation:** Analyze the detected outliers to understand their nature and potential implications for trading and risk management.

This project aims to provide insights into the dynamics of options trading and help identify potentially significant or anomalous trades through rigorous data analysis and modeling.