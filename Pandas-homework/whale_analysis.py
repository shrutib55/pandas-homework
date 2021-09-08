#!/usr/bin/env python
# coding: utf-8

#  #  A Whale off the Port(folio)
#  ---
# 
#  In this assignment, you'll get to use what you've learned this week to evaluate the performance among various algorithmic, hedge, and mutual fund portfolios and compare them against the S&P 500 Index.

# In[1]:


# Initial imports
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Cleaning
# 
# In this section, you will need to read the CSV files into DataFrames and perform any necessary data cleaning steps. After cleaning, combine all DataFrames into a single DataFrame.
# 
# Files:
# 
# * `whale_returns.csv`: Contains returns of some famous "whale" investors' portfolios.
# 
# * `algo_returns.csv`: Contains returns from the in-house trading algorithms from Harold's company.
# 
# * `sp500_history.csv`: Contains historical closing prices of the S&P 500 Index.

# ## Whale Returns
# 
# Read the Whale Portfolio daily returns and clean the data

# In[2]:


# Reading whale returns
whale_data = Path("Resources/whale_returns.csv")
whale_df = pd.read_csv(whale_data, index_col = "Date", infer_datetime_format=True, parse_dates=True)
whale_df.head()


# In[3]:


# Count nulls
whale_df.isnull().sum()


# In[4]:


# Drop nulls
whale_df.dropna(inplace = True)


# ## Algorithmic Daily Returns
# 
# Read the algorithmic daily returns and clean the data

# In[5]:


# Reading algorithmic returns
algo_data = Path("Resources/algo_returns.csv")
algo_df = pd.read_csv(algo_data, index_col = "Date", infer_datetime_format=True, parse_dates=True)
algo_df.head()


# In[6]:


# Count nulls
algo_df.isnull().sum()


# In[7]:


# Drop nulls
algo_df.dropna(inplace = True)


# ## S&P 500 Returns
# 
# Read the S&P 500 historic closing prices and create a new daily returns DataFrame from the data. 

# In[8]:


# Reading S&P 500 Closing Prices
sp500_history = Path("Resources/sp500_history.csv")
sp500_df = pd.read_csv(sp500_history, index_col = "Date", infer_datetime_format=True, parse_dates=True)
sp500_df = sp500_df.sort_index()
sp500_df.head()


# In[9]:


# Check Data Types
sp500_df.dtypes


# In[10]:


# Fix Data Types
sp500_df = sp500_df.replace({'\$': ''}, regex=True).astype(float)


# In[11]:


# Calculate Daily Returns
sp500_returns_df = sp500_df.pct_change()
sp500_returns_df.head()


# In[12]:


# Drop nulls
sp500_returns_df.dropna(inplace = True)


# In[13]:


# Rename `Close` Column to be specific to this portfolio.
sp500_returns_df = sp500_returns_df.rename(columns={'Close':'sp500'})
sp500_returns_df.head()


# ## Combine Whale, Algorithmic, and S&P 500 Returns

# In[14]:


# Join Whale Returns, Algorithmic Returns, and the S&P 500 Returns into a single DataFrame with columns for each portfolio's returns.
combined_df = pd.concat([whale_df, algo_df, sp500_returns_df], axis = "columns", join="inner")
combined_df.head()


# ---

# # Conduct Quantitative Analysis
# 
# In this section, you will calculate and visualize performance and risk metrics for the portfolios.

# ## Performance Anlysis
# 
# #### Calculate and Plot the daily returns.

# In[15]:


# Plot daily returns of all portfolios
combined_df.plot(figsize=(20,10));


# #### Calculate and Plot cumulative returns.

# In[16]:


# Calculate cumulative returns of all portfolios
cumulative_returns = (1 + combined_df).cumprod()
# Plot cumulative returns
cumulative_returns.plot(figsize=(20,10));


# ---

# ## Risk Analysis
# 
# Determine the _risk_ of each portfolio:
# 
# 1. Create a box plot for each portfolio. 
# 2. Calculate the standard deviation for all portfolios
# 4. Determine which portfolios are riskier than the S&P 500
# 5. Calculate the Annualized Standard Deviation

# ### Create a box plot for each portfolio
# 

# In[17]:


# Box plot to visually show risk
combined_df.plot.box(figsize=(20,10))


# ### Calculate Standard Deviations

# In[18]:


# Calculate the daily standard deviations of all portfolios
combined_df.std()


# ### Determine which portfolios are riskier than the S&P 500

# In[19]:


# Calculate  the daily standard deviation of S&P 500
sp500_risk = combined_df['sp500'].std()

# Determine which portfolios are riskier than the S&P 500
combined_df.std() > sp500_risk


# ### Calculate the Annualized Standard Deviation

# In[20]:


# Calculate the annualized standard deviation (252 trading days)
combined_df.std() * np.sqrt(252)


# ---

# ## Rolling Statistics
# 
# Risk changes over time. Analyze the rolling statistics for Risk and Beta. 
# 
# 1. Calculate and plot the rolling standard deviation for all portfolios using a 21-day window
# 2. Calculate the correlation between each stock to determine which portfolios may mimick the S&P 500
# 3. Choose one portfolio, then calculate and plot the 60-day rolling beta between it and the S&P 500

# ### Calculate and plot rolling `std` for all portfolios with 21-day window

# In[21]:


# Calculate the rolling standard deviation for all portfolios using a 21-day window
combined_df_rolling_std = combined_df.rolling(window=21).std()

# Plot the rolling standard deviation
combined_df_rolling_std.plot(figsize=(20,10));


# ### Calculate and plot the correlation

# In[22]:


# Calculate the correlation
correlation = combined_df.corr()

# Display de correlation matrix
correlation


# ### Calculate and Plot Beta for a chosen portfolio and the S&P 500

# In[23]:


# Calculate covariance of a single portfolio
covariance = combined_df['BERKSHIRE HATHAWAY INC'].rolling(window=60).cov(combined_df['sp500'])

# Calculate variance of S&P 500
variance = combined_df['sp500'].rolling(window=60).var()

# Computing beta
beta = covariance / variance

#Plot beta trend
beta.plot(figsize=(20,10))


# ## Rolling Statistics Challenge: Exponentially Weighted Average 
# 
# An alternative way to calculate a rolling window is to take the exponentially weighted moving average. This is like a moving window average, but it assigns greater importance to more recent observations. Try calculating the [`ewm`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html) with a 21-day half-life.

# In[24]:


# Use `ewm` to calculate the rolling window
combined_df.ewm(halflife=21).std().plot(figsize=(20,10));


# ---

# # Sharpe Ratios
# In reality, investment managers and thier institutional investors look at the ratio of return-to-risk, and not just returns alone. After all, if you could invest in one of two portfolios, and each offered the same 10% return, yet one offered lower risk, you'd take that one, right?
# 
# ### Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot

# In[25]:


# Annualized Sharpe Ratios
sharpe_ratios = (combined_df.mean() * 252) / (combined_df.std()*np.sqrt(252))
sharpe_ratios


# In[26]:


# Visualize the sharpe ratios as a bar plot
sharpe_ratios.plot(kind='bar');


# ### Determine whether the algorithmic strategies outperform both the market (S&P 500) and the whales portfolios.
# 
# Algo 1 outperformed the market and the whales portfolio.
# Algo 2 did not outperform the market or Berkshire Hathaway Inc, but it did outperform the rest of the whales portfolio.

# ---

# # Create Custom Portfolio
# 
# In this section, you will build your own portfolio of stocks, calculate the returns, and compare the results to the Whale Portfolios and the S&P 500. 
# 
# 1. Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
# 2. Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock
# 3. Join your portfolio returns to the DataFrame that contains all of the portfolio returns
# 4. Re-run the performance and risk analysis with your portfolio to see how it compares to the others
# 5. Include correlation analysis to determine which stocks (if any) are correlated

# ## Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
# 
# For this demo solution, we fetch data from three companies listes in the S&P 500 index.
# 
# * `GOOG` - [Google, LLC](https://en.wikipedia.org/wiki/Google)
# 
# * `AAPL` - [Apple Inc.](https://en.wikipedia.org/wiki/Apple_Inc.)
# 
# * `COST` - [Costco Wholesale Corporation](https://en.wikipedia.org/wiki/Costco)

# In[27]:


# Reading data from 1st stock
GOOG_data = Path("Resources/goog_historical.csv")
goog_df = pd.read_csv(GOOG_data, index_col = "Trade DATE", infer_datetime_format=True, parse_dates=True)
goog_df.head()


# In[28]:


# Reading data from 2nd stock
AAPL_data = Path("Resources/aapl_historical.csv")
aapl_df = pd.read_csv(AAPL_data, index_col = "Trade DATE", infer_datetime_format=True, parse_dates=True)
aapl_df.head()


# In[29]:


# Reading data from 3rd stock
COST_data = Path("Resources/cost_historical.csv")
cost_df = pd.read_csv(COST_data, index_col = "Trade DATE", infer_datetime_format=True, parse_dates=True)
cost_df.head()


# In[30]:


# Combine all stocks in a single DataFrame
all_stocks = pd.concat([goog_df, aapl_df, cost_df], axis=0, join="inner")


# In[31]:


# Reset Date index
all_stocks = all_stocks.reset_index()
all_stocks


# In[32]:


# Reorganize portfolio data by having a column per symbol
portfolio = all_stocks.pivot_table(values="NOCP", index="Trade DATE", columns="Symbol")
portfolio.head()


# In[33]:


# Calculate daily returns
daily_returns = portfolio.pct_change()

# Drop NAs
daily_returns = daily_returns.dropna().copy()

# Display sample data
daily_returns.head()


# ## Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock

# In[34]:


# Set weights
weights = [1/3, 1/3, 1/3]

# Calculate portfolio return
portfolio_return = daily_returns.dot(weights)

# Display sample data
portfolio_return.head()


# ## Join your portfolio returns to the DataFrame that contains all of the portfolio returns

# In[38]:


daily_returns['Weighted'] = portfolio_return
daily_returns


# In[39]:


# Only compare dates where return data exists for all the stocks (drop NaNs)
daily_returns.dropna(inplace = True)
daily_returns


# ## Re-run the risk analysis with your portfolio to see how it compares to the others

# ### Calculate the Annualized Standard Deviation

# In[41]:


# Calculate the annualized `std`
daily_returns.std() * np.sqrt(237)


# ### Calculate and plot rolling `std` with 21-day window

# In[42]:


# Calculate rolling standard deviation
combined_df_rolling_std = daily_returns.rolling(window=21).std()
# Plot rolling standard deviation
combined_df_rolling_std.plot(figsize=(20,10));


# ### Calculate and plot the correlation

# In[43]:


# Calculate and plot the correlation
correlation = daily_returns.corr()
correlation


# ### Calculate and Plot Rolling 60-day Beta for Your Portfolio compared to the S&P 500

# In[46]:


# Calculate and plot Beta
covariance = daily_returns['AAPL'].rolling(window=60).cov(sp500_df)

# Calculate variance of S&P 500
variance = sp500_df.rolling(window=60).var()

# Computing beta
beta = covariance / variance

#Plot beta trend
beta.plot(figsize=(20,10))


# ### Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot

# In[47]:


# Calculate Annualzied Sharpe Ratios
sharpe_ratios = (daily_returns.mean() * 237) / (daily_returns.std()*np.sqrt(237))
sharpe_ratios 


# In[48]:


# Visualize the sharpe ratios as a bar plot
sharpe_ratios.plot(kind='bar');


# ### How does your portfolio do?
# 
# Write your answer here!

#Costco outperformed the other stocks in the portfolio

