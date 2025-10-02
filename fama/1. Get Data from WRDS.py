#!/usr/bin/env python
# coding: utf-8

# # Accessing and Managing Financial Data with Python
#
# This notebook demonstrates how to download and organize open-source financial data using Python. We'll cover:
# - Fama-French factors and portfolios
# - q-Factors
# - Macroeconomic predictors
# - FRED data
# - Setting up an SQLite database for data management

# ## Initial Setup
#
# First, let's load the required packages and define our date range.

# In[1]:


import pandas as pd
import numpy as np
import warnings

# Suppress FutureWarning about date_parser deprecation
warnings.filterwarnings("ignore", category=FutureWarning, message=".*date_parser.*")

try:
    import tidyfinance as tf
except ImportError:
    print("tidyfinance package not available - continuing without it")


# In[2]:


# pip install tidyfinance


# In[3]:


# Define the date range for data collection
start_date = "2020-01-01"
end_date = "2024-12-31"


# Lightweight debug utility to print date coverage for any dataset
def _debug_print_date_range(dataset_name, df, date_col="date"):
    try:
        dates = pd.to_datetime(df[date_col])
        date_min = dates.min()
        date_max = dates.max()
    except Exception:
        date_min = df[date_col].min()
        date_max = df[date_col].max()
    print(f"[DEBUG] {dataset_name} date range: {date_min} -> {date_max} (rows={len(df):,})")


# ## Fama-French Data
#
# We'll download the famous Fama-French factors and portfolio returns using the `pandas-datareader` package.

# In[4]:


try:
    import pandas_datareader as pdr
except ImportError:
    print(
        "pandas_datareader package not available - please install with: pip install pandas-datareader"
    )

# ### Fama-French 3 Factors (Monthly)

# In[5]:


factors_ff3_monthly_raw = pdr.DataReader(
    name="F-F_Research_Data_Factors", data_source="famafrench", start=start_date, end=end_date
)[0]

factors_ff3_monthly = (
    factors_ff3_monthly_raw.divide(100)
    .reset_index(names="date")
    .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
    .rename(str.lower, axis="columns")
    .rename(columns={"mkt-rf": "mkt_excess"})
)

_debug_print_date_range("FF3 monthly (famafrench)", factors_ff3_monthly)

factors_ff3_monthly.head()


# ### Fama-French 5 Factors (Monthly)

# In[6]:


factors_ff5_monthly_raw = pdr.DataReader(
    name="F-F_Research_Data_5_Factors_2x3", data_source="famafrench", start=start_date, end=end_date
)[0]

factors_ff5_monthly = (
    factors_ff5_monthly_raw.divide(100)
    .reset_index(names="date")
    .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
    .rename(str.lower, axis="columns")
    .rename(columns={"mkt-rf": "mkt_excess"})
)

_debug_print_date_range("FF5 monthly (famafrench)", factors_ff5_monthly)

factors_ff5_monthly.head()


# ### Fama-French 5 Factors (Daily)

# In[6a]


factors_ff5_daily_raw = pdr.DataReader(
    name="F-F_Research_Data_5_Factors_2x3_daily", data_source="famafrench", start=start_date, end=end_date
)[0]

factors_ff5_daily = (
    factors_ff5_daily_raw.divide(100)
    .reset_index(names="date")
    .rename(str.lower, axis="columns")
    .rename(columns={"mkt-rf": "mkt_excess"})
)

_debug_print_date_range("FF5 daily (famafrench)", factors_ff5_daily)

factors_ff5_daily.head()


# ### Fama-French 3 Factors (Daily)

# In[7]:


factors_ff3_daily_raw = pdr.DataReader(
    name="F-F_Research_Data_Factors_daily", data_source="famafrench", start=start_date, end=end_date
)[0]

factors_ff3_daily = (
    factors_ff3_daily_raw.divide(100)
    .reset_index(names="date")
    .rename(str.lower, axis="columns")
    .rename(columns={"mkt-rf": "mkt_excess"})
)

_debug_print_date_range("FF3 daily (famafrench)", factors_ff3_daily)

factors_ff3_daily.head()


# ### 10 Industry Portfolios (Monthly)

# In[8]:


industries_ff_monthly_raw = pdr.DataReader(
    name="10_Industry_Portfolios", data_source="famafrench", start=start_date, end=end_date
)[0]

industries_ff_monthly = (
    industries_ff_monthly_raw.divide(100)
    .reset_index(names="date")
    .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
    .rename(str.lower, axis="columns")
)

_debug_print_date_range("10 Industry Portfolios (monthly)", industries_ff_monthly)

industries_ff_monthly.head()


# ### Alternative: Using tidyfinance package

# In[9]:


# Alternative approach using tidyfinance
factors_ff3_alt = tf.download_data(
    domain="factors_ff",
    dataset="F-F_Research_Data_Factors",
    start_date=start_date,
    end_date=end_date,
)

_debug_print_date_range("FF3 monthly (tidyfinance)", factors_ff3_alt)

factors_ff3_alt.head()


# ## q-Factors
#
# Download the Hou, Xue, and Zhang (2015) q-factors from the authors' website.

# In[10]:


import ssl

# Temporarily adjust SSL settings
ssl._create_default_https_context = ssl._create_unverified_context

factors_q_monthly_link = (
    "https://global-q.org/uploads/1/2/2/6/122679606/q5_factors_monthly_2024.csv"
)

factors_q_monthly = (
    pd.read_csv(factors_q_monthly_link)
    .assign(
        date=lambda x: (
            pd.to_datetime(x["year"].astype(str) + "-" + x["month"].astype(str) + "-01")
        )
    )
    .drop(columns=["R_F", "R_MKT", "year"])
    .rename(columns=lambda x: x.replace("R_", "").lower())
    .query(f"date >= '{start_date}' and date <= '{end_date}'")
    .assign(**{col: lambda x: x[col] / 100 for col in ["me", "ia", "roe", "eg"]})
)

_debug_print_date_range("q-factors monthly (authors)", factors_q_monthly)

# Restore default SSL settings
ssl._create_default_https_context = ssl.create_default_context

factors_q_monthly.head()


# ### Alternative: Using tidyfinance package

# In[11]:


factors_q_alt = tf.download_data(
    domain="factors_q", dataset="q5_factors_monthly", start_date=start_date, end_date=end_date
)

_debug_print_date_range("q-factors monthly (tidyfinance)", factors_q_alt)

factors_q_alt.head()


# ## Macroeconomic Predictors
#
# Download macroeconomic variables from Amit Goyal's website, commonly used as predictors for the equity premium.

# In[12]:


sheet_id = "1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG"
sheet_name = "macro_predictors.xlsx"
macro_predictors_link = (
    f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
)


# In[13]:


ssl._create_default_https_context = ssl._create_unverified_context

macro_predictors = (
    pd.read_csv(macro_predictors_link, thousands=",")
    .assign(
        date=lambda x: pd.to_datetime(x["yyyymm"], format="%Y%m"),
        dp=lambda x: np.log(x["D12"]) - np.log(x["Index"]),
        dy=lambda x: np.log(x["D12"]) - np.log(x["Index"].shift(1)),
        ep=lambda x: np.log(x["E12"]) - np.log(x["Index"]),
        de=lambda x: np.log(x["D12"]) - np.log(x["E12"]),
        tms=lambda x: x["lty"] - x["tbl"],
        dfy=lambda x: x["BAA"] - x["AAA"],
    )
    .rename(columns={"b/m": "bm"})
    .get(
        [
            "date",
            "dp",
            "dy",
            "ep",
            "de",
            "svar",
            "bm",
            "ntis",
            "tbl",
            "lty",
            "ltr",
            "tms",
            "dfy",
            "infl",
        ]
    )
    .query("date >= @start_date and date <= @end_date")
    .dropna()
)

ssl._create_default_https_context = ssl.create_default_context

_debug_print_date_range("Macro predictors (Goyal)", macro_predictors)

macro_predictors.head()


# ### Alternative: Using tidyfinance package

# In[14]:


macro_predictors_alt = tf.download_data(
    domain="macro_predictors", dataset="monthly", start_date=start_date, end_date=end_date
)

macro_predictors_alt.head()


# ## Other Macroeconomic Data (FRED)
#
# Download Consumer Price Index (CPI) data from the Federal Reserve Economic Data (FRED).

# In[15]:


cpi_monthly = (
    pdr.DataReader(name="CPIAUCNS", data_source="fred", start=start_date, end=end_date)
    .reset_index(names="date")
    .rename(columns={"CPIAUCNS": "cpi"})
    .assign(cpi=lambda x: x["cpi"] / x["cpi"].iloc[-1])
)

_debug_print_date_range("CPI monthly (FRED)", cpi_monthly)

cpi_monthly.head()


# ## Setting Up an SQLite Database
#
# Now we'll create a database to store all the data for future use.

# In[16]:


import sqlite3
import os


# In[17]:


# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Create SQLite database connection
tidy_finance = sqlite3.connect(database="data/tidy_finance_python.sqlite")


# ### Store Data in Database

# In[18]:


# Store Fama-French 3 factors
factors_ff3_monthly.to_sql(
    name="factors_ff3_monthly", con=tidy_finance, if_exists="replace", index=False
)


# In[19]:


# Store all other datasets
data_dict = {
    "factors_ff5_monthly": factors_ff5_monthly,
    "factors_ff3_daily": factors_ff3_daily,
    "factors_ff5_daily": factors_ff5_daily,
    "industries_ff_monthly": industries_ff_monthly,
    "factors_q_monthly": factors_q_monthly,
    "macro_predictors": macro_predictors,
    "cpi_monthly": cpi_monthly,
}

for key, value in data_dict.items():
    value.to_sql(name=key, con=tidy_finance, if_exists="replace", index=False)

print("All datasets successfully stored in database!")


# ### Reading Data from Database
#
# Example of how to read data back from the database.

# In[20]:


# Read specific columns from database
sample_data = pd.read_sql_query(
    sql="SELECT date, rf FROM factors_ff3_monthly", con=tidy_finance, parse_dates={"date"}
)

sample_data.head()


# In[21]:


# Read entire table
factors_q_from_db = pd.read_sql_query(
    sql="SELECT * FROM factors_q_monthly", con=tidy_finance, parse_dates={"date"}
)

factors_q_from_db.head()


# ## Managing SQLite Databases
#
# Optimize database file size by running VACUUM command.

# In[22]:


# Optimize database (uncomment to run)
# tidy_finance.execute("VACUUM")
# print("Database optimized!")


# ## Exercises
#
# 1. Download the monthly Fama-French factors manually from Kenneth French's data library and read them in via `pd.read_csv()`. Validate that you get the same data as via the `pandas-datareader` package.
#
# 2. Download the daily Fama-French 5 factors using the `pdr.DataReader()` package. After the successful download and conversion to the column format that we used above, compare the `rf`, `mkt_excess`, `smb`, and `hml` columns of `factors_ff3_daily` to `factors_ff5_daily`. Discuss any differences you might find.

# In[23]:


# Your code for Exercise 1 here


# In[24]:


# Your code for Exercise 2 here
