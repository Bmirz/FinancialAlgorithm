# -*- coding: utf-8 -*-
"""
SEC Filing Scraper
"""

# import modules
import requests
import pandas as pd

# create request header
headers = {'User-Agent': "bbmirza117@gmail.com"}

# get all companies data
companyTickers = requests.get(
    "https://www.sec.gov/files/company_tickers.json",
    headers=headers
    )

# review response / keys
print(companyTickers.json().keys())