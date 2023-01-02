# None of these imports are strictly required, but use of at least some is strongly encouraged
# Other imports which don't require installation can be used without consulting with course staff.
# If you feel these aren't sufficient, and you need other modules which require installation,
# you're welcome to consult with the course staff.

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import date
import itertools
import math
import yfinance
from typing import List


class PortfolioBuilder:

    def get_daily_data(self, tickers_list: List[str],
                       start_date: date,
                       end_date: date = date.today()
                       ) -> pd.DataFrame:
        """
        get stock tickers adj_close price for specified dates.

        :param List[str] tickers_list: stock tickers names as a list of strings.
        :param date start_date: first date for query
        :param date end_date: optional, last date for query, if not used assumes today
        :return: daily adjusted close price data as a pandas DataFrame
        :rtype: pd.DataFrame

        example call: get_daily_data(['GOOG', 'INTC', 'MSFT', ''AAPL'], date(2018, 12, 31), date(2019, 12, 31))
        """
        pass

    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        """
        calculates the universal portfolio for the previously requested stocks

        :param int portfolio_quantization: size of discrete steps of between computed portfolios. each step has size 1/portfolio_quantization
        :return: returns a list of floats, representing the growth trading  per day
        """
        pass

    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        """
        calculates the exponential gradient portfolio for the previously requested stocks

        :param float learn_rate: the learning rate of the algorithm, defaults to 0.5
        :return: returns a list of floats, representing the growth trading  per day
        """
        pass


if __name__ == '__main__':  # You should keep this line for our auto-grading code.
    print('write your tests here')  # don't forget to indent your code here!
