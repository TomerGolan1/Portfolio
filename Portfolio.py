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
        try:
            data = web.DataReader(tickers_list, 'yahoo', start_date, end_date)
        except Exception:
            raise ValueError
        df = data['Adj Close']
        for i in range(len(tickers_list)):
            if np.isnan(df.iloc[0].values[i]):
                raise ValueError
        self.df = df
        self.stock_num = len(tickers_list)
        self.num_of_days= len(df)
        return df

    def p_per_day(self, day):
        p= self.df.iloc[day].values
        return p

    def x_vector(self,day):
        """
        :param day: day 1 represent as 0

        """
        if day == 0:
            x=[]
            for i in range(self.stock_num):
                x.append(1)
            return x
        else:
            p_before = self.p_per_day(day-1)
            p = self.p_per_day(day)
            x=[]
            for i in range(self.stock_num):
                x.append(p[i] / p_before[i])
            return x

    def new_portfolio(self, last_portfolio, day):
        """
        :param last_portfolio:  portfolio of the last day. First day is day 0
        :param day: minimum day is 1!
        """
        x_vector = self.x_vector(day)
        delta_money = np.dot(last_portfolio, x_vector)
        next_po = []
        for i in range(self.stock_num):
            next_po.append((last_portfolio[i] * x_vector[i])/ delta_money )
        return next_po

    def wealth(self, portfolio, day):
        """

        :param portfolio: portfolio of the first day
        :param day: first day represents as 0
        :return: wealth at the given day
        """
        wealth = 1
        if day == 0:
            return wealth
        port = portfolio
        for i in range(1,day+1):
            wealth = wealth * (np.dot(port,self.x_vector(i)))
            port = self.new_portfolio(port,i)
        return wealth

    def wealth_list(self, portfolio, day):
        """

        :param portfolio: portfolio of the first day
        :param day: first day represents as 0
        :return: wealth at the given day
        """
        wealth =1
        wealth_l = [1]
        if day == 0:    
            return wealth_l
        port = portfolio
        for i in range(1,day+1):
            wealth = wealth * (np.dot(port,self.x_vector(i)))
            wealth_l.append(wealth)
            port = self.new_portfolio(port,i)
        return wealth_l

    def wealth_next_day(self, wealth, portfolio, day):
        """
        :param portfolio: Get the portfolio of the given day
        :param day:
        :return: return wealth of the next day
        """
        next_wealth = wealth * np.dot(portfolio, self.x_vector(day))
        return next_wealth

    def combinations(self, a):
        jumps = [i*(1/a) for i in range(a+1)]
        list1 = list(itertools.product(jumps , repeat=self.stock_num))
        portfolio_list = []
        for i in range(len(list1)):
            sum_list = sum(list1[i])
            if sum_list > 0.999 and sum_list < 1.001 :
                portfolio_list.append(list1[i])
        return portfolio_list



    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        """
        calculates the universal portfolio for the previously requested stocks

        :param int portfolio_quantization: size of discrete steps of between computed portfolios. each step has size 1/portfolio_quantization
        :return: returns a list of floats, representing the growth trading  per day
        """
        list_of_floats = [1]
        first_port = [(1/self.stock_num) for i in range(self.stock_num)]
        s1 = self.wealth(first_port, 1)
        list_of_floats.append(s1)
        port_comb0 = self.combinations(portfolio_quantization)
        port_comb = np.array(port_comb0)
        matrix_of_wealth = []
        for i in range(len(port_comb)):
            w_of_c = self.wealth_list(port_comb[i], self.num_of_days-1)
            matrix_of_wealth.append(w_of_c)
        matrix_of_wealth = np.array(matrix_of_wealth)
        for i in range(2, self.num_of_days):
            mone = 0
            divide = 0
            for k in range(len(port_comb)):
                mone += port_comb[k] * matrix_of_wealth[k][i-1]
                divide += matrix_of_wealth[k][i-1]
            best_port = list(mone / divide)
            next_w = self.wealth_next_day(list_of_floats[i-1], best_port, i)
            list_of_floats.append(next_w)
        return list_of_floats





    def algoritem3(self, last_portfolio, n, day):
        """
        :param self:
        :param n: Parameter. Represents the learning rate.
        :param day:Represent the day which in his end, we use the exp portfolio before the next day.
        :return: Recommended algoritem, as array.
        """
        last_p = np.array(last_portfolio)
        best_port = []
        divide = 0
        vector_x = np.array(self.x_vector(day-1))
        dot_mul = np.dot(last_p, vector_x)
        for i in range(len(last_p)):
            last_port_i = last_p[i]
            x_vec_i = vector_x[i]
            mone = last_port_i * np.exp((n * x_vec_i) / dot_mul)
            divide += mone
            best_port.append(mone)
        best_port = np.array(best_port)
        best_port = list(best_port / divide)
        return best_port

    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        """
        calculates the exponential gradient portfolio for the previously requested stocks

        :param float learn_rate: the learning rate of the algorithm, defaults to 0.5
        :return: returns a list of floats, representing the growth trading  per day
        """
        list_of_floats = [1]
        first_port = [(1 / self.stock_num) for i in range(self.stock_num)]
        s1 = self.wealth(first_port, 1)
        list_of_floats.append(s1)
        last_port = first_port
        for i in range(2, self.num_of_days):
            best_port = self.algoritem3(last_port, learn_rate, i)
            w = self.wealth_next_day(list_of_floats[i-1], best_port, i)
            list_of_floats.append(w)
            last_port = best_port
        return list_of_floats



if __name__ == '__main__':  # You should keep this line for our auto-grading code.
    print('write your tests here')  # don't forget to indent your code here!
    start_date= date(2020,1,1)
    end_date= date(2020,2,1)
    l = ["GOOG","AAPL","MSFT"]
    x = PortfolioBuilder()
    y = x.get_daily_data(l , start_date, end_date)
    print(y)
    print(type(y.iloc[0].values[2]), np.isnan(y.iloc[0].values[2]))





