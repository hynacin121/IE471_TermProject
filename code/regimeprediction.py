import pandas as pd
import openpyxl
import numpy as np
import cvxpy as cp
import scipy as sp
from matplotlib import pyplot as plt

class regime:

    def __init__(self, dir):
        self.dir = dir
        self.kospi = pd.read_csv(dir + 'data/KOSPI.csv', index_col=0, parse_dates=True)['Adj Close']
        self.inflation = pd.read_csv(dir + 'data/inflation.csv', index_col= 0)/100
        self.inflation.index = self.kospi.iloc[1:-3,].index
        self.inflation.columns = ['소비자물가', '농축수산물', '공업제품','집세'	,'공공서비스'	, '개인서비스'	,'근원물가'	,'생활물가']
        self.indicator = pd.read_csv(dir + 'data/indicator.csv', index_col = 0, parse_dates= True)
        self.indicator.columns =  ['선행종합지수', '재고순환지표(%p)', '경제심리지수(p)',
                                    '기계류내수출하지수(%)', '건설수주액(%)', '수출입물가비율(%)',
                                    '코스피(%)', '장단기금리차(%p)', '선행종합지수 순환변동치 전월차(p)', '선행지수 전년동월비(%)',
                                    '선행종합지수 전년동월비 전월차(%p)', '동행종합지수', '동행종합지수 전월비(%)',
                                    '광공업생산지수(%)', '서비스업생산지수(%)', '건설기성액(%)',
                                    '소매판매액지수(%)', '내수출하지수(%)', '수입액(%)', '비농림어업취업자수(%)',
                                    '동행종합지수 순환변동치 전월차(p)', '생산자제품재고지수(전월비)', '소비자물가지수변화율(전월차)(%p)',
                                    '소비재수입액(%)', '취업자수(%)', 'CP유통수익률(%p)']

        
    def zscore(self):
        inf2 = self.inflation.copy()
        inflation = self.inflation
        inf2.iloc[0,:] = [100, 100, 100, 100, 100, 100, 100, 100]

        for i in range(inf2.shape[0] -1):
          for j in range(inf2.shape[1]):
            inf2.iloc[i+1,j] = inf2.iloc[i,j] * (inflation.iloc[i+1, j]+1)

        inf_z = pd.DataFrame(columns = inflation.columns)

        for i in range(len(inflation)-48):
            mean = inflation.iloc[i:i+48,].mean()
            std = inflation.iloc[i:i+48,].std()
            zscore = pd.DataFrame((inflation.iloc[i+48,] - mean)/std).T
            inf_z = pd.concat([inf_z, zscore], axis = 0)

        inf_z.index = inflation.iloc[48:,].index      

        return(inf_z)
    
    def l1trendfiltering_findlambda(self, lambda_list):
        kospi = self.kospi
        y = kospi.to_numpy()
        y = np.log(y)
        n = y.size
        ones_row = np.ones((1, n))
        D = sp.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)
        solver = cp.CVXOPT
        reg_norm = 2

        fig, ax = plt.subplots(2, 3, figsize=(40,20))
        ax = ax.ravel()

        ii = 0

        for lambda_value in lambda_list:    
            x = cp.Variable(shape=n)    
    
    # x is the filtered trend that we initialize    
    
            objective = cp.Minimize(0.5 * cp.sum_squares(y-x) 
                  + lambda_value * cp.norm(D@x, reg_norm))    
    
    # Note: D@x is syntax for matrix multiplication    
    
            problem = cp.Problem(objective)
            problem.solve(solver=solver, verbose=False)    
    
            ax[ii].plot(kospi.index, y, linewidth=1.0, c='b')
            ax[ii].plot(kospi.index, np.array(x.value), 'b-', linewidth=1.0, c='r')
            ax[ii].set_xlabel('Time')
            ax[ii].set_ylabel('Log Premium')
            ax[ii].set_title('Lambda: {}\nSolver: {}\nObjective Value: {}'.format(lambda_value, problem.status, round(objective.value, 3)))    
    
            ii+=1
    
        plt.tight_layout()
        plt.savefig('trend_filtering_L{}.png'.format(reg_norm))
        plt.show()
    def l1trendfiltering(self, lmbda):
        df = pd.DataFrame(self.kospi)
        df = df.squeeze()
                
        y = df.to_numpy()
        y = np.log(y)
        n = y.size
        ones_row = np.ones((1, n))
        D = sp.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), n-2, n)       
        solver = cp.CVXOPT
        reg_norm = 2
        x = cp.Variable(shape=n)    
        # x is the filtered trend that we initialize    
        objective = cp.Minimize(lmbda * cp.sum_squares(y-x) 
                    + lmbda * cp.norm(D@x, reg_norm))    
        # Note: D@x is syntax for matrix multiplication    
        problem = cp.Problem(objective)
        problem.solve(solver=solver, verbose=False)
        df_x_value= pd.DataFrame(np.exp(x.value), index=df.index, columns=['x_value'])        
        df_x_value = df_x_value.pct_change().dropna()       
        
        return(df_x_value)

    def visualization(self):
        l1 = self.l1trendfiltering()
        ret = pd.DataFrame(self.kospi.pct_change().dropna()).iloc[:-1, :]
        df2 = pd.concat([ret, l1], axis =1)
        df2.columns = ['KS', 'KS_L1'] 
        df3 = df2 + 1
        df4 = df3.copy()
        df4.iloc[0,:] = [100, 100]

        for i in range(len(df3)-1):
          for j in range(2):
            df4.iloc[i+1,j] = df4.iloc[i,j] * df3.iloc[i+1, j]

        plt.plot(df4)

    def labeling_regime(self, lmbda):
        l1 = self.l1trendfiltering(lmbda)
        inf_z = self.zscore()
        index_list = []
        for i in range (3, len(l1)):
          if l1['x_value'].values[i] < 0 and l1['x_value'].values[i-1] < 0 and l1['x_value'].values[i-2] < 0 and l1['x_value'].values[i-3] < 0: 
            index_list.append(l1.index[i-3])
            index_list.append(l1.index[i-2])
            index_list.append(l1.index[i-1])
            index_list.append(l1.index[i])

        index_list = list(dict.fromkeys(index_list))      
        regime = pd.DataFrame(index = index_list)
        regime['stock'] = 0


        regime = pd.concat([inf_z.iloc[:,0], regime], axis = 1).fillna(1)
        regime['regime'] = 0
        length = regime.shape[0]
        for i in range(length):
            if (regime.iloc[i,0] < 0) & (regime.iloc[i,1] == 0):
              regime.iloc[i,2] = 4
            elif (regime.iloc[i,0] < 0) & (regime.iloc[i,1] == 1):
              regime.iloc[i,2] = 3
            elif (regime.iloc[i,0] >= 0) & (regime.iloc[i,1] == 0):
              regime.iloc[i,2] = 1
            elif (regime.iloc[i,0] >= 0) & (regime.iloc[i,1] == 1):
              regime.iloc[i,2] = 2

        regime = regime["2004":]
        regime.to_csv('regime.csv')
        
        return(regime)

    
    
    def preprocess(self):
        label_regime = self.labeling_regime(1)
        ind = self.indicator
        ind = ind[['재고순환지표(%p)', '경제심리지수(p)',
                    '기계류내수출하지수(%)', '건설수주액(%)', '수출입물가비율(%)',
                    '코스피(%)', '장단기금리차(%p)', '선행종합지수 순환변동치 전월차(p)', '선행지수 전년동월비(%)',
                    '동행종합지수 전월비(%)',
                    '광공업생산지수(%)', '서비스업생산지수(%)', '건설기성액(%)',
                    '소매판매액지수(%)', '내수출하지수(%)', '수입액(%)', '비농림어업취업자수(%)',
                    '동행종합지수 순환변동치 전월차(p)', '생산자제품재고지수(전월비)', '소비자물가지수변화율(전월차)(%p)',
                    '소비재수입액(%)', '취업자수(%)', 'CP유통수익률(%p)']]
        train_data = int(len(ind) * 0.8)
        ind = ind['2004':]
        ind.index= label_regime.index
        
        ind = pd.concat([ind, label_regime['regime']], axis = 1)
        X = ind.iloc[:, :-1]
        Y = ind.iloc[:, -1]
        Y = Y.shift(-1).dropna()
        df2 = pd.concat([X.iloc[ :-1, : ], Y], axis=1)
  
        ind_x_train = df2.iloc[:train_data, :-1]
        ind_x_test = df2.iloc[train_data:, :-1]
        ind_y_train=  df2.iloc[:train_data, -1] 
        ind_y_test = df2.iloc[train_data:, -1]

        return(ind_x_train, ind_x_test, ind_y_train, ind_y_test)

