import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import scipy.stats as stat
from sklearn.neighbors import KernelDensity
import scipy.optimize as sco
from datetime import datetime

# 单资产高斯核函数
def guassian_fit(fit_data, bw=0.3, plot=False):
    """
    Args:
        fit_data: (Series/Array) asset's return rate
        plot: (Boolean) True if to check density plot
        
    Returns:
        test_kde: (KernelDensity) fitted guassian kernel
        sigma: standard variance
        mu: average value
    """
    test_return = fit_data/100
    mu = np.average(test_return); sigma = np.std(test_return)
    std_test = ((test_return-mu)/sigma).values.reshape(-1,1)

    test_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(std_test)
    simulate_value = np.exp(test_kde.score_samples(std_test))


    if plot == True:
        print("Guassian fit result:\n")
        plt.figure(figsize=(8,5))
        plt.plot(std_test[:, 0], simulate_value, '.')
        plt.show()
    
    return test_kde, sigma, mu


# 单资产蒙特卡洛
def monte_carlo(start_point, simulate_period, iteration, plot=False):
    """
    Returns:
        S: (array) dimension = simulation_period*iteration
    """
    S0 = start_point; simulate_period = simulate_period; I = iteration

    S = np.zeros((simulate_period,I))
    S[0] = S0

    for t in range(1, simulate_period):
        S[t] = S[t-1]*((test_kde.sample(I).reshape(-1)*sigma + mu)+1)

    # Plot
    if plot==True:
        
        plt.subplot(2,2,1)
        plt.figure(figsize=(8,5))
        plt.hist(S[-1],bins = 50)
        plt.xlabel('price')
        plt.ylabel('frequency')
        plt.show()

        plt.subplot(2,2,2)
        plt.figure(figsize=(8,5))
        for i in range(10):
            plt.plot(S.T[i])
        plt.xlabel('time')
        plt.ylabel('price')
        plt.show()
        
    return S


# 定义一次仿真的风险计量
def risk_mc(each_data, target_x = 0.01, target_y = -0.01):
    """
    Args:
        each_data: (list/array) a record of one simulation period
        target_x: (float) expected return
        target_y: (float) max loss
        
    Returns:
        counter: (int) how many times that X_t<target_x & Y_t<target_y
        """
    target_x = target_x; target_y = target_y
    X_t = []; counter = 0
    for i in range(len(each_data)-1):
        if (each_data[i+1]-each_data[0]) > 0 :
            x = np.log(each_data[i+1]-each_data[0])/100
            X_t.append(x)
        elif (each_data[i+1]-each_data[0]) < 0:
            x = -np.log(each_data[0]-each_data[i+1])
            X_t.append(x)
        else:
            X_t.append(0)

    X_t = np.asarray(X_t)
    Y_t = min(np.asarray(X_t))
    
    if Y_t < target_y:
        counter += 1
    else:
        if X_t[-1] < target_x:
            counter += 1
    
    return counter

# 多次仿真
def multiple_mc(simulation_data, plot=True):
    simulation_times = len(simulation_data)
    risk_lst = []; counters = 0
    for t in range(simulation_times):
        counter = risk_mc(S.T[t])
        counters += counter
        risk = counters/(t+1)
        risk_lst.append(risk)
    
    if plot == True:
        plt.style.use('seaborn')
        title = input("asset name?\n")
        plt.plot(risk_lst)
        plt.xlabel('simulation times')
        plt.ylabel('risk measure')
        plt.title("%s risk measurement via monte-carlo"% title)
        plt.show()
        
    #return risk_lst


# cholesky decompostion
def cholesky(data, n): 
    """
    Args:
        data: (dataframe) return data with asset names as column name. 
        n: (int) 调仓周期
     
    Returns:
        asset_path: (dataframe) with assets' price walk paths according to joint probability.
    """
    cov_matrix = data.cov()
    decomp_A = np.linalg.cholesky(cov_matrix)
    
    for i in range(len(data.columns)): # i = asset index
        locals()[data.columns[i] + "_kde"], locals()[data.columns[i] + "_sigma"], locals()[data.columns[i] + "_mu"] = guassian_fit(data.iloc[:,i])
        locals()[data.columns[i] + "_joint"] = 0
    
    locals()[data.columns[0] + "_joint"] = locals()[data.columns[0] + "_kde"].sample(n)
    
    for i in range(1,len(data.columns)):
        for j in range(1,len(data.columns)):
            locals()[data.columns[i] + "_joint"] += locals()[data.columns[j-1] + "_kde"].sample(n) - np.average(data.iloc[:,j-1]) / \
                                                 decomp_A[j-1,j-1] * decomp_A[j,j-1] + locals()[data.columns[j] + "_kde"].sample(n)
    
    
    lst = []
    for i in range(len(data.columns)):
        locals()[data.columns[i] + "_joints"] = locals()[data.columns[i] + "_sigma"] *locals()[data.columns[i] + "_joint"] +locals()[data.columns[i] + "_mu"]
        lst.append(locals()[data.columns[i] + "_joints"].reshape(-1))
    lst = np.array(lst).T
    asset_path = pd.DataFrame(lst, columns=data.columns)
    
    return asset_path


# 基于联合分布的多资产蒙特卡洛
def monte_carlo_cholesky(data, assetname, simulate_period, iteration, plot=True):
    """
    Args:
        data: (dataframe)multi-assets' return data
        
    Return:
        S: (array) price walk path
    """
    S0 = data[assetname][0]; simulate_period = simulate_period; I = iteration

    S = np.zeros((simulate_period,I))
    S[0] = S0
    
    for t in range(1, simulate_period):
        S[t] = S[t-1]*((cholesky(data,I)[assetname].reshape(-1)[t-1])/100+1)
    # Plot
    if plot==True:
        plt.subplot(2,2,1)
        plt.hist(S[-1],bins = 50)
        plt.xlabel('price')
        plt.ylabel('frequency')
        plt.show()

        plt.subplot(2,2,2)
        for i in range(10):
            plt.plot(S.T[i])
        plt.xlabel('time')
        plt.ylabel('price')
        plt.show()
        
    return S


# 定义一次仿真的风险计量
def risk_counter(each_data, target_x = 0.01, target_y = -0.01):
    """
    Args:
        each_data: (list/array) a record of one simulation period
        target_x: (float) expected return
        target_y: (float) max loss
        
    Returns:
        counter: (int) how many times that X_t<target_x & Y_t<target_y
        """
    target_x = target_x; target_y = target_y
    X_t = []; counter = 0
    for i in range(len(each_data)-1):
        if (each_data[i+1]-each_data[0]) > 0 :
            x = np.log(each_data[i+1]-each_data[0])/100
            X_t.append(x)
        elif (each_data[i+1]-each_data[0]) < 0:
            x = -np.log(each_data[0]-each_data[i+1])
            X_t.append(x)
        else:
            X_t.append(0)

    X_t = np.asarray(X_t)
    Y_t = min(np.asarray(X_t))
    
    if Y_t < target_y:
        counter += 1
    else:
        if X_t[-1] < target_x:
            counter += 1
    
    return counter

# 优化模型      
def opt_weights(paths, target_x = 0.01, target_y = -0.01, trade_period=20):
    """
    Optimized asset weights for certain period
    Args: 
        paths: (dataframe) joint asset path
        
    Return:
        opt: (scipy opimazation result)
    """
    weights = np.random.random(int(len(paths.columns)/trade_period))
    weights /= np.sum(weights)

        
    def opt_func(weights):
        counters = 0
        for i in range(len(paths)):
            #joint_returns = np.array([paths.iloc[i, j] for j in range(paths.shape[1])])
            trade_return = []
            for k in range(0, trade_period):
                pair_returns = np.array([paths.iloc[i,k], paths.iloc[i,k+trade_period]])
                portfolio_returns = np.dot(pair_returns, weights)
                trade_return.append(portfolio_returns)
                
            trade_return=np.array(trade_return)
            counter = risk_counter(trade_return)
            counters += counter
        risk = counters/len(paths)
        
        return risk
                                      
    # set opt model  
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
    bnds = tuple((0,1) for x in range(len(weights)))
    opts = sco.minimize(opt_func, x0=weights, bounds = bnds, constraints = cons) # initial guess: average
    
    return opts


# 资产配置
def allocation(fulldata, data, obs_period, trade_period, iteration=1500, target_x = 0.01, target_y = -0.01):
    """
    Args:
        fulldata: (dataframe) assets' close price & return info (i.e. from 2006 to 2017)
        data: (dataframe) assets' return info from 2006 to 2017
        obs_period: (int) length of obeservation period
        obs_period: (int) length of trading period
        
    Returns:
        weight_df: (dataframe) with optimized weights allocation
    """
    weight_df = pd.DataFrame(columns=["Weights"+ str(i+1) for i in range(len(data.columns))], index=fulldata.index[obs_period:]); j=0
    
    for t in range(0, len(data), trade_period)[int(obs_period/trade_period):-1]:  # set rolling windows
        print("rolling the windows: ", j, "\n")
        mini_df = data.iloc[t-obs_period:t, :] # get assets' returns
        for assetname in data.columns:
            locals()[assetname + "_S"] =  monte_carlo_cholesky(mini_df, assetname, trade_period, iteration, plot=False)
        
        lst=[]
        for assetname in mini_df.columns:
            lst.append(locals()[assetname + "_S"])
        lst = np.array(lst); lst = lst.reshape(lst.shape[0]*lst.shape[1],lst.shape[2]).T
        joint_columns = [assetname+str(num) for assetname in mini_df.columns for num in range(1,trade_period+1)]
        joint_path = pd.DataFrame(lst, columns=joint_columns) # shape=(simulation times, trade_period*asset number)

        opt_result = opt_weights(joint_path,target_x = target_x, target_y = target_y, trade_period=trade_period)
        for i in range(trade_period):
            for asset_num in range(len(data.columns)):
                weight_df["Weights"+str(asset_num+1)][j] = np.tile(opt_result.x, (trade_period,1))[i][asset_num]
            j += 1
    weight_df = pd.concat([weight_df,fulldata.loc[weight_df.index, :]], axis=1)
        
    return weight_df


# 总风险、期内风险、期末风险计数
def risk_counter_IEH(each_data, target_x = 0.01, target_y = -0.01):
    """
    Args:
        each_data: (list/array) a record of one simulation period
        target_x: (float) expected return
        target_y: (float) max loss
        
    Returns:
        counter: (int) how many times that X_t<target_x & Y_t<target_y
        """
    target_x = target_x; target_y = target_y
    X_t = []; counter = 0; IH = 0; EH = 0
    for i in range(len(each_data)-1):
        if (each_data[i+1]-each_data[0]) > 0 :
            x = np.log(each_data[i+1]-each_data[0])/100
            X_t.append(x)
        elif (each_data[i+1]-each_data[0]) < 0:
            x = -np.log(each_data[0]-each_data[i+1])
            X_t.append(x)
        else:
            X_t.append(0)

    X_t = np.asarray(X_t)
    Y_t = min(np.asarray(X_t))
    
    if Y_t < target_y:
        counter += 1
        IH += 1
    else:
        if X_t[-1] < target_x:
            counter += 1
            EH += 1
    
    return counter, IH, EH

# 优化结果评估
def evaluation(df, trade_period, basic_info=True, net_plot=True, allocation_plot=True): 
    """
    Args:
        df: (dataframe) obtained from allocation function
        trade_period: (int) consistent with before
        basic_info:(Boolean) contains annual return, max drawback, sharpe ratio and risks
        net_plot:(Boolean) plot net value line
        
    Returns:
        new_df: (dataframe) with calculated portfolio net value and daily return
    """
    # 计算基金净值(封闭式 期内不调整份额)
    port_lst = []; asset_num = int(df.shape[1]/3) # each asset has a weight, close and chg
    for i in range(len(df)):
        port = 0; n = 0
        for j in range(asset_num):
            port += df.iloc[i,j]*df.iloc[i,j+asset_num+n]
            n += 1
        port_lst.append(port)

    offering= port_lst[0]
    port_net_value = pd.DataFrame(np.array(port_lst)/offering, columns=["Port_Value"], index=df.index)
    port_return = pd.DataFrame(np.array((port_net_value["Port_Value"]-port_net_value["Port_Value"].shift(1).fillna(0))/port_net_value["Port_Value"]), \
                               columns=["Port_Return"], index=df.index)

    new_df = pd.concat([df, port_net_value, port_return], axis=1)
    
    # 年化收益率 最大回撤 夏普比率 触发总风险概率 期内风险概率 期末风险
    if basic_info:
        #annual_return = (1 + new_df["Port_Return"][1:].mean())**(252/len(new_df["Port_Return"][1:])) - 1
        annual_return = new_df["Port_Return"][1:].mean() * 252
        e = np.argmax(np.maximum.accumulate(new_df["Port_Value"]) - new_df["Port_Value"][1:])
        s = np.argmax(new_df["Port_Value"][:e])
        max_drawdown = float(new_df["Port_Value"][s]-new_df["Port_Value"][e]) / new_df["Port_Value"][s]
        sharpe_ratio = (annual_return - 0.03)/ new_df["Port_Return"].std()  # update risk free rate here
        
        all_risk = 0; all_IH = 0; all_EH = 0; all_period = 0
        for t in range(1, len(new_df), trade_period):
            all_period += 1
            single_period = new_df["Port_Return"][t:t+trade_period]
            risk, IH, EH = risk_counter_IEH(single_period)
            all_risk += risk; all_IH += IH; all_EH += EH
        res_risk = all_risk/all_period; res_IH = all_IH/all_period; res_EH = all_EH/all_period
        
        print("Annuall return of portfolio is: ", annual_return)
        print("\n Max drawdown of portfolio is:", max_drawdown)
        print("\n Sharpe ratio of portfolio is:", sharpe_ratio)
        print("\n Total risk prob of portfolio is:", res_risk)
        print("\n In horizon risk of portfolio is:", res_IH)
        print("\n End of horizon risk of portfolio is:", res_EH)
    
    if net_plot == True:
        print("Net Curve:\n")
        plt.figure(figsize=(8,5))
        plt.style.use('seaborn')

        # pre-setting
        dates = new_df.index[[i for i in range(0,len(new_df),243)]].tolist() 
        xs = [datetime.strptime(d, '%d/%m/%Y').date() for d in dates]
        ys = range(0,len(df.index), int(len(df.index)/len(xs)))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(ys, xs, rotation=45)
        plt.gcf().autofmt_xdate()

        # plot y
        plt.plot(np.array(new_df["Port_Value"]),label="Portfolio")
        try:
            CBA_net = np.array(new_df["CBA_CLOSE"]/new_df["CBA_CLOSE"][0])
            plt.plot(CBA_net,label="CBA_Net")
        except:
            pass

        plt.legend(bbox_to_anchor=(1,1)) 
        plt.title('Portfolio Net Curve')
        plt.show()
    
    if allocation_plot==True:
        print("Asset allocation:\n")
        for i in range(1,asset_num+1):
            locals()["Weights"+ str(i)] = new_df["Weights"+ str(i)].values.tolist()

        plt.figure(figsize=(8,5))
        plt.style.use('seaborn')

        # pre-setting for x axis
        dates = new_df.index[[i for i in range(0,len(new_df),243)]].tolist() 
        xs = [datetime.strptime(d, '%d/%m/%Y').date() for d in dates]
        ys = range(0,len(df.index), int(len(df.index)/len(xs)))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(ys, xs, rotation=45)
        plt.gcf().autofmt_xdate()

        # plot y
        plt.bar(df.index, locals()["Weights"+ str(1)], label="Weights of Asset "+ str(1))
        for i in range(2,asset_num+1):
            plt.bar(df.index, locals()["Weights"+ str(i)], bottom=locals()["Weights"+ str(i-1)], label="Weights of Asset "+ str(i))
        plt.legend(bbox_to_anchor=(1,1)) 
        plt.title('Asset allocations')
        plt.show()

    return new_df


if __name__ == "__main__":
	CSI = pd.read_csv("CSI.csv", index_col=0)
	CBA = pd.read_csv("CBA.csv", index_col=0)

	# prepare data
	fulldata = pd.concat([CSI, CBA], axis=1).iloc[:, [3,5,6,8]]
	fulldata.columns = ["CSI_CLOSE", "CSI_Return", "CBA_CLOSE", "CBA_Return"]
	return_data = fulldata.iloc[:,[1,3]]
	return_data.columns = ["CSI", "CBA"]

	port = allocation(fulldata, return_data, 120, 20, iteration=2000, target_x = 0.01, target_y = -0.01).dropna()
	eva_port = evaluation(port, 20, basic_info=True, net_plot=True, allocation_plot=True)
