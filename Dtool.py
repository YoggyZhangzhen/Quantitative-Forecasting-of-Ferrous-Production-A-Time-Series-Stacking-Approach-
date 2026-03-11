import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings

from statsmodels.tsa.stattools import acf, q_stat
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric.smoothers_lowess import lowess

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


from scipy.stats import linregress

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import plot_importance


import math
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def perform_kpss_test(residuals):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kpss_test = kpss(residuals, regression='c')
        kpss_statistic = kpss_test[0]
        kpss_p_value = kpss_test[1]
        kpss_critical_values = kpss_test[3]

        print(f'KPSS Statistic: {kpss_statistic:.4f}')
        print(f'p-value: {kpss_p_value:.4f}')

        if kpss_statistic < kpss_critical_values['5%']:
            print("The residuals are stationary (fail to reject the null hypothesis).")
        else:
            print("The residuals are not stationary (reject the null hypothesis).")

        # Check for any warning and print a message if the KPSS statistic is outside the range
        if len(w) > 0 and issubclass(w[0].category, Warning):
            print("Warning: The test statistic is outside the range of p-values available in the look-up table. "
                  "The actual p-value is smaller than the returned value.")
            
            
# ADF 检验函数
def perform_adf_test(residuals):
    adf_test = adfuller(residuals)
    adf_statistic = adf_test[0]
    p_value = adf_test[1]
    critical_values = adf_test[4]

    print(f'ADF Statistic: {adf_statistic:.4f}')
    print(f'p-value: {p_value:.4f}')

    if p_value < 0.05:
        print("The residuals are stationary (reject the null hypothesis).")
    else:
        print("The residuals are not stationary (fail to reject the null hypothesis).")

# Ljung-Box 检验函数
def perform_ljungbox_test(residuals, lags=10):
    ljung_box_results = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    lb_stat = ljung_box_results['lb_stat'].values[0]
    p_value = ljung_box_results['lb_pvalue'].values[0]

    print(f"Ljung-Box Statistic: {lb_stat:.4f}")
    print(f"Ljung-Box p-value: {p_value:.4f}")


    if p_value > 0.05:
        print("The residuals are random (fail to reject the null hypothesis of no autocorrelation).")
    else:
        print("The residuals are not random (reject the null hypothesis of no autocorrelation).")


# 对齐数据
def align_and_adjust(df_resampled, base_year=2024, file_prefix='aligned_weekly'):
    """
    对齐并调整日期，适用于任何周度数据。保留缺失值 (NaN) 并调整日期，使其与指定的基准年份对齐。

    参数：
    - df_resampled: 周度数据的DataFrame，包含要处理的列。
    - base_year: 用于对齐的基准年份（默认2024年）。
    - file_prefix: 输出文件的前缀（默认'aligned_weekly'）。
    """

    # 确保索引为 DatetimeIndex
    if not isinstance(df_resampled.index, pd.DatetimeIndex):
        df_resampled.index = pd.to_datetime(df_resampled.index)

    # 创建完整的基准年份的周五时间序列
    fridays_base_year = pd.date_range(start=f'{base_year}-01-05', end=f'{base_year}-12-27', freq='W-FRI')

    # 提取年份和周数
    df_resampled['年份'] = df_resampled.index.year
    df_resampled['周数'] = df_resampled.index.isocalendar().week

    # 自动获取数据中的年份范围
    years_range = range(df_resampled['年份'].min(), df_resampled['年份'].max() + 1)
    weeks_range = range(1, 53)

    # 创建全年的周数组合
    index = pd.MultiIndex.from_product([years_range, weeks_range], names=['年份', '周数'])

    # 重建索引，使得数据对齐到完整的年份和周数组合
    df_resampled_aligned = df_resampled.set_index(['年份', '周数']).reindex(index).reset_index()

    # 保留缺失值为 NaN
    df_resampled_aligned['trend'] = df_resampled_aligned['trend'].round(3)
    df_resampled_aligned['seasonal'] = df_resampled_aligned['seasonal'].round(3)
    df_resampled_aligned['residual'] = df_resampled_aligned['residual'].round(3)

    # 使用基准年份的周五时间序列创建周数到日期的映射
    week_to_date_map_base_year = {i + 1: date for i, date in enumerate(fridays_base_year)}

    # 定义日期调整函数
    def adjust_dates(row):
        week = row['周数']
        year = row['年份']
        if week in week_to_date_map_base_year:
            base_date = week_to_date_map_base_year[week]
            adjusted_date = base_date.replace(year=int(year))
            return adjusted_date
        return pd.NaT

    # 应用日期调整
    df_resampled_aligned['日期'] = df_resampled_aligned.apply(adjust_dates, axis=1)

    # 移除未来日期（当前日期之后的行）
    current_date = pd.Timestamp.today()
    df_resampled_aligned = df_resampled_aligned[df_resampled_aligned['日期'] <= current_date]

    # 设置调整后的日期为索引
    df_resampled_aligned.set_index('日期', inplace=True)

    # 检查并提示缺失值
    missing_values = df_resampled_aligned[df_resampled_aligned.isna().any(axis=1)]
    if not missing_values.empty:
        print(f"警告：存在缺失值，缺失的周数为：\n{missing_values[['年份', '周数']]}")

    # 保存对齐后的数据到不同的CSV文件
    df_resampled_aligned[['trend']].to_csv(f'{file_prefix}_trend.csv', date_format='%Y-%m-%d')
    df_resampled_aligned[['seasonal']].to_csv(f'{file_prefix}_seasonal.csv', date_format='%Y-%m-%d')
    df_resampled_aligned[['residual']].to_csv(f'{file_prefix}_residual.csv', date_format='%Y-%m-%d')

    # 返回处理后的DataFrame
    return df_resampled_aligned


# stl 拆分存储
def test_stl_parameters(data, value_col, seasonal, trend, period, seasonal_deg, trend_deg, low_pass_deg, robust):
    """
    参数：
    - data: 输入的 DataFrame，包含待分解的时间序列数据。
    - value_col: 数据中包含时间序列值的列名。
    - seasonal: 季节性成分窗口大小。
    - trend: 趋势成分窗口大小。
    - period: 数据的周期。
    - seasonal_deg: STL 分解中的季节性多项式次数。
    - trend_deg: STL 分解中的趋势多项式次数。
    - low_pass_deg: STL 分解中的低通滤波多项式次数。
    - robust: 是否使用稳健方法。
    """

    stl = STL(
        data[value_col],
        seasonal=seasonal,
        trend=trend,
        period=period,
        low_pass=None,
        seasonal_deg=seasonal_deg,
        trend_deg=trend_deg,
        low_pass_deg=low_pass_deg,
        seasonal_jump=1,
        trend_jump=1,
        low_pass_jump=1,
        robust=robust
    )
    
    result = stl.fit()

    # Generate new column names based on the original column name
    trend_col = f'{value_col}_trend'
    seasonal_col = f'{value_col}_seasonal'
    residual_col = f'{value_col}_residual'

    # Add the decomposition results to the DataFrame with new column names
    data[trend_col] = result.trend
    data[seasonal_col] = result.seasonal
    data[residual_col] = result.resid

    # 计算残差标准差、ADF 检验、KPSS 检验、Ljung-Box 检验
    residual_std = np.std(data[residual_col])
    print(f"Residual Std Dev: {residual_std:.4f}")

    print("\nADF Test Results:")
    perform_adf_test(data[residual_col])

    print("\nKPSS Test Results:")
    perform_kpss_test(data[residual_col])

    print("\nLjung-Box Test Results:")
    perform_ljungbox_test(data[residual_col])

    plt.figure(figsize=(14, 8))

    plt.subplot(4, 1, 1)
    plt.plot(data.index, data[value_col], label='Original Data', color='blue')
    plt.title(f'Original Data (Robust={robust}, seasonal={seasonal}, trend={trend}, period={period})')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(data.index, data[trend_col], label='Trend', color='green')
    plt.title(f'Trend Component ({trend_col})')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(data.index, data[seasonal_col], label='Seasonal', color='orange')
    plt.title(f'Seasonal Component ({seasonal_col})')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(data.index, data[residual_col], label='Residual', color='red')
    plt.title(f'Residual Component ({residual_col})')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_predictions(y_test, future_data, col=True, label=True):
    # 根据col参数确定要使用的列名
    if col:
        pred_col = '预测值_全量'
        if pred_col not in future_data.columns:
            pred_col = '预测值_全值'
    else:
        pred_col = '预测值'
        
    # 检查列是否存在
    if pred_col not in future_data.columns:
        raise ValueError(f'列名 {pred_col} 不存在')
        
    # 计算调整值
    if label:
        gap = y_test.iloc[-1] - future_data[pred_col].iloc[0]

    # 调整预测值
    print(f"Gap between last actual value and first prediction: {gap}")
    return future_data


'''
future_data = plot_predictions(y_test, future_data, col=True, label=True)
'''

'''
params_to_test = [
   {
       'data': df1,      
       'value_col': '值',          
       'seasonal': 53,            
       'trend': 55,              
       'period': 53,            
       'seasonal_deg': 1,    
       'trend_deg': 2,            
       'low_pass_deg': 1,         
       'robust': False             
   }
    # 可以继续添加更多参数组合
]

for params in params_to_test:
   test_stl_parameters(**params)

'''
   
# 画出所有图
def plot_factors(df, df_name):
    """
    Plot each column in the DataFrame with the same x-axis (index).
    
    Parameters:
    - df: Pandas DataFrame containing the data to be plotted.
    - df_name: A string representing the name of the DataFrame (for title purposes).
    """
    for column in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[column], label=column)
        plt.title(f'{df_name}: {column}', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

'''
plot_factors(aligned_weekly, 'aligned_daily')
'''

# 选取特别col 画图
def plot_factors_by_pattern(df, df_name, pattern=None):
    """
    根据指定的列名模式绘制 DataFrame 中的列。
    
    Parameters:
    - df: Pandas DataFrame containing the data to be plotted.
    - df_name: A string representing the name of the DataFrame (for title purposes).
    - pattern: A string representing the pattern for selecting columns to plot (e.g., "trend", "residual").
               If None, all columns will be plotted.
    """
    # 如果给定了 pattern，选择列名中包含该 pattern 的列
    if pattern:
        columns_to_plot = [col for col in df.columns if pattern in col]
    else:
        columns_to_plot = df.columns

    # 绘制符合条件的列
    for column in columns_to_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[column], label=column)
        plt.title(f'{df_name}: {column}', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

'''
plot_factors_by_pattern(df, 'My DataFrame', pattern='trend')
plot_factors_by_pattern(df, 'My DataFrame')
'''

#空缺值填写
def fill_missing_values(df, fill_methods, return_only_filled=True):
    """
    根据每个因子的特性选择不同的填充方式
    
    参数:
    df: 需要处理的 DataFrame
    fill_methods: 一个字典，其中键是列名，值是填充方法，如 'mean', 'median', 'ffill', 'bfill', 'interpolate', 'none', 'mean_of_5', 'rolling_mean_5'
    return_only_filled: 布尔值, 是否只返回填充过的列, 默认为 True
    
    返回:
    返回一个新的 DataFrame，只包含指定列并按相应方法填充完毕
    """
    filled_df = pd.DataFrame()  # 创建一个空的 DataFrame 用于存储填充过的因子

    for factor, method in fill_methods.items():
        if factor in df.columns:

            df.loc[:, factor] = pd.to_numeric(df[factor], errors='coerce')

            if method == 'mean':
                filled_df[factor] = df[factor].fillna(df[factor].mean()).infer_objects(copy=False)
            elif method == 'median':
                filled_df[factor] = df[factor].fillna(df[factor].median()).infer_objects(copy=False)
            elif method == 'ffill':
                filled_df[factor] = df[factor].fillna(method='ffill').infer_objects(copy=False)
            elif method == 'bfill':
                filled_df[factor] = df[factor].fillna(method='bfill').infer_objects(copy=False)
            elif method == 'interpolate':
                filled_df[factor] = df[factor].infer_objects(copy=False).interpolate(method='linear')
            elif method == 'mean_of_5':
                filled_df[factor] = df[factor].copy()  # 先复制原始数据
                for i in range(len(filled_df[factor])):
                    if pd.isnull(filled_df[factor].iloc[i]):  # 检查是否为空
                        # 获取前后五个非空值，使用 pd.concat 替代 append
                        surrounding_values = pd.concat([
                            df[factor].iloc[max(0, i - 5):i].dropna(),
                            df[factor].iloc[i + 1:min(len(df[factor]), i + 6)].dropna()
                        ])
                        if len(surrounding_values) > 0:
                            # 使用周围非空值的平均值填充
                            filled_df.loc[filled_df.index[i], factor] = surrounding_values.mean()
            elif method == 'rolling_mean_5':     # 更平滑一点
                # 用滚动窗口的平均值填充
                filled_df[factor] = df[factor].fillna(df[factor].rolling(window=5, min_periods=1).mean()).infer_objects(copy=False)
            elif method == 'none':
                filled_df[factor] = df[factor]  # 不做填充，返回原始数据
            else:
                print(f"未知的填充方法: {method}")
        else:
            print(f"因子 {factor} 不存在于 DataFrame 中")

    # 如果设置了 return_only_filled=False, 则返回所有原始数据+处理过的列
    if not return_only_filled:
        remaining_df = df.drop(columns=filled_df.columns)  # 删除已处理列
        return pd.concat([remaining_df, filled_df], axis=1)
    
    return filled_df  # 只返回处理过的列

'''
fill_methods = {
   '螺纹高炉成本': 'mean',  # 使用均值填充
   '螺纹表需': 'median',     # 使用中位数填充
  '30大中城市商品房成交面积/30DMA': 'none'        # 不进行填充，保留原始数据
}

# 调用函数进行填充，并只返回被填充的列
filled_data = fill_missing_values(aligned_daily, fill_methods)

# 如果想返回整个DataFrame，包括没填充的列，可以使用:
filled_data_with_all = fill_missing_values(aligned_daily, fill_methods, return_only_filled=False)

:'interpolate'
:'rolling_mean_5'
:'rolling_mean_5'

'''

# daily数据变成weekly
def daily_to_weekly(df_daily, cols_to_process, date_column='日期', method='mean'):
    """
    将日度数据转换为周度数据，按周五对齐，并计算每周的平均值。
    
    参数:
    df_daily: 包含日度数据的 DataFrame，索引为日期。
    cols_to_process: 需要处理的列的列表。
    date_column: 用于对齐的日期列名，默认为 '日期'。
    method: 填充每周的计算方式，默认使用 'mean'（平均值），可以根据需要修改。

    返回:
    返回一个新的 DataFrame，转换为周度数据，日期对齐到每周五。
    """
    # 生成周五为频率的日期范围
    weekly_date_range = pd.date_range(start='2016-09-02', end='2024-10-04', freq='W-FRI')
    
    # 创建一个空的 DataFrame，索引为周五的日期范围
    df_weekly = pd.DataFrame(index=weekly_date_range)

    # 对每个需要处理的列进行周度转换
    for column in cols_to_process:
        if column in df_daily.columns:
            # 按周进行重采样，并计算每周的平均值，忽略缺失值
            df_weekly[column] = df_daily[column].resample('W-FRI').apply(lambda x: x.mean() if len(x.dropna()) > 0 else np.nan)
        else:
            print(f"列 {column} 不存在于 DataFrame 中")

    return df_weekly

'''

cols_to_process = ['螺纹表需', '螺纹高炉成本']

# 调用函数，将日度数据转换为周度数据
weekly_data = daily_to_weekly(df_daily, cols_to_process)

'''

def plot_comparison_multiple(df, main_col, compare_cols, start_date=None, end_date=None):
    """
    将一个主要指标与多个其他指标进行比较，并且允许选择指定时间范围。
    
    Parameters:
    - df: 包含多个指标的 Pandas DataFrame，索引为日期。
    - main_col: 主要指标的列名（字符串），该列将与多个其他指标进行对比。
    - compare_cols: 需要与主要指标进行比较的其他列的列表。
    - start_date: 可选，开始日期，指定要绘制的时间范围。
    - end_date: 可选，结束日期，指定要绘制的时间范围。
    """
    
    # 如果指定了时间范围，限制数据范围
    if start_date and end_date:
        df = df.loc[start_date:end_date]

    # 归一化数据函数
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    
    # 归一化主要指标
    main_data_normalized = normalize(df[main_col])

    # 绘制主要指标与多个其他指标的对比图
    for col in compare_cols:
        if col in df.columns:
            compare_data_normalized = normalize(df[col])
            
            # 绘制图表
            plt.figure(figsize=(10, 6))
            plt.plot(main_data_normalized.index, main_data_normalized, label=main_col, color='b')
            plt.plot(compare_data_normalized.index, compare_data_normalized, label=col, linestyle='--', color='r')
            
            # 添加标题和标签
            plt.title(f'Comparison: {main_col} vs {col}', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Normalized Value', fontsize=12)
            
            # 图例和网格
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"列 '{col}' 不存在于 DataFrame 中。")

'''

plot_comparison_multiple(
    filled_data, 
    main_col='螺纹总库存_trend', 
    compare_cols=['螺纹总库存_seasonal', '螺纹总库存_residual'], 
    start_date='2021-01-01', 
    end_date='2023-01-01'
)

---

# 预先定义多个比较组合
comparisons = {
    '螺纹总库存_trend_vs_日均铁水产量': {
        'main_col': '螺纹总库存_trend',
        'compare_cols': ['日均铁水产量']
    },
    '螺纹总库存_trend_vs_多个指标': {
        'main_col': '螺纹总库存_trend',
        'compare_cols': ['日均铁水产量', '螺纹总库存']
    }
}

# 选择一个组合来进行比较
selected_comparison_key = '螺纹总库存_trend_vs_多个指标' # 你可以从上面打印的组合中选择一个
selected_comparison = comparisons[selected_comparison_key]

# 调用 plot_comparison_multiple 函数，传入选择的组合和时间范围
plot_comparison_multiple(
    filled_data, 
    selected_comparison['main_col'], 
    selected_comparison['compare_cols'], 
    start_date='2021-01-01',  # 可选的开始时间
    end_date='2023-01-01'     # 可选的结束时间
)
---

# 假设你已经提前定义了主要指标和比较指标
main_col = '螺纹总库存_trend'
compare_cols = ['日均铁水产量', '螺纹总库存']

# 现在调用 plot_comparison_multiple 函数时，直接使用这些变量
plot_comparison_multiple(filled_data, main_col, compare_cols,start_date='2021-01-01', end_date='2023-01-01')

'''





def process_outliers(data, column, window=29, std_multiplier=2):
    """
    处理数据中的异常波动，使用滚动窗口计算标准差，超出指定标准差倍数的异常值进行平滑处理。
    
    参数:
    - data (pd.DataFrame): 输入数据，必须包含一个日期索引和处理的列。
    - column (str): 需要处理的列名。
    - window (int): 滑动窗口大小，用于计算标准差，默认为20天。
    - std_multiplier (float): 标准差倍数，用于判断异常值，默认为2倍标准差。
    
    返回:
    - pd.DataFrame: 返回处理后的DataFrame，包含异常处理的列。
    """
    processed_data = data.copy()
    
    # 计算滑动均值和标准差
    rolling_mean = processed_data[column].rolling(window=window, min_periods=1).mean()
    rolling_std = processed_data[column].rolling(window=window, min_periods=1).std()
    
    # 定义上限和下限
    upper_limit = rolling_mean + std_multiplier * rolling_std
    lower_limit = rolling_mean - std_multiplier * rolling_std
    
    # 平滑处理超过阈值的异常值
    processed_data[column] = np.where(
        processed_data[column] > upper_limit, upper_limit,
        np.where(processed_data[column] < lower_limit, lower_limit, processed_data[column])
    )
    
    return processed_data


'''
processed_data = process_outliers(df, column="WTI连1-连4月差", window=20, std_multiplier=2)

'''




def align_data(xls, sheet_name, date_freq, start_date='2016-09-02', end_date='2024-10-04'):
    """
    读取并对齐数据，根据指定频率对齐每日、每周、每月或每年数据。
    
    参数:
    - xls: Excel 文件路径或文件对象
    - sheet_name: 表名，如 '日度数据'、'周度数据' 等
    - date_freq: 日期频率（'D' 表示每日, 'W-FRI' 表示每周五, 'M' 表示每月最后一天, 'A' 表示每年最后一天）
    - start_date: 对齐的开始日期
    - end_date: 对齐的结束日期
    
    返回:
    - 对齐后的 DataFrame
    """
    data = pd.read_excel(xls, sheet_name=sheet_name, header=0)
    date_range = pd.date_range(start=start_date, end=end_date, freq=date_freq)
    aligned_data = pd.DataFrame(index=date_range)

    for i in range(0, len(data.columns), 2):
        if i + 1 < len(data.columns):  # 确保成对的列存在
            factor_name = data.columns[i]  # 因子名称
            df = data[[data.columns[i], data.columns[i + 1]]].copy()  # 提取因子的两列
            df.columns = ['日期', factor_name]  # 重新命名列
            df['日期'] = pd.to_datetime(df['日期'], format='%Y-%m-%d', errors='coerce')  # 转换日期为 datetime
            df.drop_duplicates(subset=['日期'], inplace=True)  # 去掉重复日期
            df.set_index('日期', inplace=True)  # 将日期设置为索引
            aligned_data[factor_name] = df.reindex(aligned_data.index)[factor_name]  # 对齐到指定频率的日期

    return aligned_data

'''
aligned_daily = align_data(xls, sheet_name='日度数据', date_freq='D', start_date='2016-09-02', end_date='2024-10-04')
'''


def reverse_column(df, column_name):
    """
    将指定列的数值进行逆序，使得最大值变为最小值，最小值变为最大值。
    
    参数:
    df (pd.DataFrame): 包含要逆序列的 DataFrame
    column_name (str): 要逆序的列名

    返回:
    pd.Series: 逆序后的列
    """
    max_value = df[column_name].max()
    min_value = df[column_name].min()
    return max_value + min_value - df[column_name]



'''
sheet_daily['美国首次申领失业金人数/4WMA_逆序'] = reverse_column(sheet_daily, '美国首次申领失业金人数/4WMA')
'''



'''
def plot_scatter_with_fit(df, main_col, compare_cols, start_date=None, end_date=None):
    """
    绘制主列与多个列的散点图及线性拟合线，评估线性关系。
    
    参数:
        df (DataFrame): 输入数据。
        main_col (str): 主列名。
        compare_cols (list): 要对比的列名列表。
        start_date (str): 开始日期（可选），格式 'YYYY-MM-DD'。
        end_date (str): 结束日期（可选），格式 'YYYY-MM-DD'。
    """
    # 过滤日期范围
    if start_date:
        df = df[df['Date'] >= start_date]
    if end_date:
        df = df[df['Date'] <= end_date]
    
    # 检查主列是否存在
    if main_col not in df.columns:
        print(f"主列 '{main_col}' 不存在于 DataFrame 中。")
        return
    
    # 绘制主列与多个对比列的散点图和拟合直线
    for col in compare_cols:
        if col in df.columns:
            # 提取主列和对比列数据
            x = df[main_col]
            y = df[col]
            
            # 计算线性回归
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            line = slope * x + intercept  # 拟合直线公式
            
            # 绘制散点图和拟合直线
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, alpha=0.7, edgecolors='k', label='Data Points')
            plt.plot(x, line, color='r', linestyle='--', label=f'Fit Line (R^2={r_value**2:.2f})')
            
            # 添加标题和标签
            plt.title(f'{main_col} vs {col}', fontsize=16)
            plt.xlabel(main_col, fontsize=12)
            plt.ylabel(col, fontsize=12)
            
            # 图例和网格
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"列 '{col}' 不存在于 DataFrame 中。")
'''


def plot_scatter_with_fit(df, main_col, compare_cols, start_date=None, end_date=None):
    """
    绘制主列与多个列的散点图及线性拟合线，剔除超级异常值（基于 3 倍 IQR），评估线性关系。
    
    参数:
        df (DataFrame): 输入数据。
        main_col (str): 主列名。
        compare_cols (list): 要对比的列名列表。
        start_date (str): 开始日期（可选），格式 'YYYY-MM-DD'。
        end_date (str): 结束日期（可选），格式 'YYYY-MM-DD'。
    """
    # 过滤日期范围
    if start_date:
        df = df[df['Date'] >= start_date]
    if end_date:
        df = df[df['Date'] <= end_date]
    
    # 检查主列是否存在
    if main_col not in df.columns:
        print(f"主列 '{main_col}' 不存在于 DataFrame 中。")
        return
    
    # 内部函数：剔除异常值
    def remove_outliers(series, threshold=3.0):
        """
        基于 IQR 剔除异常值。
        
        参数:
            series (Series): 输入数据。
            threshold (float): IQR 倍数阈值（默认 3 倍）。
        
        返回:
            Series: 剔除异常值后的数据。
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return series[(series >= lower_bound) & (series <= upper_bound)]
    
    # 绘制主列与多个对比列的散点图和拟合直线
    for col in compare_cols:
        if col in df.columns:
            # 提取主列和对比列数据，并剔除缺失值
            valid_data = df[[main_col, col]].dropna()
            x = valid_data[main_col]
            y = valid_data[col]
            
            # 剔除异常值
            x_clean = remove_outliers(x, threshold=3.0)
            y_clean = remove_outliers(y, threshold=3.0)
            clean_data = valid_data[x.index.isin(x_clean.index) & y.index.isin(y_clean.index)]
            
            # 检查数据点是否足够
            if len(clean_data) < 2:
                print(f"列 '{col}' 数据不足，无法绘制拟合线。")
                continue
            
            x = clean_data[main_col]
            y = clean_data[col]
            
            # 计算线性回归
            slope, intercept, r_value, _, _ = linregress(x, y)
            line = slope * x + intercept  # 拟合直线公式
            
            # 绘制散点图和拟合直线
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, alpha=0.7, edgecolors='k', label='Data Points')
            plt.plot(x, line, color='r', linestyle='--', label=f'Fit Line (R^2={r_value**2:.2f})')
            
            # 添加标题和标签
            plt.title(f'{main_col} vs {col}', fontsize=16)
            plt.xlabel(main_col, fontsize=12)
            plt.ylabel(col, fontsize=12)
            
            # 图例和网格
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"列 '{col}' 不存在于 DataFrame 中。")


'''
plot_scatter_with_fit(df, 
                        main_col='主指标', 
                        compare_cols=['对比指标1', '对比指标2'], 
                        start_date='2022-01-10', 
                        end_date='2022-03-10')
'''


def plot_feature_importance(booster, X_train, importance_type='weight', title='特征重要性排序', xlabel='特征重要性'):
    """
    绘制特征重要性的排序图
    :param booster: xgboost 模型的 booster
    :param X_train: 训练数据，用于映射特征名称
    :param importance_type: 'weight', 'gain' 或 'cover' 用于获取特征重要性
    :param title: 图表的标题 (可选)
    :param xlabel: X 轴标签 (可选)
    """
    # 获取特征重要性
    feature_importance = booster.get_score(importance_type=importance_type)

    # 创建 DataFrame 用于排序
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    })

    # 将特征名称从 f0, f1 等映射到实际的列名
    feature_names = dict(zip([f'f{i}' for i in range(len(X_train.columns))], X_train.columns))
    importance_df['feature_name'] = importance_df['feature'].map(feature_names)

    # 按重要性降序排序
    importance_df_sorted = importance_df.sort_values('importance', ascending=True)
    # 绘制水平条形图
    plt.figure(figsize=(8, 6))  # 设置适中的图形大小
    plt.barh(range(len(importance_df_sorted)), importance_df_sorted['importance'])
    plt.yticks(range(len(importance_df_sorted)), importance_df_sorted['feature_name'], fontsize=8)  # 调整字体大小
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()  # 调整布局，避免标签重叠
    plt.show()


# 使用示例：
'''
plot_feature_importance(xgb_model.get_booster(), X_train, importance_type='gain', title='特征重要性 (Gain)', xlabel='增益')
plot_feature_importance(xgb_model.get_booster(), X_train, importance_type='weight', title='特征重要性 (Weight)', xlabel='权重')
plot_feature_importance(xgb_model.get_booster(), X_train, importance_type='cover', title='特征重要性 (Cover)', xlabel='覆盖率')
'''

    
def plot_feature_distribution(df, feature_columns, bins=30, figsize=(18, 12)):
    """
    根据传入的 DataFrame 和指定的特征列列表，绘制各个特征的分布图（直方图+核密度估计曲线），
    以便观察数据的分布情况。

    参数：
        df: pandas.DataFrame
            包含数据的 DataFrame。
        feature_columns: list
            要展示分布的特征列名称列表。
        bins: int
            直方图的柱子数量，默认采用 30 个柱子。
        figsize: tuple
            整体图形的尺寸，默认为 (18, 12)。

    返回：
        None，函数会直接展示绘制的图形。
    """

    n_features = len(feature_columns)
    n_cols = 3  # 每行放置3个图
    n_rows = math.ceil(n_features / n_cols)
    
    plt.figure(figsize=figsize)
    for idx, col in enumerate(feature_columns):
        plt.subplot(n_rows, n_cols, idx + 1)
        # 去除空值后绘制直方图，并绘制核密度估计曲线
        data = df[col].dropna()
        sns.histplot(data, bins=bins, kde=True)
        plt.title(f"{col} 分布")
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()


# 领先性
# 计算密歇根大学消费者信心指数的shift值，使得值为52.2时落在sheet的最后一天
# 找到密歇根大学消费者信心指数为52.2的日期
# 定义一个函数来处理因子的shift值计算
def calculate_shift_value(sheet, column_name, target_value, last_day_index=pd.Timestamp('2025-05-30'), default_shift=6):
    # 检查last_day_index是否为周末，如果是则顺延至下周一
    if last_day_index.weekday() >= 5:  # 5是周六，6是周日
        # 计算需要增加的天数，使日期变为下周一
        days_to_add = 7 - last_day_index.weekday() if last_day_index.weekday() == 6 else 2
        last_day_index = last_day_index + pd.Timedelta(days=days_to_add)
        print(f"输入日期为周末，已顺延至下周一: {last_day_index.strftime('%Y-%m-%d')}")

    target_value_indices = sheet.index[sheet[column_name] == target_value]
    if not target_value_indices.empty:
        # 只在2025年4月的数据中查找目标值
        import datetime
        current_date = datetime.datetime.now()
        target_value_indices = target_value_indices[target_value_indices.map(lambda x: x.year == current_date.year and x.month == current_date.month)]
        if target_value_indices.empty:
            print(f"在2025年4月没有找到{column_name}为{target_value}的记录")
            return default_shift
        target_date = target_value_indices[0]
        # 获取两个日期在DataFrame中的位置索引
        target_position = sheet.index.get_loc(target_date)
        last_position = sheet.index.get_loc(last_day_index)
        # 计算行差
        shift_value = last_position - target_position
        print(f"{column_name}计算得到的shift值为: {shift_value}")
        return int(shift_value)
    else:
        print(f"未找到{column_name}为{target_value}的记录，使用默认shift值{default_shift}")
        return default_shift
    



def plot_xgb_feature_importance(model, X_train, title='XGBoost多维度特征重要性对比（标准化后）'):
    """
    绘制XGBoost多维度特征重要性对比图，并附带中文业务含义表格

    参数：
    - model: 已训练好的xgboost.Booster模型
    - X_train: 训练集DataFrame，用于映射特征名
    - title: 图表标题（可选）
    """
    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

    # 提取各类重要性指标
    df_importance = pd.DataFrame()
    for imp_type in importance_types:
        scores = model.get_score(importance_type=imp_type)
        feature_names = dict(zip([f'f{i}' for i in range(len(X_train.columns))], X_train.columns))
        scores_with_names = {feature_names.get(k, k): v for k, v in scores.items()}
        temp_df = pd.DataFrame.from_dict(scores_with_names, orient='index', columns=[imp_type])
        df_importance = df_importance.join(temp_df, how='outer') if not df_importance.empty else temp_df

    df_importance = df_importance.fillna(0)
    df_importance = df_importance.sort_values(by='weight', ascending=False)

    # 归一化
    df_importance_norm = df_importance.copy()
    for col in importance_types:
        df_importance_norm[col] = df_importance_norm[col] / df_importance_norm[col].max()

    df_importance = df_importance_norm

    # 绘图
    plt.figure(figsize=(14, 15))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    color_map = dict(zip(importance_types, colors))

    y_pos = np.arange(len(df_importance))
    width = 0.13

    for i, imp_type in enumerate(importance_types):
        plt.barh(y_pos + i * width, df_importance[imp_type],
                 width, label=imp_type, color=color_map[imp_type], alpha=0.85)

    plt.yticks(y_pos + width * 2, df_importance.index, fontsize=12)
    plt.title(title, fontsize=16)
    plt.xlabel('标准化重要性值', fontsize=13)
    plt.legend(title='重要性类型', bbox_to_anchor=(1.03, 1), loc='upper left')

    # 中文业务含义表格
    table_data = [
        ['weight', '因子在预测场景中被频繁用作分组依据的次数。高 → 该因子适合快速划分市场状态，是第一层“筛子”。'],
        ['gain', '每次用该因子分组能提升多少预测准确度。高 → 对定价预测有极强“辨别能力”，是决定性因素。'],
        ['cover', '该因子每次分组时所覆盖的数据量。高 → 适合划分普遍性特征，适用于大部分情况。'],
        ['total_gain', '累计提升了多少准确度（次数×单次提升）。高 → 综合贡献大，稳定又强势。'],
        ['total_cover', '累计覆盖了多少样本。高 → 广泛参与预测过程，缺它预测不完整。']
    ]

    table = plt.table(
        cellText=table_data,
        colLabels=None,
        cellLoc='left',
        loc='bottom',
        bbox=[-0.12, -0.8, 1.18, 0.55]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))

    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.get_text().set_fontsize(10)  # 控制字体大小，避免文字超出单元格
        cell.get_text().set_wrap(True)    # 允许自动换行（有的版本有效）
        cell.get_text().set_ha('left')
        cell.get_text().set_va('center')

    plt.tight_layout()
    plt.show()
    return df_importance

