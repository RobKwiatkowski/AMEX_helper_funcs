import pandas as pd
import numpy as np
import gc

from multiprocessing import Pool, cpu_count
from itertools import repeat


def read_data(path: str, train: bool = True, sample: bool = False, cust_ratio: int = 0.2) -> pd.DataFrame:
    """ Reads raw data and prepares the panda's DataFrame, samples data if required

    args:
        path: path to the file with raw data
        train: True if to read a train file
        sample: True if to draw a sample
        cust_ratio: a ratio of customers to be sampled from the raw data

    return:
        df: panda's Dataframe
    """

    df = pd.read_parquet(path)
    if sample:
        n_customers = df['customer_ID'].nunique()
        sampled_no_of_cust = int(n_customers * cust_ratio)
        cust_ids = np.random.choice(df['customer_ID'].unique(), sampled_no_of_cust)
        df = df[df['customer_ID'].isin(cust_ids)]
        print(f'Customers in sampled database: {sampled_no_of_cust}')
        print(f'Rows in sampled database: {df.shape[0]}')
    else:
        if train:
            print('Using the entire training database.')
        else:
            print('Using the entire testing database.')
    print(f"Database shape: {df.shape}")

    return df


def prepare_chunks_cust(df: pd.DataFrame, columns: list, n_chunks: int = 12) -> list:
    """ Prepares chunks for multiprocessing grouped by customers
    args:
        df: pandas dataframe containing customer ID's
        columns: list of columns to be used
        n_chunks: number of chunks to be generated

    return: list of pandas dataframes
    """
    cust_unique_ids = df['customer_ID'].unique()
    cust_ids_split = np.array_split(cust_unique_ids, n_chunks)

    ready_chunks = []
    for cust_ids in cust_ids_split:
        subset = df[df['customer_ID'].isin(cust_ids)][columns]
        ready_chunks.append(subset)

    return ready_chunks


def _ewmt(chunk: pd.DataFrame, periods: list) -> pd.DataFrame:
    """ Calculates Exponential Weighted Mean for a chunk

    Args:
        chunk: pandas database
        periods: list, periods of halflife value

    Returns: pandas dataframe
    """
    results = []
    cust_ids = chunk['customer_ID']
    for hl in periods:
        chunk = chunk.ewm(halflife=hl).mean()
        chunk = chunk.add_suffix(f'ewm{hl}')
        ids_chunk = pd.concat([cust_ids, chunk], axis=1)
        results.append(ids_chunk.set_index('customer_ID'))  # change to concat()
    df = pd.concat(results, axis=1)

    return df


def calc_ewm(chunks: list, periods: tuple = (2, 4)) -> pd.DataFrame:
    """ Calculates EWM

    Args:
        chunks: list containing pandas dataframes
        periods: list containing periods for EWM

    Returns: pandas dataframe
    """
    ewm_results = []
    p1 = Pool(cpu_count())
    ewm_results.append(p1.starmap(_ewmt, zip(chunks, repeat(periods))))
    p1.close()
    p1.join()

    gc.collect()
    final = pd.concat(ewm_results[0]).reset_index()
    del ewm_results
    final = final.groupby("customer_ID").agg(['mean', 'std', 'min', 'max', 'last'])
    final.columns = ['_'.join(x) for x in final.columns]

    return final


def _cat_stat(df: pd.DataFrame, cat_features: list, stats: list = ('count', 'first', 'nunique')) -> pd.DataFrame:
    """ Calculates categorical statistics for a chunk

    Args:
        df: pandas dataframe
        cat_features: list of categorical columns
        stats: stats to be calculated

    Returns: pandas dataframe with statistics
    """
    data_cat_agg = df.groupby("customer_ID")[cat_features].agg(stats)
    data_cat_agg.columns = ['_'.join(x) for x in data_cat_agg.columns]
    return data_cat_agg


def calc_categorical_stats(chunks: list) -> pd.DataFrame:
    """ Calculates categorical statistics for all chunks

    Args:
        chunks: list of pandas dataframe

    Returns: pandas dataframe with calculated statistics
    """
    p2 = Pool(cpu_count())
    results = p2.map(_cat_stat, chunks)
    p2.close()
    p2.join()

    results = pd.concat(results)
    return results


def prepare_date_features(df: pd.DataFrame) -> pd.DataFrame:
    def _take_first_col(series): return series.values[0]

    def _last_2(series): return series.values[-2] if len(series.values) >= 2 else -127

    def _last_3(series): return series.values[-3] if len(series.values) >= 3 else -127

    # Converting S_2 column to datetime column
    df['S_2'] = pd.to_datetime(df['S_2'])

    # How many rows of records does each customer has?
    df['rec_len_date'] = df.loc[:, ['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('count')

    # Encode the 1st statement and the last statement time
    df['S_2_first'] = df.loc[:, ['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('min')
    df['S_2_last'] = df.loc[:, ['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('max')

    # For how long(days) the customer is receiving the statements
    df['S_2_period'] = (df[['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('max') -
                        df[['customer_ID', 'S_2']].groupby(by=['customer_ID'])['S_2'].transform('min')).dt.days

    # Days Between 2 statements
    df['days_between_statements'] = \
        df[['customer_ID', 'S_2']].sort_values(by=['customer_ID', 'S_2']).groupby(by=['customer_ID'])['S_2'].transform(
            'diff').dt.days
    df['days_between_statements'] = df['days_between_statements'].fillna(0)
    df['days_between_statements_mean'] = df[['customer_ID', 'days_between_statements']].sort_values(
        by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('mean')
    df['days_between_statements_std'] = df[['customer_ID', 'days_between_statements']].sort_values(
        by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('std')
    df['days_between_statements_max'] = df[['customer_ID', 'days_between_statements']].sort_values(
        by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('max')
    df['days_between_statements_min'] = df[['customer_ID', 'days_between_statements']].sort_values(
        by=['customer_ID', 'days_between_statements']).groupby(by=['customer_ID']).transform('min')
    df['S_2'] = (df['S_2_last'] - df['S_2']).dt.days

    # Difference between S_2_last(max) and S_2_last
    df['S_2_last_diff_date'] = (df['S_2_last'].max() - df['S_2_last']).dt.days

    # Difference between S_2_first(min) and S_2_first
    df['S_2_first_diff_date'] = (df['S_2_first'].min() - df['S_2_first']).dt.days

    # Get the (day,month,year) and drop the S_2_first because we can't directly use them
    df['S_2_first_dd_date'] = df['S_2_first'].dt.day
    df['S_2_first_mm_date'] = df['S_2_first'].dt.month
    df['S_2_first_yy_date'] = df['S_2_first'].dt.year

    df['S_2_last_dd_date'] = df['S_2_last'].dt.day
    df['S_2_last_mm_date'] = df['S_2_last'].dt.month
    df['S_2_last_yy_date'] = df['S_2_last'].dt.year

    agg_df = df.groupby(by=['customer_ID']).agg({'S_2': ['last', _last_2, _last_3],
                                                 'days_between_statements': ['last', _last_2, _last_3]})

    agg_df.columns = [i + '_' + j for i in ['S_2', 'days_between_statements'] for j in ['last', 'last_2', 'last_3']]
    df = df.groupby(by=['customer_ID']).agg(_take_first_col)
    df = df.merge(agg_df, how='inner', left_index=True, right_index=True)
    df = df.drop(['S_2', 'days_between_statements', 'S_2_first', 'S_2_last_x'], axis=1)

    return df


def _calc_num(chunk, stats):
    final = chunk.groupby("customer_ID").agg(stats)
    final.columns = ['_'.join(x) for x in final.columns]

    return final


def calc_numerical_stats(chunks: list, stats: list = ('min', 'max', 'mean', 'median')):
    """ Calculates categorical statistics for all chunks

    Args:
        chunks: list of pandas dataframe
        stats: aggregate statistics to be used

    Returns: pandas dataframe with calculated statistics
    """

    p3 = Pool(cpu_count())
    r = p3.starmap(_calc_num, zip(chunks, repeat(stats)))
    p3.close()
    p3.join()
    r = pd.concat(r)

    return r


def sum_common_cols(df: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    """ Calculates statistics of columns belonging to the same category
    Args:
        df: Pandas Dataframe with raw data
        cat_features: list of categorical columns

    Returns: Pandas DataFrame with calculated statistics per type of columns
    """
    # filter out all numeric data
    num_cols = [c for c in df.columns if c not in cat_features + ['customer_ID']]

    del_cols = [c for c in num_cols if (c.startswith("D"))]
    pay_cols = [c for c in num_cols if (c.startswith("P"))]
    bal_cols = [c for c in num_cols if (c.startswith("B"))]
    ris_cols = [c for c in num_cols if (c.startswith("R"))]
    spe_cols = [c for c in num_cols if (c.startswith("S"))]

    df['balance_sum'] = df[[i for i in bal_cols]].sum(numeric_only=True, axis=1)
    df['delinquent_sum'] = df[[i for i in del_cols]].sum(numeric_only=True, axis=1)
    df['spend_sum'] = df[[i for i in spe_cols]].sum(numeric_only=True, axis=1)
    df['payment_sum'] = df[[i for i in pay_cols]].sum(numeric_only=True, axis=1)
    df['risk_sum'] = df[[i for i in ris_cols]].sum(numeric_only=True, axis=1)

    return df[['customer_ID', 'balance_sum', 'delinquent_sum',
               'spend_sum', 'payment_sum', 'risk_sum']].groupby('customer_ID').sum()


def nans_per_cust(df):
    nan_df = df.drop('customer_ID', axis=1).isna().groupby(df['customer_ID'], sort=False).sum().add_suffix('_col')
    nan_df.reset_index(inplace=True)
    nan_df.set_index('customer_ID', inplace=True)
    df2 = nan_df.sum(axis=1)
    nan_by_customer = pd.DataFrame(df2)
    nan_by_customer = nan_by_customer.rename(columns={0: 'by_cust_ID_NAN'})
    return nan_by_customer


def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])
    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1] / gini[0] + top_four)


def ab_features(df):
    df.reset_index(inplace=True, drop=False)
    df = df.rename(columns={'index': 'row_id'})
    for b_col in [f'B_{i}' for i in [11, 14, 17]] + ['D_39', 'D_131'] + [f'S_{i}' for i in [16, 23]]:
        for p_col in ['P_2', 'P_3']:
            if b_col in df.columns:
                df[f'{b_col}-{p_col}' + '_AB'] = df[b_col] - df[p_col]
    df = df.sort_values('row_id')
    df = df.drop(['row_id'], axis=1)
    return df


def calculate_correlations(df):
    return 0

