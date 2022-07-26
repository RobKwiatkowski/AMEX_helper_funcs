import pandas as pd
import numpy as np
import gc
from multiprocessing import Pool, cpu_count
from itertools import repeat


def read_data(path, train=True, sample=False, cust_ratio=0.2):
    """
    args:
        path: string
        train: bool, True if to read a train file
        sample: bool,  True if to draw a sample
        cust_ratio: float,  a ratio of customers to be sampled
    """

    df = pd.read_parquet(path)
    if sample:
        n_customers = df['customer_ID'].nunique()
        no_of_cust = int(n_customers * cust_ratio)
        cust_ids = np.random.choice(df['customer_ID'].unique(), no_of_cust)
        df = df[df['customer_ID'].isin(cust_ids)]
        print(f'Customers in sampled database: {no_of_cust}')
        print(f'Rows in sampled database: {df.shape[0]}')
    else:
        if train:
            print('Using the entire training database.')
        else:
            print('Using the entire testing database.')
    print(f"Database shape: {df.shape}")

    return df


def prepare_chunks_cust(df, columns, n_chunks=12):
    """
    Prepares chunks by customers
    args:
        df: pandas dataframe
        columns: columns to be used
        n_chunks: number of chunks to be generated

    :return: list of pandas dataframes
    """
    cust_unique_ids = df['customer_ID'].unique()
    cust_ids_split = np.array_split(cust_unique_ids, n_chunks)
    ready_chunks = []

    for c_ids in cust_ids_split:
        sub = df[df['customer_ID'].isin(c_ids)][columns]
        ready_chunks.append(sub)
    return ready_chunks


def _ewmt(chunk, periods):
    """
    Calculates EWM for a chunk
    Args:
        chunk: pandas database
        periods: list, periods of halflife value

    Returns: pandas dataframe

    """
    results = []
    cust_ids = chunk['customer_ID']
    for t in periods:
        chunk = chunk.ewm(halflife=t).mean()
        chunk = chunk.add_suffix(f'ewm{t}')
        ids_chunk = pd.concat([cust_ids, chunk], axis=1)
        results.append(ids_chunk.set_index('customer_ID'))
    df = pd.concat(results, axis=1)
    return df


def calc_ewm(chunks, periods=(2, 4)):
    """
    Calculates EWM
    Args:
        chunks: list, contains pandas dataframes
        periods: list, contains periods for EWM

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


def _cat_stat(df):
    """
    Calculates categorical statistics for a chunk
    Args:
        df: pandas dataframe

    Returns: pandas dataframe with statistics

    """
    cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
    data_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'first', 'last', 'nunique'])
    data_cat_agg.columns = ['_'.join(x) for x in data_cat_agg.columns]
    return data_cat_agg


def calc_categorical_stats(chunks):
    """
    Calculates categorical statistics for all chunks
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


def prepare_date_features(df):
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


def calc_numerical_stats(chunks, stats=('min', 'max', 'mean')):
    """
    Calculates categorical statistics for all chunks
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


def sum_common_cols(df):
    """

    Args:
        df: Pandas Dataframe

    Returns:

    """
    cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
    num_cols = [c for c in df.columns if c not in cat_features + ['customer_ID']]
    del_cols = [c for c in num_cols if (c.startswith("D"))]
    pay_cols = [c for c in num_cols if (c.startswith("P"))]
    bal_cols = [c for c in num_cols if (c.startswith("B"))]
    ris_cols = [c for c in num_cols if (c.startswith("R"))]
    spe_cols = [c for c in num_cols if (c.startswith("S"))]
    "A function calculating all numerics per category for each customer"
    df['balance_sum'] = df[[i for i in bal_cols]].sum(axis=1)
    df['delinquent_sum'] = df[[i for i in del_cols]].sum(axis=1)
    df['spend_sum'] = df[[i for i in spe_cols]].sum(axis=1)
    df['payment_sum'] = df[[i for i in pay_cols]].sum(axis=1)
    df['risk_sum'] = df[[i for i in ris_cols]].sum(axis=1)
    return df[['customer_ID', 'balance_sum', 'delinquent_sum',
               'spend_sum', 'payment_sum', 'risk_sum']].groupby('customer_ID').sum()


# def round():
#     num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
#     num_cols = [col for col in num_cols if 'last' in col]
#     for col in num_cols:
#         train[col + '_round2'] = train[col].round(2)
#         test[col + '_round2'] = test[col].round(2)
#
#
# def get_differences():
#     num_cols = [col for col in train.columns if 'last' in col]
#     num_cols = [col[:-5] for col in num_cols if 'round' not in col]
#     for col in num_cols:
#         try:
#             train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
#             test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
#         except:
#             pass
#
# def transform_floats():
#     num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
#     for col in tqdm(num_cols):
#         train[col] = train[col].astype(np.float16)
#         test[col] = test[col].astype(np.float16)