if __name__ == '__main__':
    import glob
    import os
    import gc
    import numpy as np
    import pandas as pd
    import helper_functions as hf
    from create_final_df import create_final, check_readiness

    class CFG:
        train = False
        sample = False

    if CFG.train:
        path = 'outputs/final_dfs/train_df.parquet'
    else:
        path = 'outputs/final_dfs/test_df.parquet'


    def ewm(cols_to_use, parallel=True, periods=(2, 4)):
        ewm_cols = [c for c in cols_to_use if c in num_features]
        print(f'Number of columns for EWMs: {len(ewm_cols)}')
        if parallel:
            for i, split_cols in enumerate(np.array_split(ewm_cols, 4)):
                split_cols = list(split_cols)
                split_cols.insert(0, 'customer_ID')  # customer_ID should always be in a chunk
                chunks = hf.prepare_chunks_cust(data_raw, split_cols, n_chunks=6)
                df_ewms = hf.calc_ewm(chunks, periods=periods)
                df_ewms.reset_index(inplace=True)
                print(f'writing EWMs{i} data...')
                df_ewms.to_parquet(f'outputs/df_ewm{i}.parquet')
        else:
            cols_to_use.insert(0, 'customer_ID')
            results = hf.calc_ewm(cols_to_use, periods=periods)
            results.to_parquet(f'outputs/df_ewm.parquet')

        return 0


    pd.options.display.width = None
    pd.options.display.max_columns = 15

    # cleaning outputs folder
    files = glob.glob('outputs/*.*')
    nl = '\n'
    print(f'Files to be removed:{nl}{files}')
    for f in files:
        os.remove(f)

    # reading raw data
    data_raw = hf.read_data(path, train=CFG.train, sample=CFG.sample, cust_ratio=0.01)

    # dropping some columns
    cols_to_drop = ['D_88', 'D_110', 'B_39', 'D_73', 'B_42', 'D_88', 'D_77', 'D_139', 'D_141', 'D_143', 'D_110', 'B_1']
    data_raw.drop(cols_to_drop, axis=1, inplace=True)

    # dumping customer_IDs to a reference file
    c_ids = data_raw['customer_ID'].unique()
    pd.DataFrame(c_ids).to_csv('outputs/cust_ids.csv', index=False)

    # creating lists of various column types
    cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_66', 'D_68', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126']
    num_features = [c for c in data_raw.columns if c not in cat_features+['customer_ID']]

    payment_cols = [col for col in list(data_raw.columns) if 'P_' in col]
    delinquency_cols = [col for col in list(data_raw.columns) if 'D_' in col]
    spend_cols = [col for col in list(data_raw.columns) if 'S_' in col]
    balance_cols = [col for col in list(data_raw.columns) if 'B_' in col]
    risk_cols = [col for col in list(data_raw.columns) if 'R_' in col]

    # EWM
    # ewm(spend_cols)

    # datetime stats
    df = data_raw.loc[:, ['customer_ID', 'S_2']]
    df_date = hf.prepare_date_features(df)
    print('Writing datetime stats...')
    df_date.to_csv('outputs/df_date.csv')
    del df_date
    gc.collect()

    # categorical stats
    chunks_to_process = hf.prepare_chunks_cust(data_raw, ['customer_ID']+cat_features)
    df_cats = hf.calc_categorical_stats(chunks_to_process)
    print('Writing categorical stats...')
    df_cats.to_csv('outputs/df_cats.csv')
    del df_cats
    gc.collect()

    # numerical stats
    chunks_to_use = hf.prepare_chunks_cust(data_raw, ['customer_ID']+num_features)
    df_nums = hf.calc_numerical_stats(chunks_to_use, stats=['min', 'max', 'mean'])
    print('Writing numerical stats...')
    df_nums.to_csv('outputs/df_nums.csv')
    del df_nums
    gc.collect()

    # columns summation
    df_sums = hf.sum_common_cols(data_raw)
    print('Writing columns summations...')
    df_sums.to_csv('outputs/df_sums.csv')
    del df_sums
    gc.collect()

    create_final(train=CFG.train)
    check_readiness()
