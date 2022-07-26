import pandas as pd

from src.helper_functions import prepare_chunks_cust


def test_prepare_chunks_cust():
    test_df = pd.DataFrame({'customer_ID': ['a', 'a', 'a', 'b', 'b', 'c', 'd'],
                            'values_1': [1, 2, 3, 4, 5, 6, 7]})

    results_1 = prepare_chunks_cust(test_df, columns=['customer_ID', 'values_1'], n_chunks=2)
    expected = 2
    assert len(results_1) == expected
