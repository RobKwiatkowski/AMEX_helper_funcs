import pandas as pd
import numpy as np
import joblib
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from helper_functions import amex_metric


class CFG:
    input_dir = 'outputs/final_dfs/'
    seed = 42
    n_folds = 4
    target = 'target'


print("Reading input data...")
train_df = pd.read_parquet(CFG.input_dir + 'train_df.parquet')
print("Training data read.")
test_df = pd.read_parquet(CFG.input_dir + 'test_df.parquet')
print("Testing data read.")

model = CatBoostClassifier(iterations=50000,
                           learning_rate=0.06,
                           depth=10,
                           loss_function='CrossEntropy',
                           random_seed=CFG.seed,
                           task_type='GPU',
                           devices='0:1')

train_data = train_df.drop('target', axis=1)
train_labels = train_df['target']

features = [col for col in train_data.columns if col not in ['customer_ID', CFG.target]]

# Create a numpy array to store test predictions from every fold
test_predictions = np.zeros(len(test_df))
# Create a numpy array to store out of folds predictions
oof_predictions = np.zeros(len(train_df))

kfold = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
print('Running CatBoost CV model')
for fold, (trn_ind, val_ind) in enumerate(kfold.split(train_data, train_labels)):
    print(f'Fold {fold}')
    x_train, x_val = train_data[features].iloc[trn_ind], train_data[features].iloc[val_ind]
    y_train, y_val = train_labels.iloc[trn_ind], train_labels.iloc[val_ind]

    eval_dataset = Pool(x_val, y_val)

    model.fit(x_train,
              y_train,
              # cat_features=cat_features,
              eval_set=eval_dataset,
              verbose=False,
              early_stopping_rounds=2000,
              plot=False)

    # Save a current model
    # joblib.dump(model, f'outputs/model_fold_{fold}_seed{CFG.seed}.pkl')
    # Predict validation
    val_pred = model.predict(x_val)
    # Add predictions to the out_of_folds array
    oof_predictions[val_ind] = val_pred
    # Compute fold metric
    score = amex_metric(y_val, val_pred)
    print(f'Fold {fold} CV score is: {score}')
    gc.collect()

# Compute out of folds metric
score = amex_metric(train_df[CFG.target], oof_predictions)
print(f'Out of the folds CV score is: {score}')

# Create a dataframe to store out of folds predictions
oof_df = pd.DataFrame({'customer_ID': train_df.index, 'target': train_df[CFG.target], 'prediction': oof_predictions})
oof_df.to_csv(f'outputs/oof_catboost_{CFG.n_folds}fold_seed{CFG.seed}.csv', index=False)

# Create a dataframe to store test prediction
test_predictions = model.predict(test_df[features])
test_pred = pd.DataFrame({'customer_ID': test_df.index, 'prediction': test_predictions})
test_pred.to_csv(f'outputs/test_catboost_{CFG.n_folds}fold_seed{CFG.seed}.csv', index=False)
