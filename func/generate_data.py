import pandas as pd
from sklearn.datasets import make_blobs
import numpy as np

def generate_data(num_runs,num_reps, num_categs, num_features, cluster_std):

    total_num_trials = num_runs * num_reps * num_categs

    X,y = make_blobs(n_samples=total_num_trials, n_features=num_features, centers=num_categs, cluster_std=cluster_std, random_state=0)

    df = pd.concat(
        (pd.DataFrame(X,columns=[f'feature_{i+1}' for i in range(X.shape[1])]),
        pd.DataFrame(y,columns=['y'])),
        axis=1)

    # Create a cross validation column
    df['CV'] = np.nan
    for i_cat in range(num_categs):
        df.loc[y==i_cat,'CV'] = range(num_runs * num_reps)

    df['CV'] = np.floor(np.divide(df['CV'],num_reps))+1

    return df
