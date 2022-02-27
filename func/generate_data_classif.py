import pandas as pd
from sklearn.datasets import make_classification
import numpy as np

def generate_data_classif(num_runs,num_reps, num_categs, num_features,num_informative,class_separation):

    total_num_trials = num_runs * num_reps * num_categs
    print(f'total number of trials is: {total_num_trials}')
    X,y = make_classification(n_samples=total_num_trials, n_features=num_features, 
                              n_classes=num_categs, n_clusters_per_class=1, random_state=0,
                              n_informative=num_informative,n_redundant=num_features-num_informative,
                              weights=[1/num_categs]*num_categs,flip_y=0,class_sep=class_separation)
    
    df = pd.concat(
        (pd.DataFrame(X,columns=[f'feature_{i+1}' for i in range(X.shape[1])]),
        pd.DataFrame(y,columns=['y'])),
        axis=1)

    
    # Create a cross validation column
    df['CV'] = np.nan
    for i_cat in range(num_categs):
        print(np.sum(df['y']==i_cat))
        df.loc[y==i_cat,'CV'] = range(num_runs * num_reps)
        

    df['CV'] = np.floor(np.divide(df['CV'],num_reps))+1

    return df
