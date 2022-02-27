import pandas as pd 

def get_train_test(df,iCV, average_test):
    
    num_categs = df['y'].nunique()

    df_train_tmp = df[df['CV']!=iCV]
    X_train = df_train_tmp.loc[:,df.columns.drop(['y','CV'])]
    y_train = df_train_tmp.loc[:,'y']
    
    df_test_tmp = df[df['CV']==iCV]
    X_test = df_test_tmp.loc[:,df.columns.drop(['y','CV'])]
    y_test = df_test_tmp.loc[:,'y']
    
    if average_test == True:
        averaged_X_test = pd.DataFrame(columns=X_test.columns)
        averaged_y_test = []
        for i_cat in range(num_categs):
            averaged_X_test = pd.concat((averaged_X_test,X_test.loc[y_test==i_cat,:].mean().to_frame().T),axis=0)
            averaged_y_test.append(i_cat)
        
        X_test= averaged_X_test
        y_test= averaged_y_test
        del averaged_y_test; del averaged_X_test;
        
    return X_train, y_train, X_test, y_test