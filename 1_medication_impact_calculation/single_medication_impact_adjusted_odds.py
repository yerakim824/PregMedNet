import os
import sys
import warnings


def set_threads_for_external_libraries(n_threads=64):
    """
    Tries to disable BLAS or similar implicit  parallelization engines
    by setting the following environment attributes to `1`:
    - OMP_NUM_THREADS
    - OPENBLAS_NUM_THREADS
    - MKL_NUM_THREADS
    - VECLIB_MAXIMUM_THREADS
    - NUMEXPR_NUM_THREADS
    This can be useful since `numpy` and co. are using these libraries to parallelize vector and matrix operations.
    However, by default, the grab ALL CPUs an a machine. Now, if you use parallelization, e.g., based on `joblib` as in
    `sklearn.model_selection.GridSearchCV` or `sklearn.model_selection.cross_validate`, then you will overload your
    machine and the OS scheduler will spend most of it's time switching contexts instead of calculating.
    Parameters
    ----------
    n_threads: int, optional, default: 1
        Number of threads to use for
    Notes
    -----
    - This ONLY works if you import this file BEFORE `numpy` or similar libraries.
    - BLAS and co. only kick in when the data (matrices etc.) are sufficiently large.
      So you might not always see the `CPU stealing` behavior.
    - For additional info see: "2.4.7 Avoiding over-subscription of CPU ressources" in the
      `joblib` docs (https://buildmedia.readthedocs.org/media/pdf/joblib/latest/joblib.pdf).
    - Also note that there is a bug report for `joblib` not disabling BLAS and co.
      appropriately: https://github.com/joblib/joblib/issues/834
    Returns
    -------
    None
    """

    if (
            "numpy" in sys.modules.keys()
            or "scipy" in sys.modules.keys()
            or "sklearn" in sys.modules.keys()
            ):
        warnings.warn("This function should be called before `numpy` or similar modules are imported.")

    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
    print('thread is limited to {}'.format(n_threads))

set_threads_for_external_libraries(n_threads=64)


import pandas as pd
import numpy as np
import time
"""
Required Datasets:
1. CORRECTED_RAW_ODDS_2024.csv (select med-dz pairs)
2. FINAL_DATA_DEC2023_ver2.csv (main analysis)
"""
## Generate the list of med-disease pairs to run ##
print('ADJ OR NALAB2: FINAL VERSION (2024) with revision!')
print('start to generate the list of med-disease pairs to run')
or_df = pd.read_csv('CORRECTED_RAW_ODDS_2024.csv').drop(columns=['Unnamed: 0'],axis=1)
final_df = or_df.copy() 

med_dz_dict = {}
med_dz_dict['nalab2']=final_df.iloc[:289,:]
med_dz_dict['nalab3']=final_df.iloc[289:578,:]
med_dz_dict['nalab4']=final_df.iloc[578:867,:]
med_dz_dict['nalab5']=final_df.iloc[867:1156,:]
med_dz_dict['nalab6']=final_df.iloc[1156:,:]
med_dz_dict['test'] = final_df.iloc[36:38,:]

current_server = 'nalab2'
print('CURRENT SERVER: {}'.format(current_server))

med_dz_df = med_dz_dict[current_server]
med_dz_pair = med_dz_df[['Disease','Medication']]
med_dz_pair['Med_modi'] = med_dz_pair['Medication'].apply(lambda x: x.replace(',','_').replace(' ','_').replace('-','_').replace(')','_').replace('(','_'))

rename_med_name = {}
reverse_med_name = {}
for old, new in zip(med_dz_pair['Medication'],med_dz_pair['Med_modi']):
    rename_med_name[old]=new
    reverse_med_name[new]=old

## Load the main data/Data Preprocessing ##
print('start to load the main data/data preprocessing')
from sklearn.preprocessing import MinMaxScaler
from collinearity import SelectNonCollinear
from numpy.linalg import inv, det
from category_encoders.binary import BinaryEncoder
df = pd.read_csv('FINAL_DATA_DEC2023_ver2.csv')
df = df.set_index('ENROLID_BABY')
### 1. Binary Encoding
df_encoder_test =  df.copy()
encoder = BinaryEncoder(cols=["EGEOLOC"], drop_invariant=True)
encoder.fit(df_encoder_test)
df_encoder = encoder.transform(df_encoder_test)
df = df_encoder.copy()
### 2. Scale AGE_MOM and GA
baby_dz_list = [i for i in df.columns if i.__contains__('_Baby')]
mom_dz_list = [i for i in df.columns if i.__contains__('_Mom')]
others_df = df.loc[:,:'AGE_MOM']
baby_dz = df[baby_dz_list]
mom_dz = df[mom_dz_list]
mom_med = df.loc[:,'5-methyltetrahydrofolic acid, Oral':]
norm_other_df = others_df.copy()
norm_other_df = norm_other_df.reset_index()
scaler = MinMaxScaler()
scaler.fit(norm_other_df[['GESTATIONAL_AGE']])
norm_other_df[['GESTATIONAL_AGE']]=pd.DataFrame(scaler.transform(norm_other_df[['GESTATIONAL_AGE']]),columns=['GESTATIONAL_AGE'])
scaler.fit(norm_other_df[['AGE_MOM']])
norm_other_df[['AGE_MOM']]=pd.DataFrame(scaler.transform(norm_other_df[['AGE_MOM']]),columns=['AGE_MOM'])
norm_other_df = norm_other_df.set_index('ENROLID_BABY')
confounders = pd.concat([norm_other_df,mom_dz],axis=1)
### 3. Drop null columns and amphetamine based on VIF
confounders_corr = confounders.corr().abs()
nan_col = list(confounders_corr[confounders_corr['GESTATIONAL_AGE'].isna()].index)
confounders = confounders.drop(columns=nan_col,axis=1)
confounders = confounders.drop(columns=['REGION','SUD_Psystimul_Amphetamine_Mom']) ##'EGEOLOC',
odds_df = pd.concat([baby_dz,confounders,mom_med],axis=1)
renamed_or_df = odds_df.rename(columns=rename_med_name)

## Prepare Analysis: confounder list ##
print('prepare alpha and conf_list')
conf_list=""
for i in confounders.columns:
    if conf_list == "":
        conf_list = conf_list + i
    elif i in ['REGION','EGEOLOC']:
        conf_list = conf_list+ " + C(" +i +")"
    else:
        conf_list = conf_list+" + "+i
num_conf = len(confounders.columns)
alpha_dim = num_conf

####### Start Analysis #######
print('start the main analysis')
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import optuna
import time
import logging
start = time.time()

store_dict = {}
ix =0
for dz, med in zip(med_dz_pair['Disease'],med_dz_pair['Med_modi']):
    conf_col =list(confounders.columns)
    feature_list = [dz]+conf_col+[med]
    sel_data = renamed_or_df[feature_list]
    
    X = sel_data.drop(dz, axis=1)  # Features
    y = sel_data[dz]  # Target variable
    
    ## Hyperparameter Tuning to determine the best shrinkage coefficient ##
    def objective(trial):
        shrink_coeff = trial.suggest_float('shrink_coeff', 0.01, 10, log=True)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        f1_scores = []
        print('Start to Calculate -- Shrink Coefficinet: {}'.format(shrink_coeff))
        for train_index, test_index in skf.split(X,y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            train_data = X_train.copy()
            train_data[dz] = y_train
            test_data = X_test.copy()
            test_data[dz] = y_test

            stats_model_input = dz+' ~ {} + {}'.format(conf_list,med)
            shrink_coeff_list= np.full(alpha_dim+1,shrink_coeff)
            shrink_coeff_list = np.append(shrink_coeff_list,0)
                    
            try:
                logit_model = smf.logit(stats_model_input, data=train_data)
                result = logit_model.fit_regularized(method='l1', alpha=shrink_coeff_list, maxiter=100, 
                                                    trim_mode='auto', full_output=True, disp=True, maxfun=100)
                predictions = result.predict(X_test)
                binary_predictions = (predictions >= 0.5).astype(int)
                f1 = f1_score(y_test, binary_predictions)
                f1_scores.append(f1)
                print('Non-singular matrix -- conducted calculation!')
            except Exception as e:
                logging.error("Error occurred with shrink_coeff %s: %s", shrink_coeff, str(e))
                pass

        print('f1_scores:',f1_scores)
        average_f1_score = np.mean(f1_scores)
        print(average_f1_score)
        return average_f1_score
    if __name__ == "__main__":
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10, n_jobs=-1) 
    
    try:    
        best_params = study.best_params
        best_shrink_coeff = best_params['shrink_coeff']
        print("Best shrink coefficient {}-{}:".format(dz,med), best_shrink_coeff)
    except:
        best_shrink_coeff = 1
        print("Hyperparameter tuning failed {}-{}:".format(dz,med), best_shrink_coeff)
    
    data = sel_data.copy()
    stats_model_input = dz+' ~ {} + {}'.format(conf_list,med)
    shrink_coeff_list= np.full(alpha_dim+1,best_shrink_coeff)
    shrink_coeff_list = np.append(shrink_coeff_list,0)

    logit_model = smf.logit(stats_model_input, data=data)
    result = logit_model.fit_regularized(method='l1', alpha=shrink_coeff_list, maxiter=100, 
                                    trim_mode='auto', full_output=True, disp=True, maxfun=100)

    beta = result.params[med]
    OR = np.exp(beta)
    pval = result.pvalues[med]
    SE = result.bse[med]
    OR_LL = np.exp(result.conf_int(alpha=0.05)[0][med])
    OR_UL = np.exp(result.conf_int(alpha=0.05)[1][med])
    store_dict[ix]={'Disease':dz,'Medication':med,'odds ratio':OR,'p-val':pval,'95% CI (LL)':OR_LL,'95% CI (UL)': OR_UL,'best_shrink_coeff':best_shrink_coeff}
    print('{}th calculation is done: {} and {}.'.format(ix,dz,med))
    print(ix/med_dz_pair.shape[0]*100,'% is done')
    ix+=1
    
end = time.time()
adj_odds_table = pd.DataFrame.from_dict(store_dict,'index')

# Reverse the medication names
reverse_med_name_df = pd.Series(reverse_med_name).to_frame().reset_index().rename(columns={'index':'Medication',0:'Medication_reversed'})
final_adj_or = pd.merge(adj_odds_table,reverse_med_name_df,on='Medication',how='left').drop(columns=['Medication']).rename(columns={'Medication_reversed':'Medication'})
reorder_col = ['Disease', 'Medication', 'odds ratio', 'p-val', '95% CI (LL)', '95% CI (UL)','best_shrink_coeff']
final_adj_or = final_adj_or[reorder_col]


save_name = 'ADJ_ODDS_RATIO_{}_2024.csv'.format(current_server)
final_adj_or.to_csv(save_name)
print('Data is saved')
## Record Elapsed Time ##
elicit_time = (end-start)/60/60
f = open("Elapsed_Time_Records.txt", "a+")
f.write("Elapsed Time for Adjusted OR in {}: {} hours".format(current_server,elicit_time))
f.write('\n')
f.close()

print('Analysis is done!')
