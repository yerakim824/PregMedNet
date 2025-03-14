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
import statsmodels.formula.api as smf
import time

start = time.time()
df = pd.read_csv('Dec2023_Raw_Data/FINAL_DATA_DEC2023_ver2.csv')
df = df.set_index('ENROLID_BABY')

baby_dz_list = [i for i in df.columns if i.__contains__('_Baby')]
baby_dz = df[baby_dz_list]
mom_med = df.loc[:,'5-methyltetrahydrofolic acid, Oral':]

odds_df = pd.concat([baby_dz,mom_med],axis=1)

med_start_loc = odds_df.columns.get_loc('5-methyltetrahydrofolic acid, Oral')
baby_dz = odds_df.iloc[:,:med_start_loc]
mom_med = odds_df.iloc[:,med_start_loc:]
dz_col = baby_dz.columns
med_col = mom_med.columns

print('DEC2023: start to calculate odds ratio -- 2024. (confidence interval)')
d = {}
i = 0


### ORIGINAL CODE ####
for ix in range(med_start_loc,len(odds_df.columns)):
    med = odds_df.iloc[:,ix].name    
    for dz in dz_col:
      Count = ((odds_df[dz]!=0)&(odds_df[med]!=0)).sum()
      if Count <=10:
          pass
      else:
        stats_model_input=dz+' ~ odds_df.iloc[:,{}]'.format(ix)
        try:
            reg = smf.logit(stats_model_input, data=odds_df).fit()
            pval = reg.pvalues[1]
            OR = np.exp(reg.params[1])
            SE = reg.bse[1]
            # OR_LL = OR-1.96*SE
            # OR_UL = OR+1.96*SE
            OR_LL = np.exp(reg.conf_int(alpha=0.05)[0]['odds_df.iloc[:, {}]'.format(ix)])
            OR_UL = np.exp(reg.conf_int(alpha=0.05)[1]['odds_df.iloc[:, {}]'.format(ix)])
        except:
            pval = np.nan
            OR = np.nan
            SE = np.nan
            OR_LL=np.nan
            OR_UL=np.nan
        d[i]={'Disease':dz,'Medication':med,'odds ratio':OR,'p-val':pval,'Count':Count, '95% CI (LL)':OR_LL,'95% CI (UL)': OR_UL}
        i+=1


odds_table = pd.DataFrame.from_dict(d,'index')
odds_table.to_csv('RESULTS_DEC2023_VER3/RESULT_RAW_OR_FINAL_2024.csv')

end = time.time()
elicit_time = (end-start)/60/60
print('ANALYSIS IS DONE! ELICIT TIME:',elicit_time,'hours')

