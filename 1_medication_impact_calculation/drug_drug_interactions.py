print('Calculate raw DDI (2024 ver.): changed med1, med2 / Double-check if the prev results was correct')
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

print('FEB 2024 Drug-Drug Interactions: Start to Load the Dataset')


import math
import time
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
# from sklearn.preprocessing import MinMaxScaler
# from collinearity import SelectNonCollinear
# from numpy.linalg import inv, det

server = 'nalab2'
print('current server: {}'.format(server))

## Load the relevant dataset
df = pd.read_csv('Dec2023_Raw_Data/FINAL_DATA_DEC2023_ver2.csv')
df = df.set_index('ENROLID_BABY')

## Define the disease and medication list
baby_dz_list = [i for i in df.columns if i.__contains__('_Baby')]
mom_dz_list = [i for i in df.columns if i.__contains__('_Mom')]
## baby_dz_list update
baby_dz_list =['RDS_Baby', 'Jaundice_Baby', 'BPD_OLD_Baby', 
               'Other_Resp_Dis_Baby', 'Kernicterus_Baby', 'TTN_Baby',
               'Sepsis_Baby', 'IVH_Baby', 'GALD_Baby', 'Seizures_Baby',
               'NEC_Baby', 'PPHN_Baby', 'Hypoglycemia_Baby', 'NAS_Baby',
               'Arrhythmia_Baby', 'Pneumonia_Baby', 'UTI_Baby', 'SGA_Baby',
               'LGA_Baby', 'Postmaturity_Baby', 'Neonatal_Death_Baby', 'ROP_Baby',
               'Anemia_AOP_Baby','Anemia_All_Baby']

## Divide the dataframe into neonatal complications and maternal medications
baby_dz = df[baby_dz_list]
mom_med = df.loc[:,'5-methyltetrahydrofolic acid, Oral':]
mom_med_count = mom_med.sum()
ga_df = df[['GESTATIONAL_AGE']]

disease_list = baby_dz.columns
ddi_df = pd.concat([baby_dz,mom_med,ga_df],axis=1)
tot_iteration = math.comb(1152, 2)*(len(baby_dz_list))
start = time.time()
d_final={}
j=0 ## Index number (valid indices)
i=0 ## Number of pairs

for DISEASE in disease_list:
    for med1 in mom_med.columns:
        med1_ix = mom_med.columns.get_loc(med1)
        for med2 in mom_med.columns[med1_ix+1:]:
            sel_df = ddi_df[[DISEASE,med1,med2,'GESTATIONAL_AGE']]
            sel_df['Interaction'] = sel_df[med1]*sel_df[med2]
            med12_count = sel_df['Interaction'].sum()
            if med12_count==0:
                pass
            elif ((sel_df['Interaction']!=0)&(sel_df[DISEASE]!=0)).sum()<=10:
                pass
            else:
                stats_model_input = '{} ~ sel_df.iloc[:,1] + sel_df.iloc[:,2] + Interaction + GESTATIONAL_AGE'.format(DISEASE)
                try:
                    reg = smf.logit(stats_model_input, data=sel_df).fit()
                    b1= reg.params['sel_df.iloc[:, 1]']
                    b2=reg.params['sel_df.iloc[:, 2]']
                    b3=reg.params['Interaction']
                    b1_pval = reg.pvalues['sel_df.iloc[:, 1]']
                    b2_pval = reg.pvalues['sel_df.iloc[:, 2]']
                    b3_pval=reg.pvalues['Interaction']
                except:
                    b1= np.nan
                    b2= np.nan
                    b3= np.nan
                    b1_pval = np.nan
                    b2_pval = np.nan
                    b3_pval= np.nan
                j+=1
                dz_med12_count = ((sel_df['Interaction']!=0)&(sel_df[DISEASE]!=0)).sum()
                med1_count = mom_med_count[med1]
                med2_count = mom_med_count[med2]
                d_final[j]={'Disease':DISEASE,'Med1':med1,'Med2':med2,'b1':b1,'b2':b2,'b3':b3,'pval(b1)':b1_pval,'pval(b2)':b2_pval,'pval(b3)':b3_pval,'Med1(count)':med1_count,'Med2(count)':med2_count,'Med1/2(count)':med12_count,'Dz+Med1/2(count)':dz_med12_count} 
                print(d_final[j])
            i+=1
            print(i,'th iteration {}% is done, and'.format(str(round(i/tot_iteration*100,2))),j,'th valid calculations are done')



print('Final: ',j,' pairs showed co-prescription out of ',i,'pairs')
end =time.time()
print('Elapsed Time: ',(end-start)/60/60,'hours')
ddi_table = pd.DataFrame.from_dict(d_final,'index')

FILE_NAME='RESULTS_FEB2024/RESULT_DDI_adjusted_GA.csv'
ddi_table.to_csv(FILE_NAME)

## Record Elapsed Time ##
elicit_time = (end-start)/60/60
f = open("Elapsed_Time_Records.txt", "a+")
f.write("Elapsed Time for raw DDI (2024): {} hours".format(elicit_time))
f.write('\n')
f.write('Final: {} pairs showed co-prescription out of {} pairs'.format(j,i))
f.write('\n')
f.close()