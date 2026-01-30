import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
sns.set_theme(style='white',font_scale=2)

pathToSummaryDataframe = 'path/to/SummaryStatistics_Final.csv'
stats_df = pd.read_csv(pathToSummaryDataframe)
stats_df = stats_df.drop('Unnamed: 0', axis=1) # Drop index
stats_df = stats_df[stats_df['DomainTissueFraction']>=0.8] # Drop regions with low tissue coverage
stats_df.reset_index(inplace=True, drop=True)
# For convenience further down
info_df = stats_df[stats_df.columns[0:9]]
data_df = stats_df[stats_df.columns[9:]]
data_df = data_df.astype(float)
data_df_unfilled = copy.deepcopy(data_df)

#%% Clean data_df
#% Fill columns (not Phil Collins)
pd.set_option('mode.use_inf_as_na', True)
# Deal with NaNs and missing data in data_df
# Process as follows:
    # PH: H0 features -> 0
    # PH: H1 features ->
    # PCF: ending _r=x -> 1
    # PCF: ending _gmax, _gmin -> 1
    # PCF: ending _rpeak, _rtrough -> Column mean
    # QCM: all nan -> 0
    # WassersteinDistance: -> Column mean
    # TCM: As for PH
col_names = data_df.columns
fill_cols = []
fill_vals = []

# Full PCFs don't seem to add value, get rid of them
PCFcols = [v for v in col_names if '_r=' in v]
stats_df = stats_df.drop(PCFcols, axis=1)

# PH / TCM
fill_cols.append([v for v in col_names if v.startswith('PH')])
fill_vals.append(0)
fill_cols.append([v for v in col_names if v.startswith('TCM')])
fill_vals.append(0)
# PCF
fill_cols.append([v for v in col_names if v.endswith('_gmax')])
fill_vals.append(1)
fill_cols.append([v for v in col_names if v.endswith('_gmin')])
fill_vals.append(1)
fill_cols.append([v for v in col_names if '_r=' in v])
fill_vals.append(1)
for i in [20,40,100]:
    # Fill all integrals as though PCF were constant at 1
    fill_cols.append([v for v in col_names if v.endswith(f'_int0_{i}')])
    fill_vals.append(i)
#QCM 
fill_cols.append([v for v in col_names if v.startswith('QCM')])
fill_vals.append(0)
for i in range(len(fill_cols)):
    data_df[fill_cols[i]] = data_df[fill_cols[i]].fillna(value=fill_vals[i])

# For Wasserstein and _rpeak/_rtrough, fill with column mean
fill_cols = [v for v in col_names if v.startswith('Wasserstein')]
temp = [v for v in col_names if v.endswith('_rtrough')]
fill_cols.extend(temp)
temp = [v for v in col_names if v.endswith('_rpeak')]
fill_cols.extend(temp)
fill_vals = [np.nanmean(data_df[v]) for v in fill_cols]
for i in range(len(fill_cols)):
    data_df[fill_cols[i]] = data_df[fill_cols[i]].fillna(fill_vals[i])

#%% Helpers

cm = plt.cm.tab10
celltypes = {'CD146':cm(0),
             'CD34':cm(1),
             'Cytotoxic T Cell':cm(2),
             'Macrophage':cm(3),
             'Neutrophil':cm(4),
             'Periostin':cm(6),
             'Podoplanin':cm(5),
             'SMA':cm(8),
             'T Helper Cell':cm(9),
             'Treg Cell':plt.cm.tab20b(0)}
              # ,
              # 'Epithelium (imm)':[1,0.9,0.9,1],
              # 'Epithelium (str)':[0.9,1,0.9,1]}




#%% For all cell pairs of interest, how useful is a given metric in discriminating ad/car?
from scipy.stats import mannwhitneyu
import time

IDs = np.unique(info_df.sampleID)
all_outputs = []
for ID in IDs:
    df_output = []
    # First build statistics vector
    patient_mask = info_df['sampleID'] == ID
    patient_df = data_df[patient_mask]
    
    for ct1 in celltypes:
        for ct2 in celltypes:
            cellpair = f'{ct1}-{ct2}'
            # print(f'Beginning {cellpair}')      
            stats = [f'Count_{ct1}', f'Count_{ct2}']
            #PH
            phstats = ['H0-nBars','H1-nBars','H0-death-mean','H0-death-sd','H1-birth-mean','H1_birth_sd','H1-death-mean','H1-death-sd',
                       'H1-persistence-mean','H1-persistence-sd','nLoops']
            to_add = [[f'PH_{ct}_{w}' for ct in (ct1, ct2)] for w in phstats]
            for v in to_add:
                stats.extend(v)
            to_add = []
            to_add.append(f'QCM_{ct1}-{ct2}')
            to_add.append(f'WassersteinDistance_{ct1}-{ct2}')
            to_add.append(f'PCF_{ct1}-{ct2}_gmax')
            to_add.append(f'PCF_{ct1}-{ct2}_rpeak')
            to_add.append(f'PCF_{ct1}-{ct2}_gmin')
            to_add.append(f'PCF_{ct1}-{ct2}_rtrough')
            to_add.append(f'PCF_{ct1}-{ct2}_int0_20')
            to_add.append(f'PCF_{ct1}-{ct2}_int0_40')
            to_add.append(f'PCF_{ct1}-{ct2}_int0_100')
            for w in phstats:
                to_add.append(f'TCM_{ct1}-{ct2}_{w}')
            stats.extend(to_add)
            stats = [v for v in stats if v in data_df.columns]
            
            # Now that we have stats, get a mask for the relevant ROIs
            mask = (patient_df[f'Count_{ct1}'] >= 20) & (patient_df[f'Count_{ct2}'] >= 20)
            diseases = info_df[patient_mask][mask]['disease']
            names = info_df[patient_mask][mask]['Name']
            
            df_temp = patient_df[mask]
            df_temp['disease'] = diseases
            df_temp['Name'] = names
 
            canmask = df_temp.disease[mask] == 'cancer'
            admask = df_temp.disease[mask] == 'adenoma'
            if (len(canmask) <= 1) or (len(admask) <= 1):
                # Can't classify as all from same disease stage
                print('here')
                # continue
            # Otherwise, we can continue to u tests
    
            def getDataframeOutput(df_temp, admask, canmask, stat):
                canvals = np.array(df_temp[canmask][stat])
                advals = np.array(df_temp[admask][stat])
                U, p = mannwhitneyu(x=advals,y=canvals)
                dataframe_row = {'ct1':ct1,'ct2':ct2,'ID':ID,'Statistic':stat,'U':U,'pvalue':p,'n_adenoma_ROIs':len(advals), 'n_cancer_ROIs':len(canvals),'median_adenoma':np.median(advals), 'median_carcinoma':np.median(canvals)}
                return dataframe_row
            
            rows = []
            tic = time.time()
            for stat in stats:
                rows.append(getDataframeOutput(df_temp, admask, canmask, stat))
            toc = time.time()
            print(f'{ID} - {cellpair} complete: Time {toc - tic}')
            df_output.extend(rows)
    df_out_temp = pd.DataFrame(df_output)
    df_out_temp.to_csv(f'./temp_withU/{int(ID)}.csv')
    all_outputs.extend(df_output)
    
    
#%% Read in all the csvs and combine
dfs = []
for ID in IDs:
    print(ID)
    df = pd.read_csv(f'./temp_withU/{int(ID)}.csv')
    dfs.append(df)
    
df_output = pd.concat(dfs)
df_output.ID = pd.to_numeric(df_output.ID,downcast='integer')
testdf = pd.DataFrame(df_output)
testdf = testdf.drop('Unnamed: 0',axis=1)
# Sort dataframe so that all ct1 == ct2 are at the top; this is useful for drop_duplicates in a minute, as otherwise single-cell metrics will potentially be influenced by ROIs that have been dropped due to insufficient numbers of the second cell
samecell_mask = testdf.ct1 == testdf.ct2
testdf = pd.concat([testdf[samecell_mask], testdf[~samecell_mask]], axis=0).reset_index(drop=True)

newindex = [f'{testdf.iloc[v].ID}_{testdf.iloc[v].ct1}-{testdf.iloc[v].ct2}' for v in range(len(testdf))]
testdf['newindex'] = newindex
testdf['StatisticType'] = [v.split('_')[0] for v in testdf.Statistic]
testdf['StatType'] = ['_'.join(v.split('_')[:1] + v.split('_')[2:]) for v in testdf.Statistic]
uniquestat = [f'{testdf.iloc[v].ID}_{testdf.iloc[v].Statistic}' for v in range(len(testdf))]
testdf['UniqueStatID'] = uniquestat
testdf = testdf.drop_duplicates(subset='UniqueStatID')

# # Now save this
testdf.to_csv('./all_pvals_signed.csv')






