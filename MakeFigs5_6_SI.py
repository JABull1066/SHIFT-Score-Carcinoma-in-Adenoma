import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import muspan as ms
import os
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.stats import chi2
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style='ticks',font_scale=1,rc=custom_params)

#%% Helpers

cm = plt.cm.tab10
celltypes = {'CD146':plt.cm.tab10(0),
              'CD34':plt.cm.tab10(1),
              'Cytotoxic T Cell':plt.cm.tab10(2),
              'Macrophage':plt.cm.tab10(9),
              'Neutrophil':plt.cm.tab10(4),
              'Periostin':plt.cm.tab10(6),
              'Podoplanin':plt.cm.tab10(5),
              'SMA':plt.cm.tab10(8),
              'T Helper Cell':plt.cm.tab10(3),
              'Treg Cell':plt.cm.tab20b(0)
              ,
              'Epithelium (imm)':[1,0.9,0.9,1],
              'Epithelium (str)':[0.9,1,0.9,1]}
cells = [v for v in celltypes if not v.startswith('Epi')]

cell_subscripts = {'CD146':'CD146',
        'CD34':'CD34',
        'Cytotoxic T Cell':'T',
        'Macrophage':'M',
        'Neutrophil':'N',
        'Periostin':'P',
        'Podoplanin':'Po',
        'SMA':'S',
        'T Helper Cell':'Th',
        'Treg Cell':'Tr',
        'Epithelium (stromal panel)':'E1',
        'Epithelium (immune panel)':'E2'}

def get_stat_to_text(ct1,ct2):
    if ct1 in cell_subscripts:
        i = cell_subscripts[ct1]
        j = cell_subscripts[ct2]
    else:
        # Helpful for when we want to just say 'i' and 'j'
        i = ct1
        j = ct2
    stat_to_tex = {'Count':f'$N_{i}$', 
                   'PCF_gmax':f'$g^{{Hi}}_{{{i}{j}}}$',
                   'PCF_gmin':f'$g^{{Lo}}_{{{i}{j}}}$',
                   'PCF_int0_100':f'$\int_0^{{100}}g_{{{i}{j}}}(r)dr$', 
                   'PCF_int0_20':f'$\int_0^{{20}}g_{{{i}{j}}}(r)dr$',
                   'PCF_int0_40':f'$\int_0^{{40}}g_{{{i}{j}}}(r)dr$', 
                   'PCF_rpeak':f'$r^{{Hi}}_{{{i}{j}}}$',
                   'PCF_rtrough':f'$r^{{Lo}}_{{{i}{j}}}$',
                   'PH_H0-death-mean':f'$\overline{{d^0_{{{i}}}}}$',
                   'PH_H0-death-sd':f'$\sigma(d_{{{i}}}^0)$', 
                   'PH_H0-nBars':f'$N^0_{{{i}}}$', 
                   'PH_H1-birth-mean':f'$\overline{{b^1_{{{i}}}}}$',
                   'PH_H1-death-mean':f'$\overline{{d^1_{{{i}}}}}$',
                   'PH_H1-death-sd':f'$\sigma(d_{{{i}}}^1)$', 
                   'PH_H1-nBars':f'$N^1_{{{i}}}$',
                   'PH_H1-persistence-mean':f'$\overline{{P^1_{{{i}}}}}$', 
                   'PH_H1-persistence-sd':f'$\sigma(P^1_{{{i}}})$', 
                   'PH_nLoops':f'$\mathcal{{N}}^1_{{{i}}}$',
                   'QCM':f'$QCM_{{{i}{j}}}$', 
                   'TCM_H0-death-mean':f'$\overline{{d^0_{{{i}{j}}}}}$',
                   'TCM_H0-death-sd':f'$\sigma(d_{{{i}{j}}}^0)$',
                   'TCM_H0-nBars':f'$N^0_{{{i}{j}}}$',
                   'TCM_H1-birth-mean':f'$\overline{{b^1_{{{i}{j}}}}}$',
                   'TCM_H1-death-mean':f'$\overline{{d^1_{{{i}{j}}}}}$', 
                   'TCM_H1-death-sd':f'$\sigma(d_{{{i}{j}}}^1)$',
                   'TCM_H1-nBars':f'$N^1_{{{i}{j}}}$',
                   'TCM_H1-persistence-mean':f'$\overline{{P^1_{{{i}{j}}}}}$',
                   'TCM_H1-persistence-sd':f'$\sigma(P^1_{{{i}{j}}})$',
                   'WassersteinDistance':f'$Wass_{{{i}{j}}}$'}
    return stat_to_tex


#%%
testdf = pd.read_csv('./p-values_Final.csv')
testdf = testdf.drop('Unnamed: 0',axis=1)
# Sort dataframe so that all ct1 == ct2 are at the top; this is useful for drop_duplicates in a minute, as otherwise single-cell metrics will potentially be influenced by ROIs that have been dropped due to insufficient numbers of the second cell
samecell_mask = testdf.ct1 == testdf.ct2
testdf = pd.concat([testdf[samecell_mask], testdf[~samecell_mask]], axis=0).reset_index(drop=True)
testdf['pvalue'][testdf['pvalue'].isna()] = 1
testdf['effect size'] = testdf['U']/(testdf['n_adenoma_ROIs']*testdf['n_cancer_ROIs'])
testdf['rank-biserial correlation'] = 1 - 2*testdf['U']/(testdf['n_adenoma_ROIs']*testdf['n_cancer_ROIs'])

testdf['log p'] = np.log10(testdf['pvalue'])

# # Use sign of rank-biserial correlation
testdf['log p directional'] = testdf['log p']
flip_signs = np.sign(testdf['rank-biserial correlation']) == 1
testdf['log p directional'][flip_signs] = -1*testdf['log p directional'][flip_signs]

# For convenience in calcuating these things, we only calculated one `direction' for symmetrical stats
# In order that we can account for these, we duplicate those rows in testdf with ct1 and ct2 flipped
# This applies to:
symmetric_stats = ['QCM','WassersteinDistance','PCF']
symstat_mask = sum([testdf['StatisticType'] == v for v in symmetric_stats]) > 0
diffcell_mask = testdf['ct1'] != testdf['ct2']
mask = diffcell_mask & symstat_mask
temp = testdf[mask]

# Construct new rows for dataframe
newdicts = []
for i in range(len(temp)):
    thisdict = temp.iloc[i].to_dict()
    # Update ct1, ct2, newindex, Statistic, and UniqueStatID
    ct1 = thisdict['ct1']
    ct2 = thisdict['ct2']
    toupdate = ['ct1','ct2','newindex','Statistic','UniqueStatID']
    for s in toupdate:
        thisdict[s] = thisdict[s].replace(ct1, "-CT1-")
        thisdict[s] = thisdict[s].replace(ct2, "-CT2-")
        thisdict[s] = thisdict[s].replace("-CT1-", ct2)
        thisdict[s] = thisdict[s].replace("-CT2-", ct1)
    newdicts.append(thisdict)
temp = pd.DataFrame(newdicts)
testdf = pd.concat([testdf,temp],ignore_index=True)

def make_colormap(cbar_max=np.log10(0.00005), critical_value_for_grey=np.log10(0.05),N=256):
    pvals = np.linspace(-cbar_max,cbar_max,N)
    to_zero = np.abs(pvals) < np.abs(critical_value_for_grey)
    cmap = [plt.cm.RdBu_r(v) for v in range(N)]
    for v in range(len(cmap)):
        if to_zero[v]:
            cmap[v] = (0.7,0.7,0.7,1)
    
    from matplotlib.colors import ListedColormap
    new_cm = ListedColormap(cmap)
    return new_cm

(ct1, ct2) = ('SMA','Neutrophil')
stat_to_tex = get_stat_to_text(ct1, ct2)
vmax = 5
new_cm = make_colormap(cbar_max=vmax, critical_value_for_grey=np.log10(0.05),N=256)

#%% Get all stats with these things
mask = ((testdf.ct1 == ct1) & (testdf.ct2 == ct2)) #| ((testdf.ct1 == ct2) & (testdf.ct2 == ct1))
tempdf = testdf[mask]
labs = []
for i in range(len(tempdf)):
    c1 = tempdf.iloc[i]['ct1']
    c2 = tempdf.iloc[i]['ct2']
    stattype = tempdf.iloc[i]['StatType']
    stat_to_tex = get_stat_to_text(c1, c2)
    labs.append(stat_to_tex[stattype])
tempdf[' Statistic '] = labs

glue = tempdf.pivot(index="ID", columns=" Statistic ", values="log p directional")

corr = glue.corr()
sns.set_theme(style='ticks',font_scale=1.8,rc=custom_params)
plt.figure(figsize=(15,12))
sns.heatmap(corr,xticklabels=True, yticklabels=True,cmap='RdBu_r',vmin=-1,vmax=1,cbar_kws={'label':'Correlation'})
plt.tight_layout()
plt.savefig(f'./SI_AppendixF/Correlations_{ct1}-{ct2}.png')
plt.savefig(f'./SI_AppendixF/Correlations_{ct1}-{ct2}.svg')
sns.set_theme(style='ticks',font_scale=1,rc=custom_params)

#%%

def get_SHIFT_scores(all_statistics_df, ct1, ct2):
    # Filter dataframe to cell types of interest
    mask = ((all_statistics_df.ct1 == ct1) & (all_statistics_df.ct2 == ct2))
    filtered_df = all_statistics_df[mask]
    labs = []
    for i in range(len(filtered_df)):
        c1 = filtered_df.iloc[i]['ct1']
        c2 = filtered_df.iloc[i]['ct2']
        stattype = filtered_df.iloc[i]['StatType']
        stat_to_tex = get_stat_to_text(c1, c2)
        labs.append(stat_to_tex[stattype])
    filtered_df['Stat'] = labs
    
    # Now do Fishers Method alternative
    pvalue_dataframe = filtered_df.pivot(index="ID", columns="Stat", values="pvalue")
    sig_scores_fisher = -2*np.sum(np.log(pvalue_dataframe),axis=1)
    k = np.shape(pvalue_dataframe)[1]# Number of tests being combined
    pvals = 1 - chi2.cdf(sig_scores_fisher,2*k)
    ss_dict_fisherp = {pvalue_dataframe.index[i]:pvals[i] for i in range(len(pvalue_dataframe.index))}
    
    # Now assign signs
    # Pull out specific statistics
    # {stat : direction of stat change associated with increased order}
    stats_for_sign_check = {'PCF_int0_20' : 1,
                            'PCF_int0_40' : 1,
                            'PCF_int0_100' : 1,
                            'QCM' : 1,
                            'PCF_gmax' : 1, 
                            'PCF_gmin' : 1,
                            'PCF_rtrough' : 1,
                            'PCF_rpeak' : -1,
                            'WassersteinDistance' : -1
                            }
    consensus = []
    for stat in stats_for_sign_check:
        if np.any(filtered_df['StatType'] == stat):
            # Only include statistics that exist for this cell pair
            x = filtered_df[filtered_df['StatType'] == stat]
            signs = np.sign(x['rank-biserial correlation']) == np.sign(stats_for_sign_check[stat])
            consensus.append({x.iloc[i]['ID']:signs.iloc[i] for i in range(len(signs))})
    consensus_df = pd.DataFrame(consensus)
    signs = np.sum(consensus_df) > 0.5*len(consensus_df)
    
    SHIFT = {}
    for ID in pvalue_dataframe.index:
        if signs[ID]:
            sign = 1
        else:
            sign = -1
        # Cap SHIFT scores at 10
        val = max([np.log10(ss_dict_fisherp[ID]),np.float64(-20)])
        # if val < -10:
        # print(val)
        SHIFT[ID] = -sign*val
    
    return SHIFT



def get_all_SHIFT_values(df):
    outvals = []
    for ct1 in cells:
        for ct2 in cells:
            SHIFT = get_SHIFT_scores(df, ct1, ct2)
            outvals.extend([{'ct1':ct1,'ct2':ct2,'ID':ID,'SHIFT':SHIFT[ID]} for ID in SHIFT])
    outdf = pd.DataFrame(outvals)
    return outdf

lim = 20
SHIFT_df = get_all_SHIFT_values(testdf)



#%% How correlated are the different statistics?
stat_to_text = get_stat_to_text('i','j')
corrs = []
for ct1 in cells:
    for ct2 in cells:
        mask = ((testdf.ct1 == ct1) & (testdf.ct2 == ct2))
        tempdf = testdf[mask]
        glue = tempdf.pivot(index="ID", columns="StatType", values="log p directional")
        glue['nancount'] = np.sum(np.isnan(glue),axis=1)
        glue = glue.sort_values('nancount',ascending=False)
        glue = glue.drop(columns=['nancount'])
        cols = list(np.sum(np.isnan(glue),axis=0).sort_values().index)
        glue = glue[cols]

        corr = glue.corr()
        corrs.append({'ct1':ct1,'ct2':ct2,'cellpair':f'{ct1}-{ct2}','corr':corr})
        
dfs = [v['corr'] for v in corrs]
df_concat = pd.concat(dfs)
out = df_concat.groupby(df_concat.index)
df_means = out.mean()
df_means = df_means.reindex(sorted(df_means.columns), axis=1)
df_means = df_means.rename(stat_to_text,axis=0)
df_means = df_means.rename(stat_to_text,axis=1)
sns.set_theme(style='ticks',font_scale=1.8,rc=custom_params)
plt.figure(figsize=(15,12))
sns.heatmap(df_means,xticklabels=True, yticklabels=True,cmap='RdBu_r',vmin=-1,vmax=1,cbar_kws={'label':'Correlation'})
plt.gca().collections[0].cmap.set_bad('grey')
plt.tight_layout()
plt.xlabel('Summary statistic')
plt.ylabel('Summary statistic')
plt.savefig('./SI_AppendixF/Correlations_all.png')
plt.savefig('./SI_AppendixF/Correlations_all.svg')
sns.set_theme(style='ticks',font_scale=1,rc=custom_params)

#%% SHIFT Matrices
IDs = np.unique(testdf.ID)
for ID in [2358, 25914, 41632]:
    this_shift = SHIFT_df[SHIFT_df['ID'] == ID]
    glue = this_shift.pivot(index="ct1", columns="ct2", values="SHIFT")
    plt.figure(figsize=(10,8))
    sns.heatmap(glue,xticklabels=True, yticklabels=True,cbar_kws={'label': '$\gamma(n,i,j)$','extend':'both'},cmap='RdBu_r', vmax=lim, vmin=-lim)#,center=0)#
    # plt.title(ID)
    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18) # Size here overrides font_prop
    plt.tight_layout()
    plt.savefig(f'./Fig5_Components/SigScore_new_{ID}.png')
    plt.savefig(f'./Fig5_Components/SigScore_new_{ID}.svg')
#%% Cluster patients based on signficance scores
sns.set_theme(style='ticks',font_scale=0.8,rc=custom_params)
# First, get vector for each patient
tempdf = SHIFT_df
clusteron = 'SHIFT'
# clusteron = 'Directional mean log q'
tempdf['Celltype Pair'] = [f'{tempdf.ct1.iloc[v]}-{tempdf.ct2.iloc[v]}' for v in range(len(tempdf))]
glue = tempdf.pivot(index="ID", columns="Celltype Pair", values=clusteron)
Y = glue.fillna(0)
# Do our own linkages explicitly so we can access clusters


criterion = 'distance'#'distance'
method = 'ward'#'weighted'
t = 350

row_linkage = hierarchy.linkage(
    distance.pdist(Y), method=method, metric='correlation')
col_linkage = hierarchy.linkage(
    distance.pdist(Y.T), method=method, metric='correlation')


clusters = hierarchy.fcluster(row_linkage, t, criterion=criterion)
print(len(np.unique(clusters)))
cluster_cols = [plt.cm.tab20(v) for v in clusters]

# Sort column colours
c1 = [celltypes[v.split('-')[0]] for v in Y.columns]
c2 = [celltypes[v.split('-')[1]] for v in Y.columns]
color_df = pd.DataFrame({"Celltype 1": c1, "Celltype 2": c2}, index=Y.columns)

clustermap = sns.clustermap(Y, cmap='RdBu_r',row_colors=cluster_cols,col_colors=color_df, row_linkage=row_linkage,col_linkage=col_linkage, xticklabels=True, yticklabels=True,metric='correlation',robust=True,center=0,figsize=(15,12),cbar_kws={'label':'$\gamma(n,i,j)$'})
plt.tight_layout()
plt.savefig(f'./Fig5_Components/Clustermap_t={t}.png')
plt.savefig(f'./Fig5_Components/Clustermap_t={t}.svg')
sns.set_theme(style='ticks',font_scale=1,rc=custom_params)

#%% Zoom in on individual patients
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
# Update path below to point to "ForRelease_SummaryStatisticsData.csv", available via the link in github
pathToSummaryDataframe = 'path/to/AllPairwiseStatistics_SummaryStatistics.csv'
stats_df = pd.read_csv(pathToSummaryDataframe)
stats_df = stats_df.drop('Unnamed: 0', axis=1) # Drop index
stats_df = stats_df[stats_df['DomainTissueFraction']>=0.8] # Drop regions with low tissue coverage
stats_df.reset_index(inplace=True, drop=True)

path_to_points = 'path/to/AllPointcloudsAsCSVs_central/'
path_to_ROI_coords = 'path/to/AllPointcloudsAsCSVs_central_coordinates.csv'
df_coords = pd.read_csv(path_to_ROI_coords)


#%%
def getROI(the_file):
    ROI = the_file.split('.')[0]
    xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
    ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
    df = pd.read_csv(f'{path_to_points}{the_file}.csv')
    
    
    domain = ms.domain(ROI)
    # First add only non-Epithelium points
    epi_mask = np.array([v.startswith('E') for v in df['Celltype']])
    domain.add_points(np.asarray([df['x'][~epi_mask],df['y'][~epi_mask]]).T,collection_name='Cells')
    domain.add_labels('Celltype',df['Celltype'][~epi_mask])
    cols = {v:celltypes[v] for v in celltypes if v in np.unique(df.Celltype)}
    domain.update_colors(cols,label_name='Celltype')
    
    # Add epithelium points as a second population, just in case we want them later for vis
    domain.add_points(np.asarray([df['x'][epi_mask],df['y'][epi_mask]]).T,collection_name='Epithelium',zorder=0)
    # Reset the boundary
    bdy = [[xmin,ymin],[xmin+1000,ymin],[xmin+1000,ymin+1000],[xmin,ymin+1000]]
    domain.estimate_boundary(method='specify',specify_boundary_coords=bdy)
    return domain

#%% Plot some metric for WSI for these patients
sns.set_theme(style='ticks',font_scale=2,rc=custom_params)
(ct1, ct2) = ('Macrophage','Periostin')
stat_to_tex = get_stat_to_text(ct1, ct2)

loop = [(f'WassersteinDistance_{ct1}-{ct2}', 300, stat_to_tex['WassersteinDistance']), # For SI
        (f'QCM_{ct1}-{ct2}', 2.5, stat_to_tex['QCM']), # For SI
        (f'TCM_{ct1}-{ct2}_H1-death-mean', 10, stat_to_tex['TCM_H1-death-mean']), # For SI
        (f'PCF_{ct1}-{ct2}_gmin', 2, stat_to_tex['PCF_gmin'])
         ]
for (metric, vmax, metric_label) in loop:
    to_plot = [41251, 5827, 30268,
               14680, 25914, 41248]
    for ID in to_plot:
        minidf = stats_df[stats_df.sampleID == ID]
        patient_df = []
        for i in range(len(minidf)):
            the_file = minidf.iloc[i].Name
            ROI = the_file.split('.')[0]
            xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
            ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
            patient_dict = {'disease': ROI.split('_')[1],
                            metric: minidf.iloc[i][metric],
                            'x' : xmin,
                            'y' : ymin}
            patient_df.append(patient_dict)
        patient_df = pd.DataFrame(patient_df)
        
        plt.figure(figsize=(6,8))
        diseasecols = {'adenoma':'b','cancer':'r'}
        rectangles = []
        
        ecs = []
        for i in range(len(patient_df)):
            row = patient_df.iloc[i]
            rect = Rectangle((row['x'], row['y']), 1000, 1000)
            rectangles.append(rect)
            ecs.append(diseasecols[row['disease']])
        if metric.startswith('QCM'):
            vmin = -vmax
        else:
            vmin = 0
        
        if np.any([metric.startswith(v) for v in ['Wass','TCM']]):
            cm = plt.cm.get_cmap('plasma')
            cm.set_bad([0.7,0.7,0.7,1])
        else:
            cm = plt.cm.get_cmap('RdBu_r')
            cm.set_bad([0.7,0.7,0.7,1])
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        coll = PatchCollection(
        rectangles, cmap=cm, array=patient_df[metric], norm=norm
        )
        coll.set_edgecolors(ecs)
        plt.gca().add_collection(coll)
        plt.xlim([patient_df.x.min(),patient_df.x.max()])
        plt.ylim([patient_df.y.min(),patient_df.y.max()])
        plt.gca().axis('equal')
        
    
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ticks=np.linspace(vmin,vmax,3), 
                     label=metric_label,
                     ax=plt.gca())
        plt.title(f'{ID}')
        os.makedirs(f'./Fig6_Components/{ID}/', exist_ok=True)
        plt.savefig(f'./Fig6_Components/{ID}/{metric}_heatmap.png')   
        plt.savefig(f'./Fig6_Components/{ID}/{metric}_heatmap.svg')   
            
    
        from scipy.stats import mannwhitneyu
        advals = np.array(minidf[minidf.disease=='adenoma'][metric])
        canvals = np.array(minidf[minidf.disease=='cancer'][metric])
        advals = advals[~np.isnan(advals)]
        canvals = canvals[~np.isnan(canvals)]
        U, p = mannwhitneyu(x=advals,y=canvals)
        rankbiserial = 2*U/(len(advals)*len(canvals)) - 1
        print(ID, metric, U, rankbiserial, p)
        
        
        plt.figure(figsize=(8,8))
        # Boxplots for each disease category
        sns.boxplot(
            data=minidf,
            x='disease',            # side-by-side categories
            y=metric,                 # your numeric variable
            palette=diseasecols,
            width=0.6,
            showfliers=False,        # usually cleaner when overlaying points,
            order=['adenoma','cancer']
        )
        # Points overlaid on top (use stripplot or swarmplot)
        sns.swarmplot(
            data=minidf,
            x='disease',
            y=metric,
            color='k',
            size=5,
            alpha=1,#0.6,
            order=['adenoma','cancer'],
            dodge=False             # no hue here, so no dodge needed
            # Alternatively: sns.swarmplot(..., size=4) for nicer non-overlapping layout
        )
        plt.xlabel(ID)
        plt.ylabel(metric_label)
        
        from statannotations.Annotator import Annotator
        diseasepal = {"cancer": "r", "adenoma": "b"}
        
        pairs = [('adenoma','cancer')]
        # Create the annotator bound to your axis and data
        annot = Annotator(
            plt.gca(),
            pairs,
            data=minidf,
            x='disease',
            y=metric,
            order=['adenoma','cancer']
        )
        
        # Configure: test type, label style, and placement
        annot.configure(
            test='Mann-Whitney',         # uses scipy.stats.mannwhitneyu under the hood
            text_format='star',          # 'star' | 'simple' (p=) | 'full'
            loc='inside',               # place bracket & text above the axes
            verbose=1
        )
        
        
        # Run the test and draw the bracket + label
        annot.apply_and_annotate()
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'./Fig6_Components/{ID}/{metric}_boxplot.png')   
        plt.savefig(f'./Fig6_Components/{ID}/{metric}_boxplot.svg')  
    
#%%
candidate_ROIs = {41251:['adenoma_ID-104','cancer_ID-157'],
                  5827 :['adenoma_ID-79','cancer_ID-147'],
                  30268:['adenoma_ID-170','cancer_ID-1265'],
                  14680:['adenoma_ID-101','adenoma_ID-121','cancer_ID-212','cancer_ID-161'],
                  25914:['adenoma_ID-73','cancer_ID-735'],
                  41248:['adenoma_ID-283','cancer-105']
                  }

for ID in candidate_ROIs:
    domains = []
    vals = []
    lh = []
    for dis in ['adenoma','cancer']:
        minidf = stats_df[stats_df['sampleID'] == ID]
        minidf = minidf[minidf.disease == dis]
        # Filter to only ROIs with at least 20 of each cell type of interest
        for ct in [ct1, ct2]:
            minidf = minidf[minidf[f'Count_{ct}']>= 20]
        # Plot top 3 and bottom 3 values of metric
        low = list(np.argsort(minidf[metric])[0:5])
        domains.extend(list(minidf.iloc[low].Name))
        vals.extend(list(minidf.iloc[low][metric]))
        lh.extend(['Low']*len(low))
        
        high = list(np.argsort(minidf[metric])[-5:])
        domains.extend(list(minidf.iloc[high].Name))
        vals.extend(list(minidf.iloc[high][metric]))
        lh.extend(['High']*len(high))

    for ind, this in enumerate(domains):
        domain = getROI(this)
        domain.update_colors({'Macrophage':plt.cm.tab10(9),'T Helper Cell':plt.cm.tab10(3)},label_name='Celltype')
        ms.visualise.visualise(domain,('Constant',[0.7,0.7,0.7,1]),marker_size=5,show_boundary=True)
        population_A = ms.query.query(domain,('label','Celltype'),'is',ct2)
        population_B = ms.query.query(domain,('label','Celltype'),'is',ct1)
        if len(ms.query.interpret_query(population_A|population_B)) > 0:
            # Catch any ROIs where there aren't any of either cell type
            domain.unit_of_length = 'pixels'
            ms.visualise.visualise(domain,'Celltype',objects_to_plot=population_A|population_B,ax=plt.gca(),marker_size=50, add_scalebar=True, add_cbar=False, scalebar_kwargs={'borderpad':-0.7, 'loc':'lower right'}, show_boundary=True)
        # sns.set(font_scale=1)
        plt.title(f'{this} - {vals[ind]:.3f} - {"_".join([str(int(v)) for v in domain.bounding_box[0,:]])}')
        os.makedirs(f'./Fig6_Components/{ID}/', exist_ok=True)
        plt.savefig(f'./Fig6_Components/{ID}/{lh[ind]}_{this}.png')   
        plt.savefig(f'./Fig6_Components/{ID}/{lh[ind]}_{this}.svg')   
        plt.close()

#%%
plt.close('all')
from tqdm import tqdm
# Plot PCFs
(ct1, ct2) = ('Macrophage','Periostin')
for ID in to_plot:
    gs = []
    diseasetypes = []
    colarray = []
    nct1s = []
    nct2s = []
    admask = []
    for dis in ['adenoma','cancer']:
        mask = [v.startswith(f'{ID}_{dis}') for v in df_coords.name]
        options = list(df_coords.name[mask])
        for option in tqdm(options):
            try:
                domain = getROI(option)
                # Cross PCF
                population_A = ms.query.query(domain,('label','Celltype'),'is',ct1)
                population_B = ms.query.query(domain,('label','Celltype'),'is',ct2)
                nct1 = len(ms.query.interpret_query(population_B))
                nct2 = len(ms.query.interpret_query(population_A))
                if (nct1 < 20) or (nct2 < 20):
                    # Skip ROIs with low cell counts
                    continue
                r, g = ms.spatial_statistics.cross_pair_correlation_function(domain, population_A, population_B,max_R=150,annulus_step=5,annulus_width=10,visualise_output=False)
                gs.append(g)
                diseasetypes.append(dis)
                colarray.append(diseasecols[dis])
                nct1s.append(nct1)
                nct2s.append(nct2)
                admask.append(dis == 'adenoma')
            except:
                continue

    gs = np.asarray(gs)
    diseasetypes = np.array(diseasetypes)
    nct1s = np.array(nct1s)
    nct2s = np.array(nct2s)
    admask = np.array(admask)
    
    
    # Bootstrap
    plt.figure(figsize=(8,6))
    plt.gca().axhline(1,c='k',linestyle=':',lw=2)
    gad = np.mean(gs[admask].T,axis=1)
    gca = np.mean(gs[~admask].T,axis=1)
    # Generate bootstrap
    advals = gs[admask]
    cavals = gs[~admask]
    adboot = np.mean(advals[np.random.choice(len(advals), (len(advals),1000), replace=True)],axis=0)
    caboot = np.mean(cavals[np.random.choice(len(cavals), (len(cavals),1000), replace=True)],axis=0)
    plt.plot(r,gad,c='b',lw=4)
    plt.plot(r,gca,c='r',lw=4)
    plt.fill_between(r, np.percentile(adboot,5,axis=0),np.percentile(adboot,95,axis=0), color='b',alpha=0.5)
    plt.fill_between(r, np.percentile(caboot,5,axis=0),np.percentile(caboot,95,axis=0), color='r',alpha=0.5)
    plt.xlabel('$r$')
    plt.ylabel('$g_{'+ct1[0]+ct2[0]+'}(r)$')
    plt.tight_layout()
    plt.savefig(f'./Fig6_Components/{ID}/PCF_{ct1}-{ct2}.svg')   
    plt.savefig(f'./Fig6_Components/{ID}/PCF_{ct1}-{ct2}.png')
