import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import muspan as ms
import os
import pandas as pd
import random
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from scipy.stats import chi2
sns.set(style="white",font_scale=2)

#%% Helpers
outpath = './SI_AppendixG_Components/'

def random_points_on_circle(center, radius, npoints):
    cx, cy = center
    points = []
    for _ in range(npoints):
        angle = 2*np.pi*np.random.rand()
        x = cx + ( 0.8+0.4*np.random.rand() ) * radius * np.cos(angle)
        y = cy + ( 0.8+0.4*np.random.rand() ) *radius * np.sin(angle)
        points.append((x, y))
    return points

    

def make_virtual_point_cloud(na, nb, ab_relationship, a_pattern, b_pattern=None, name=None):
    # ab_relationship - options are "correlated", "uncorrelated", "anticorrelated"
    # a_pattern - options are "aggregation", "circles", "unstructured"
    # b_pattern - if ab_relationship is "uncorrelated", options are "aggregation", "circles", "unstructured"
    #           - otherwise, b is related to a pattern according to ab_relationship
    if a_pattern == 'unstructured':
        points_a = 1000*np.random.rand(na,2)
    elif a_pattern == 'circles':
        nseeds = np.random.randint(5,20)
        radii = 100*np.random.rand(nseeds) + 10
        points_a = []
        seeds = 1000*np.random.rand(nseeds,2)
        for i in range(nseeds):
            points_a.extend(random_points_on_circle(seeds[i], radii[i], int(na/nseeds)))
        points_a = np.array(points_a)
    elif a_pattern == 'aggregation':
        nseeds = np.random.randint(5,50)
        points_a = []
        seeds = 1000*np.random.rand(nseeds,2)
        for i in range(nseeds):
            sigma = 20*np.random.rand() + 10
            pts = sigma*np.random.randn(int(na/nseeds),2) + seeds[i]
            points_a.extend(pts)
        points_a = np.array(points_a)
    else:
        raise ValueError('No')
        
    # Now place points of b
    if ab_relationship == 'uncorrelated':
        assert(b_pattern is not None)
        if b_pattern == 'unstructured':
            points_b = 1000*np.random.rand(nb,2)
        elif b_pattern == 'circles':
            nseeds = np.random.randint(5,20)
            radii = 100*np.random.rand(nseeds) + 10
            points_b = []
            seeds = 1000*np.random.rand(nseeds,2)
            for i in range(nseeds):
                points_b.extend(random_points_on_circle(seeds[i], radii[i], int(nb/nseeds)))
            points_b = np.array(points_b)
        elif b_pattern == 'aggregation':
            nseeds = np.random.randint(5,50)
            points_b = []
            seeds = 1000*np.random.rand(nseeds,2)
            for i in range(nseeds):
                sigma = 20*np.random.rand() + 10
                pts = sigma*np.random.randn(int(nb/nseeds),2) + seeds[i]
                points_b.extend(pts)
            points_b = np.array(points_b)
        else:
            raise ValueError('No')
    elif ab_relationship == "correlated":
        # randomly sample bs as long as they're close to as
        n = 0
        points_b = []
        while n < nb:
            loc = 1000*np.random.rand(2)
            if np.min(cdist(points_a, [loc])) < 20:
                # accept loc
                points_b.append(loc)
                n += 1
            else:
                # accept loc with probability 10%
                if np.random.rand() < 0.1:
                    points_b.append(loc)
                    n += 1
    elif ab_relationship == "anticorrelated":
        # randomly sample bs as long as they're far from as
        n = 0
        points_b = []
        while n < nb:
            loc = 1000*np.random.rand(2)
            if np.min(cdist(points_a, [loc])) > 20:
                # accept loc
                points_b.append(loc)
                n += 1
            else:
                # accept loc with probability 10%
                if np.random.rand() < 0.1:
                    points_b.append(loc)
                    n += 1
    else:
        raise ValueError('Nope')
    # Make muspan domain
    points_b = np.array(points_b)
    points = np.vstack((points_a,points_b))
    labels = ['A']*np.shape(points_a)[0] + ['B']*np.shape(points_b)[0]
    
    if name is None:
        name = 'Unnamed point cloud'
    domain = ms.domain(name)
    domain.add_points(points)
    domain.add_labels('Celltype', labels)
    ID = domain.add_shapes([[[0,0],[1000,0],[1000,1000],[0,1000]]], 'bdy', return_IDs=True)
    domain = ms.helpers.crop_domain(domain,('collection','bdy'))[ID[0]]
    return domain

i = 'A'
j = 'B'
stat_to_tex = {'PCF_gmax':f'$g^{{Hi}}_{{{i}{j}}}$',
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
               'PH_H1-persistence-sd':f'$\sigma(P^1_{{{i}{j}}})$', 
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
#%%
def make_virtual_patient(ad_relation,car_relation, nad, ncar):
    ad_domains = []
    car_domains = []
    for a in range(nad):
        # print(f'ad {a}')
        a_pattern = random.choice(["aggregation", "circles", "unstructured"])
        if ad_relation == 'uncorrelated':
            b_pattern = random.choice(["aggregation", "circles", "unstructured"])
        else:
            b_pattern = None
        na = np.random.randint(100, 2000)
        nb = np.random.randint(100, 2000)
        name = f'Apattern-{a_pattern}_Bpattern-{b_pattern}'
        domain = make_virtual_point_cloud(na, nb, ad_relation, a_pattern, b_pattern, name=name)
        ad_domains.append(domain)
    for b in range(ncar):
        # print(f'car {b}')
        a_pattern = random.choice(["aggregation", "circles", "unstructured"])
        if car_relation == 'uncorrelated':
            b_pattern = random.choice(["aggregation", "circles", "unstructured"])
        else:
            b_pattern = None
        na = np.random.randint(100, 2000)
        nb = np.random.randint(100, 2000)
        name = f'Apattern-{a_pattern}_Bpattern-{b_pattern}'
        domain = make_virtual_point_cloud(na, nb, car_relation, a_pattern, b_pattern, name=name)
        car_domains.append(domain)    
    return {'ad':ad_domains,'car':car_domains}


#%% Time to make a virtual patient cohort
remake_csv = False
if remake_csv:
    # Set random seed
    np.random.seed(314159)
    # Make 9 groups
    # Group CaCc - correlated adenoma, correlated carcinoma
    # Group CaAc - correlated adenoma, anticorrelated carcinoma
    # Group CaUc - correlated adenoma, uncorrelated carcinoma
    # Group AaCc - anticorrelated adenoma, correlated carcinoma
    # Group AaAc - anticorrelated adenoma, anticorrelated carcinoma
    # Group AaUc - anticorrelated adenoma, uncorrelated carcinoma
    # Group UaCc - uncorrelated adenoma, correlated carcinoma
    # Group UaAc - uncorrelated adenoma, anticorrelated carcinoma
    # Group UaUc - uncorrelated adenoma, uncorrelated carcinoma
    
    npergroup = 5
    results = []
    for i in range(9*npergroup):
        print(f'Patient {i}')
        j = i%9
        if j in [0,1,2]:
            ad_relation = 'correlated'
        elif j in [3,4,5]:
            ad_relation = 'anticorrelated'    
        elif j in [6,7,8]:
            ad_relation = 'uncorrelated'    
            
        if j % 3 == 0:
            car_relation = 'correlated'
        elif j % 3 == 1:
            car_relation = 'anticorrelated'
        elif j % 3 == 2:
            car_relation = 'uncorrelated'
        
        # Choose a random number of ROIs for this patient
        nad = np.random.randint(10,50)
        ncar = np.random.randint(10,50)
        domains = make_virtual_patient(ad_relation,car_relation, nad, ncar)
        folder = f'AdRel-{ad_relation}_CaRel-{car_relation}'
        os.makedirs(f'{outpath}Domains/{folder}',exist_ok=True)
        for dis in ['ad','car']:
            for p, domain in enumerate(domains[dis]):
                filename = f'{outpath}Domains/{folder}/Patient-{i}_AdRel-{ad_relation}_CaRel-{car_relation}_Dis-{dis}_ID-{p}'
                ms.io.save_domain(domain,f'{filename}.muspan')
                ms.visualise.visualise(domain,'Celltype',show_boundary=True)
                plt.title(domain.name)
                plt.gcf().patch.set_alpha(0)
                plt.savefig(f'{filename}.png')
                plt.close()
                
                # Now calculate the stats for this domain
                outline = {'sampleID':i,
                           'disease':dis,
                           'quadratLabel':p}
                
                # SUMMARY STATS
                counts, cts = ms.summary_statistics.label_counts(domain,'Celltype',normalised=False)
                
                # CELL COUNTS
                outline['Count_A'] = counts[0]
                outline['Count_B'] = counts[1]
                    
                r, g = ms.spatial_statistics.cross_pair_correlation_function(domain,('Celltype','A'),('Celltype','B'),max_R=150,annulus_step=5,annulus_width=10,visualise_output=False)
                W = ms.distribution.sliced_wasserstein_distance(domain, ('Celltype','A'),('Celltype','B'))
                
                (cta, ctb) = ('A', 'B')
                outline[f'WassersteinDistance_{cta}-{ctb}'] = W
                outline[f'PCF_{cta}-{ctb}_gmax'] = g[np.nanargmax(g)]
                outline[f'PCF_{cta}-{ctb}_rpeak'] = r[np.nanargmax(g)]
                outline[f'PCF_{cta}-{ctb}_gmin'] = g[np.nanargmin(g)]
                outline[f'PCF_{cta}-{ctb}_rtrough'] = r[np.nanargmin(g)]
                outline[f'PCF_{cta}-{ctb}_int0_20'] = np.trapz(g[0:5],r[0:5])
                outline[f'PCF_{cta}-{ctb}_int0_40'] = np.trapz(g[0:9],r[0:9])
                outline[f'PCF_{cta}-{ctb}_int0_100'] = np.trapz(g[0:21],r[0:21])
                
                # TCMs
                TCM = ms.spatial_statistics.topographical_correlation_map(domain, ('Celltype','A'), ('Celltype','B'),radius_of_interest=50,kernel_sigma=50)
                lsf = ms.topology.level_set_filtration(TCM)
                ms.visualise.persistence_diagram(lsf)
                plt.close() # Don't ask, the visualise and close gets round a bug in muspan 1.1.0 that's fixed since 1.2.0
                vec, stats = ms.topology.vectorise_persistence(lsf)
                for j in range(len(vec)):
                    statname = '-'.join(stats[j].split(' '))
                    outline[f'TCM_{cta}-{ctb}_{statname}'] = vec[j]
                    
                TCM = ms.spatial_statistics.topographical_correlation_map(domain, ('Celltype','B'), ('Celltype','A'),radius_of_interest=50,kernel_sigma=50)
                lsf = ms.topology.level_set_filtration(TCM)
                ms.visualise.persistence_diagram(lsf)
                plt.close() # Don't ask, the visualise and close gets round a bug in muspan 1.1.0 that's fixed since 1.2.0
                vec, stats = ms.topology.vectorise_persistence(lsf)            
                for j in range(len(vec)):
                    statname = '-'.join(stats[j].split(' '))
                    outline[f'TCM_{ctb}-{cta}_{statname}'] = vec[j]
                    
                results.append(outline)
                
    df = pd.DataFrame(results)
    df.to_csv(f'{outpath}Results.csv')

#%% Read in the csv and do statistical tests
df = pd.read_csv(f'{outpath}Results.csv')
info_df = df[df.columns[1:4]]
data_df = df[df.columns[4:]]

# Clean data as per real analysis
pd.set_option('mode.use_inf_as_na', True)
col_names = data_df.columns
fill_cols = []
fill_vals = []

# TCM
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

IDs = np.unique(info_df.sampleID)
all_outputs = []
for ID in IDs:
    print(ID)    
    patient_mask = info_df['sampleID'] == ID
    patient_df = data_df[patient_mask]
    
    ct1 = 'A'
    ct2 = 'B'

    cellpair = f'{ct1}-{ct2}'
    stats = []
    
    phstats = ['H0-nBars','H1-nBars','H0-death-mean','H0-death-sd','H1-birth-mean','H1_birth_sd','H1-death-mean','H1-death-sd',
               'H1-persistence-mean','H1-persistence-sd','nLoops']
    to_add = []
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
    
    diseases = info_df[patient_mask]['disease']
    quadratLabel = info_df[patient_mask]['quadratLabel']
    
    df_temp = patient_df
    df_temp['disease'] = diseases
    df_temp['quadratLabel'] = quadratLabel
    
    canmask = df_temp.disease == 'car'
    admask = df_temp.disease == 'ad'
    
    # Continue to u tests
    def getDataframeOutput(df_temp, admask, canmask, stat):
        canvals = np.array(df_temp[canmask][stat])
        advals = np.array(df_temp[admask][stat])
        U, p = mannwhitneyu(x=advals,y=canvals)
        # plt.savefig(f'./figs_{ct1}-{ct2}/all_stats_{ID}/{ct1}-{ct2}_{stat}.png')
        dataframe_row = {'ct1':ct1,'ct2':ct2,'ID':ID,'Statistic':stat,'U':U,'pvalue':p,'n_adenoma_ROIs':len(advals), 'n_cancer_ROIs':len(canvals),'median_adenoma':np.median(advals), 'median_carcinoma':np.median(canvals)}
        return dataframe_row
    
    rows = []
    for stat in stats:
        rows.append(getDataframeOutput(df_temp, admask, canmask, stat))
    all_outputs.extend(rows)


#%% Some demonstration statistics and violin plots
sns.set(style="white",font_scale=2)
ID = 3
patient_mask = info_df['sampleID'] == ID
patient_df = data_df[patient_mask]
patient_df['disease'] = info_df[patient_mask]['disease']


ct1 = 'A'
ct2 = 'B'

cellpair = f'{ct1}-{ct2}'

canvals = np.array(df_temp[canmask][stat])
advals = np.array(df_temp[admask][stat])
U, p = mannwhitneyu(x=advals,y=canvals)

diseasecols = {'car':'r','ad':'b'}

for stat in stats:
    
    plt.figure(figsize=(8,12))
    sns.violinplot(patient_df,y=stat,hue='disease',split=True,inner='quart',gap=.1,cut=0,palette=diseasecols,legend=False)
    # plt.ylim([0,3.5])
    plt.xlabel(ID)
    lab = stat.replace('_A-B','')
    lab = lab.replace('__','_')
    plt.ylabel(stat_to_tex[lab])
    plt.tight_layout()
    plt.gcf().patch.set_alpha(0)
    plt.savefig(f'{outpath}ExampleViolins/{ID}_{stat}_violin.png')   
    plt.savefig(f'{outpath}ExampleViolins/{ID}_{stat}_violin.svg')   
    plt.close()
    
    plt.figure(figsize=(4,8))
    # Boxplots for each disease category
    sns.boxplot(
        data=patient_df,
        x='disease',            # side-by-side categories
        y=stat,                 # your numeric variable
        palette=diseasecols,
        width=0.6,
        showfliers=False        # usually cleaner when overlaying points
    )
    # Points overlaid on top (use stripplot or swarmplot)
    sns.swarmplot(
        data=patient_df,
        x='disease',
        y=stat,
        color='k',
        size=5,
        alpha=1,#0.6,
        dodge=False             # no hue here, so no dodge needed
        # Alternatively: sns.swarmplot(..., size=4) for nicer non-overlapping layout
    )
    plt.xlabel(ID)
    plt.ylabel(stat_to_tex[lab])
    
    from statannotations.Annotator import Annotator
    diseasepal = {"car": "r", "ad": "b"}

    pairs = [('ad','car')]
    # Create the annotator bound to your axis and data
    annot = Annotator(
        plt.gca(),
        pairs,
        data=patient_df,
        x='disease',
        y=stat,
        order=['ad','car']
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
    plt.savefig(f'{outpath}ExampleViolins/{ID}_{stat}_boxscatter.png')   
    plt.savefig(f'{outpath}ExampleViolins/{ID}_{stat}_boxscatter.svg')   
    
#%% Now plot the aggregated stats for some example patient
from tqdm import tqdm
ID = 6
ad_relation = 'uncorrelated'
car_relation = 'correlated'
# ID = 7
# ad_relation = 'uncorrelated'
# car_relation = 'anticorrelated'
folder = f'AdRel-{ad_relation}_CaRel-{car_relation}'
files = os.listdir(f'{outpath}Domains/{folder}/')
domains = [v for v in files if v.startswith(f'Patient-{ID}_')]
domains = [v for v in domains if v.endswith('.muspan')]
PCFs = {'ad':[],'car':[]}
Ws = {'ad':[],'car':[]}
coolplots = True
for filename in tqdm(domains):
    # Crude way of only plotting one simulation
    if coolplots:
        visualise_output = True
        coolplots = False
    else:
        visualise_output = False
    dis = filename.split('_')[3].split('-')[1]
    domain = ms.io.load_domain(f'{outpath}Domains/{folder}/{filename}')
    # Now calculate the stats for this domain
    
    r, g = ms.spatial_statistics.cross_pair_correlation_function(domain,('Celltype','A'),('Celltype','B'),max_R=150,annulus_step=5,annulus_width=10,visualise_output=visualise_output)
    PCFs[dis].append(g)
    
    W = ms.distribution.sliced_wasserstein_distance(domain, ('Celltype','A'), ('Celltype','B'))
    Ws[dis].append(W)

#%%
sns.set(style="white",font_scale=2)
df_output = pd.DataFrame(all_outputs)
df_output.ID = pd.to_numeric(df_output.ID,downcast='integer')
testdf = pd.DataFrame(df_output)

newindex = [f'{testdf.iloc[v].ID}_{testdf.iloc[v].ct1}-{testdf.iloc[v].ct2}' for v in range(len(testdf))]
testdf['newindex'] = newindex
testdf['StatisticType'] = [v.split('_')[0] for v in testdf.Statistic]
testdf['StatType'] = ['_'.join(v.split('_')[:1] + v.split('_')[2:]) for v in testdf.Statistic]
uniquestat = [f'{testdf.iloc[v].ID}_{testdf.iloc[v].Statistic}' for v in range(len(testdf))]
testdf['UniqueStatID'] = uniquestat
testdf = testdf.drop_duplicates(subset='UniqueStatID')

# # Now save this
testdf.to_csv('./synthetic-study_pvals_signed.csv')

#%%
testdf = pd.read_csv('./synthetic-study_pvals_signed.csv')
testdf = testdf.drop('Unnamed: 0',axis=1)

testdf['pvalue'][testdf['pvalue'].isna()] = 1
testdf['effect size'] = testdf['U']/(testdf['n_adenoma_ROIs']*testdf['n_cancer_ROIs'])
testdf['rank-biserial correlation'] = 1 - 2*testdf['U']/(testdf['n_adenoma_ROIs']*testdf['n_cancer_ROIs'])


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


def get_SHIFT_scores(all_statistics_df, ct1, ct2):
    # Filter dataframe to cell types of interest
    mask = ((all_statistics_df.ct1 == ct1) & (all_statistics_df.ct2 == ct2))
    filtered_df = all_statistics_df[mask]
    
    # Now do Fishers Method
    pvalue_dataframe = filtered_df.pivot(index="ID", columns="Statistic", values="pvalue")
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
        # Cap SHIFT scores at 20
        val = max([np.log10(ss_dict_fisherp[ID]),np.float64(-20)])
        SHIFT[ID] = -sign*val
    
    return SHIFT



def get_all_SHIFT_values(df):
    outvals = []
    (ct1, ct2) = ('A','B')
    SHIFT = get_SHIFT_scores(df, ct1, ct2)
    outvals.extend([{'ct1':ct1,'ct2':ct2,'ID':ID,'SHIFT':SHIFT[ID]} for ID in SHIFT])
    outdf = pd.DataFrame(outvals)
    return outdf

lim = 20
SHIFT_df = get_all_SHIFT_values(testdf)

#%% Visualise as "QR code"
qrdf = []
# for i in list(ss_dict.keys()):
for i in range(len(SHIFT_df)):
    ID = SHIFT_df.iloc[i]['ID']
    j = ID%9
    if j in [0,1,2]:
        ad_relation = 'correlated'
        ca = 'tab:blue'
    elif j in [3,4,5]:
        ad_relation = 'anticorrelated'    
        ca = 'tab:red'
    elif j in [6,7,8]:
        ad_relation = 'uncorrelated'    
        ca = 'tab:gray'
        
    if j % 3 == 0:
        car_relation = 'correlated'
        cc = 'tab:blue'
    elif j % 3 == 1:
        car_relation = 'anticorrelated'
        cc = 'tab:red'
    elif j % 3 == 2:
        car_relation = 'uncorrelated'
        cc = 'tab:gray'
    thisdict = {'ad_relation':ad_relation,'car_relation':car_relation,'ID':ID,'Score':SHIFT_df.iloc[i]['SHIFT']}
    qrdf.append(thisdict)
qrdf = pd.DataFrame(qrdf)

states = ['anticorrelated','uncorrelated','correlated']
outarray = np.zeros((3,3))
for i in range(len(states)):
    for j in range(len(states)):
        temp = qrdf[qrdf['ad_relation'] == states[i]]
        temp = temp[temp['car_relation'] == states[j]]
        outarray[i,j] = np.mean(temp['Score'])

#%% Create heatmap
plt.figure(figsize=(12.7,10))
sns.heatmap(outarray, annot=True, fmt=".1f", cmap="coolwarm",
            xticklabels=states, yticklabels=states)
# plt.title("Mean $\gamma(P,A,B)$")
plt.xlabel("Pathological State 2")
plt.ylabel("Pathological State 1")
plt.yticks(rotation=0)

colorbar = plt.gca().collections[0].colorbar
colorbar.set_label("$\gamma(P,A,B)$")

plt.tight_layout()

plt.savefig(f'{outpath}Synthetic_SHIFT_Heatmap.png')
plt.savefig(f'{outpath}Synthetic_SHIFT_Heatmap.svg')

