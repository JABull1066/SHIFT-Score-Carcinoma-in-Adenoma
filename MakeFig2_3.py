import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import muspan as ms
import copy
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
sns.set_theme(style='white',font_scale=2)

outpath = './Fig2_3_Components/'

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
    return stat_to_tex


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
#%%
# Quick write out of dataframe for SI

roi_counts = pd.crosstab(
    index=info_df["sampleID"],
    columns=info_df["disease"]
)

# Fill missing combinations with 0 and enforce column order
roi_counts = roi_counts.reindex(columns=["cancer", "adenoma"], fill_value=0)

roi_counts.to_csv('./ROIcounts.csv', index=True)

#%% Choose a subset of statistics
cells = [v for v in celltypes.keys() if not v.startswith('Epi')]
stats = [f'Count_{v}' for v in cells]
#PH
phstats = ['H0-nBars','H1-nBars','H0-death-mean','H0-death-sd','H1-birth-mean','H1_birth_sd','H1-death-mean','H1-death-sd',
           'H1-persistence-mean','H1-persistence-sd','nLoops']
to_add = [[f'PH_{ct}_{w}' for ct in cells] for w in phstats]
for v in to_add:
    stats.extend(v)
to_add = []
for ct1 in cells:
    for ct2 in cells:
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


mask = (data_df[f'Count_{ct1}'] >= 20) & (data_df[f'Count_{ct2}'] >= 20)
df = data_df[mask]

diseases = info_df[mask]['disease']
names = info_df[mask]['Name']
IDs = info_df[mask]['sampleID']
df['sampleID'] = IDs
df['disease'] = diseases
df['Name'] = names
diseasepal = {"cancer": "r", "adenoma": "b"}

#%% Specific patient
from scipy.stats import mannwhitneyu

to_plot = [
           (28717, 'PCF_Neutrophil-SMA_rpeak'),
           (28717, 'PCF_Neutrophil-SMA_gmin'),
           (28717, 'PCF_Neutrophil-SMA_int0_40'),
           (28717, 'PCF_Neutrophil-SMA_gmax'),
           (28717, 'PCF_Neutrophil-SMA_rtrough'),
           (28717, 'PCF_Neutrophil-SMA_int0_20'),
           (28717, 'PCF_Neutrophil-SMA_int0_100')
           ]
stat_to_tex = get_stat_to_text('Neutrophil', 'SMA')
av = []
cv = []
for ID, stat in to_plot:
    label = stat_to_tex['PCF_' + ('_').join(stat.split('_')[2:])]
    minidf = df[df['sampleID']==ID]
    
    plt.figure(figsize=(4,8))
    # Boxplots for each disease category
    sns.boxplot(
        data=minidf,
        x='disease',            # side-by-side categories
        y=stat,                 # your numeric variable
        palette=diseasepal,
        width=0.6,
        showfliers=False        # usually cleaner when overlaying points
    )
    # Points overlaid on top (use stripplot or swarmplot)
    sns.swarmplot(
        data=minidf,
        x='disease',
        y=stat,
        color='k',
        size=5,
        alpha=1,#0.6,
        dodge=False             # no hue here, so no dodge needed
        # Alternatively: sns.swarmplot(..., size=4) for nicer non-overlapping layout
    )
    plt.xlabel(ID)
    plt.ylabel(label)
    
    from statannotations.Annotator import Annotator
    diseasepal = {"cancer": "r", "adenoma": "b"}

    pairs = [('adenoma','cancer')]
    # Create the annotator bound to your axis and data
    annot = Annotator(
        plt.gca(),
        pairs,
        data=minidf,
        x='disease',
        y=stat,
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
    plt.savefig(f'{outpath}BoxAndScatterExample_{stat}.png')
    plt.savefig(f'{outpath}BoxAndScatterExample_{stat}.svg')
    
    canmask = minidf.disease == 'cancer'
    admask = minidf.disease == 'adenoma'

    canvals = np.array(minidf[canmask][stat])
    advals = np.array(minidf[admask][stat])
    U, p = mannwhitneyu(x=advals,y=canvals)
    print(ID, stat, U, p)
    av.append(advals)
    cv.append(canvals)
    
plt.figure()
plt.scatter(av[0],av[1],c='b')
plt.scatter(cv[0],cv[1],c='r')
plt.xlabel(to_plot[0][1])
plt.ylabel(to_plot[1][1])

#%% Repeat for the SI
from scipy.stats import mannwhitneyu

to_plot = ['QCM_Neutrophil-SMA',
           'WassersteinDistance_Neutrophil-SMA',
           'PCF_Neutrophil-SMA_gmax',
           'PCF_Neutrophil-SMA_rpeak',
           'PCF_Neutrophil-SMA_gmin',
           'PCF_Neutrophil-SMA_rtrough',
           #
           'TCM_Neutrophil-SMA_H0-death-mean',
           'TCM_Neutrophil-SMA_H0-death-sd',
           'TCM_Neutrophil-SMA_H0-nBars',
           'PCF_Neutrophil-SMA_int0_20',
           'PCF_Neutrophil-SMA_int0_40',
           'PCF_Neutrophil-SMA_int0_100',
           #
           'TCM_Neutrophil-SMA_H1-birth-mean',
           'TCM_Neutrophil-SMA_H1-death-mean',
           'TCM_Neutrophil-SMA_H1-death-sd',
           'TCM_Neutrophil-SMA_H1-persistence-mean',
           'TCM_Neutrophil-SMA_H1-persistence-sd',
           'TCM_Neutrophil-SMA_H1-nBars'
           ]
stat_to_tex = get_stat_to_text('Neutrophil', 'SMA')

fig, axes = plt.subplots(3,6,figsize=(3*8, 6*8.1))

for i, stat in enumerate(to_plot):
    yax = i%6
    xax = int(np.floor(i/6))
    ID = 28717
    x = stat.split('_')
    x.remove('Neutrophil-SMA')
    x = ('_').join(x)
    label = stat_to_tex[x]
    
    
    minidf = df[df['sampleID']==ID]


    # Boxplots for each disease category
    sns.boxplot(
        data=minidf,
        x='disease',            # side-by-side categories
        y=stat,                 # your numeric variable
        palette=diseasepal,
        width=0.6,
        showfliers=False,        # usually cleaner when overlaying points
        ax=axes[xax,yax]
    )
    # Points overlaid on top (use stripplot or swarmplot)
    sns.swarmplot(
        data=minidf,
        x='disease',
        y=stat,
        color='k',
        size=5,
        alpha=1,#0.6,
        dodge=False,             # no hue here, so no dodge needed
        ax=axes[xax,yax]
        # Alternatively: sns.swarmplot(..., size=4) for nicer non-overlapping layout
    )
    # plt.xlabel(ID)
    axes[xax,yax].set_ylabel(label)
    
    pairs = [('adenoma','cancer')]
    # Create the annotator bound to your axis and data
    annot = Annotator(
        axes[xax,yax],
        pairs,
        data=minidf,
        x='disease',
        y=stat,
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
plt.savefig(f'{outpath}SI_BoxAndScatterExample_all.png')
plt.savefig(f'{outpath}SI_BoxAndScatterExample_all.svg')
    


#%% # Do full PCF
path_to_points = 'path/to/AllPointcloudsAsCSVs_central/'
path_to_ROI_coords = 'path/to/AllPointcloudsAsCSVs_central_coordinates.csv'
df_coords = pd.read_csv(path_to_ROI_coords)
(ct1, ct2) = ('Neutrophil','SMA')
gs = []
diseasetypes = []
colarray = []
nct1s = []
nct2s = []
skipped = []

minidf = df[df['sampleID']==28717]
for i in range(len(minidf)):
    print(i, len(minidf))
    # Read in the ROI
    the_file = minidf.iloc[i].Name
    ROI = the_file.split('.')[0]
    xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
    ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
    df_patient = pd.read_csv(f'{path_to_points}{the_file}.csv')
    
    outline = {'Name':ROI,
               'sampleID':ROI.split('_')[0],
               'disease':ROI.split('_')[1],
               'quadratLabel':int(ROI.split('_')[2].split('-')[1]),
               'xMin':xmin,
               'yMin':ymin}
    
    domain = ms.domain(ROI)
    # First add only non-Epithelium points
    epi_mask = np.array([v.startswith('E') for v in df_patient['Celltype']])
    domain.add_points(np.asarray([df_patient['x'][~epi_mask],df_patient['y'][~epi_mask]]).T,collection_name='Cells')
    domain.add_labels('Celltype',df_patient['Celltype'][~epi_mask])
    cols = {v:celltypes[v] for v in celltypes if v in np.unique(df_patient.Celltype)}
    domain.update_colors(cols,label_name='Celltype')
    
    # Add epithelium points as a second population, just in case we want them later for vis
    domain.add_points(np.asarray([df_patient['x'][epi_mask],df_patient['y'][epi_mask]]).T,collection_name='Epithelium',zorder=0)
    # Reset the boundary
    bdy = [[xmin,ymin],[xmin+1000,ymin],[xmin+1000,ymin+1000],[xmin,ymin+1000]]
    domain.estimate_boundary(method='specify',specify_boundary_coords=bdy)

    # Cross PCF
    population_A = ms.query.query(domain,('label','Celltype'),'is',ct2)
    population_B = ms.query.query(domain,('label','Celltype'),'is',ct1)
    nct1 = len(ms.query.interpret_query(population_B))
    nct2 = len(ms.query.interpret_query(population_A))
    if (nct1 < 20) or (nct2 < 20):
        # Skip ROIs with low cell counts
        skipped.append(i)
        continue
    r, g = ms.spatial_statistics.cross_pair_correlation_function(domain, population_A, population_B,max_R=150,annulus_step=5,annulus_width=10,visualise_output=False)
    gs.append(g)
    diseasetypes.append(ROI.split('_')[1])
    if ROI.split('_')[1] == 'adenoma':
        c = 'b'
    else:
        c = 'r'
    colarray.append(c)
    nct1s.append(nct1)
    nct2s.append(nct2)
gs = np.asarray(gs)
diseasetypes = np.array(diseasetypes)
nct1s = np.array(nct1s)
nct2s = np.array(nct2s)


# Bootstrap
plt.figure(figsize=(10,5))
plt.gca().axhline(1,c='k',linestyle=':',lw=2)
admask = diseasetypes == 'adenoma'
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
plt.savefig(f'{outpath}PCF_example.png')
plt.savefig(f'{outpath}PCF_example.svg')

#%% Plot ROIs
to_plot = ['28717_adenoma_ID-12','28717_cancer_ID-60']

for i, ROI in enumerate(to_plot):
    the_file = f'{ROI}.csv'
    print(the_file)
    xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
    ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
    df_patient = pd.read_csv(f'{path_to_points}{the_file}')

    domain = ms.domain(ROI,'pixels')
    # First add only non-Epithelium points
    epi_mask = np.array([v.startswith('E') for v in df_patient['Celltype']])
    if np.sum(~epi_mask) < 5:
        continue
    if np.sum(epi_mask) < 5:
        continue
    domain.add_points(np.asarray([df_patient['x'][~epi_mask],df_patient['y'][~epi_mask]]).T,collection_name='Cells')
    domain.add_labels('Celltype',list(df_patient['Celltype'][~epi_mask]))
    cols = {v:celltypes[v] for v in celltypes if v in np.unique(df_patient.Celltype)}
    domain.update_colors(cols,label_name='Celltype')
    
    # Add epithelium points as a second population, just in case we want them later for vis
    domain.add_points(np.asarray([df_patient['x'][epi_mask],df_patient['y'][epi_mask]]).T,collection_name='Epithelium',zorder=0)
    
    # Boundary - first find the area of the alpha shape to estimate amount of tissue in domain
    domain.estimate_boundary(method='alpha shape',alpha_shape_kwargs={'alpha':500})
    PolygonArea = domain.boundary.area
    if PolygonArea / 1000000.0 < 0.8:
        assert(1==2)
        # This should never be reached as we've dropped ROIs with less than 80% tissue coverage
        continue
    # Reset the boundary
    bdy = [[xmin,ymin],[xmin+1000,ymin],[xmin+1000,ymin+1000],[xmin,ymin+1000]]
    domain.estimate_boundary(method='specify',specify_boundary_coords=bdy)

    # ms.visualise.visualise(domain,objects_to_plot=('collection','Epithelium'),figure_kwargs={'figsize':(12,8)},show_boundary=True)
    ms.visualise.visualise(domain,('Constant',[0.7,0.7,0.7,1]),marker_size=5)
    populations = ms.query.query(domain,('label','Celltype'),'in',[ct1,ct2])
    ms.visualise.visualise(domain,'Celltype',objects_to_plot=populations,ax=plt.gca(),marker_size=50,add_scalebar=True,show_boundary=True,scalebar_kwargs={'pad':-1})
    # plt.title(ROI)
    plt.savefig(f'{outpath}example_{ROI}.png')
    plt.savefig(f'{outpath}example_{ROI}.svg')
    plt.close()
    
    ms.visualise.visualise(domain,'Celltype',marker_size=50,add_scalebar=True,show_boundary=True,scalebar_kwargs={'pad':-1})
    plt.savefig(f'{outpath}example_{ROI}_all.png')
    plt.savefig(f'{outpath}example_{ROI}_all.svg')
    plt.close()
    
    
    # ms.visualise.visualise(domain,objects_to_plot=('collection','Epithelium'),figure_kwargs={'figsize':(12,8)},show_boundary=True)
    ms.visualise.visualise(domain,('Constant',[0.7,0.7,0.7,1]),marker_size=5,add_cbar=False)
    populations = ms.query.query(domain,('label','Celltype'),'in',[ct1,ct2])
    ms.visualise.visualise(domain,'Celltype',objects_to_plot=populations,ax=plt.gca(),marker_size=50,add_scalebar=False,add_cbar=False,show_boundary=True,scalebar_kwargs={'pad':-1})
    plt.gca().set_axis_off()
    plt.savefig(f'{outpath}example_{ROI}_nomarkup.png')
    plt.savefig(f'{outpath}example_{ROI}_nomarkup.svg')
    plt.close()

    population_A = ms.query.query(domain,('label','Celltype'),'is',ct2)
    population_B = ms.query.query(domain,('label','Celltype'),'is',ct1)

    r, g = ms.spatial_statistics.cross_pair_correlation_function(domain, population_A, population_B,max_R=150,annulus_step=5,annulus_width=10,visualise_output=False)
    # Bootstrap
    plt.figure(figsize=(10,5))
    # plt.gca().axhline(1,c='k',linestyle=':',lw=2)
    # Generate bootstrap
    if ROI.split('_')[1] == 'adenoma':
        c = 'b'
    else:
        c = 'r'
    plt.plot(r,g,c=c,lw=4)
    
    
    # --- Fill area under g(r) from 0 to R_int ---
    R_int = 40
    mask = (r >= 0) & (r <= R_int)
    # Fill the area under the curve between 0 and R_int with the same color and alpha=0.5
    plt.fill_between(r[mask], g[mask], y2=0, color=c, alpha=0.5)
    I = np.trapz(g[mask], r[mask])
    print('Area', I)
    
    # Optionally annotate the integral value near r=R_int
    # plt.annotate(fr'$I=\int_0^{R_int} g_{{{ct1[0]}{ct2[0]}}}(r)\,dr \approx {I:.3f}$',
    #              xy=(R_int, g[mask][-1]),
    #              xytext=(R_int + 5, g[mask][-1] + 0.5),
    #              arrowprops=dict(arrowstyle='->', color=c),
    #              color=c)
    
   
    # --- Add the LaTeX-style x-axis tick at R_int ---
    int_tick_label = rf'$\int_0^{{{R_int}}} g_{{{ct1[0]}{ct2[0]}}}(r)\,dr$'
    
    yticks = {}
    xticks = {}
    
    # End ticks you want to keep:
    xticks['0'] = 0
    xticks['150'] = 150
    yticks['0'] = 0
    yticks['3'] = 3
    
    # Annotations:
    yticks[rf'$g^{{Hi}}_{{{ct1[0]}{ct2[0]}}}$'] = float(np.max(g))
    plt.hlines(np.max(g), -50, r[np.argmax(g)], color='k', linestyle=':')
    
    yticks[rf'$g^{{Lo}}_{{{ct1[0]}{ct2[0]}}}$'] = float(np.min(g))
    plt.hlines(np.min(g), -50, r[np.argmin(g)], color='k', linestyle=':')
    
    xticks[rf'$r^{{Hi}}_{{{ct1[0]}{ct2[0]}}}$'] = float(r[np.argmax(g)])
    plt.vlines(r[np.argmax(g)], 0, np.max(g), color='k', linestyle=':')
    
    xticks[rf'$r^{{Lo}}_{{{ct1[0]}{ct2[0]}}}$'] = float(r[np.argmin(g)])
    plt.vlines(r[np.argmin(g)], 0, np.min(g), color='k', linestyle=':')
    
    xticks[int_tick_label] = float(R_int)
    
    # Interpolated vline height already set above:
    # plt.vlines(R_int, 0, g_Rint, color='k', linestyle=':')
    
    # --- Apply spacing and padding ---
    # Define helpers
    def spaced_ticks(ticks_dict, min_sep=4.0, delta=2.0):
        items = sorted(ticks_dict.items(), key=lambda kv: kv[1])
        positions = []
        labels = []
        last_pos = None
        for lbl, pos in items:
            new_pos = float(pos)
            if last_pos is not None and (new_pos - last_pos) < min_sep:
                new_pos = last_pos + min_sep
            positions.append(new_pos)
            labels.append(lbl)
            last_pos = new_pos
        return positions, labels
    
    # Compute spaced ticks
    xpos, xlab = spaced_ticks(xticks, min_sep=0, delta=0)
    ypos, ylab = spaced_ticks(yticks, min_sep=0, delta=0)
    
    plt.xticks(xpos, xlab)
    plt.yticks(ypos, ylab)
    
    # Pad tick labels slightly away from axes
    plt.tick_params(axis='x', which='both', pad=8)
    plt.tick_params(axis='y', which='both', pad=8)
    
    plt.xlabel('$r$')
    plt.ylabel('$g_{'+ct1[0]+ct2[0]+'}(r)$')
    plt.xlim([0,150])
    plt.ylim([0,3])
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{outpath}example_{ROI}_PCF.png')
    plt.savefig(f'{outpath}example_{ROI}_PCF.svg')
    print(xticks)
    print(yticks)

#%%
# Now make 40 x 5 table of PCF stats for Neutrophil / SMA for all patients
R_int = 40
pvals = pd.read_csv('./p-values_Final.csv')
stats = {'PCF_Neutrophil-SMA_gmax':rf'$g^{{Hi}}_{{{ct1[0]}{ct2[0]}}}$',
         'PCF_Neutrophil-SMA_rpeak':rf'$r^{{Hi}}_{{{ct1[0]}{ct2[0]}}}$',
         'PCF_Neutrophil-SMA_gmin':rf'$g^{{Lo}}_{{{ct1[0]}{ct2[0]}}}$',
         'PCF_Neutrophil-SMA_rtrough':rf'$r^{{Lo}}_{{{ct1[0]}{ct2[0]}}}$',
         'PCF_Neutrophil-SMA_int0_40':rf'$\int_0^{{{R_int}}} g_{{{ct1[0]}{ct2[0]}}}(r)\,dr$'
         }

# filter
p5 = pvals[pvals['Statistic'].isin(stats.keys())].copy()

# If there can be multiple rows per (ID, stat), pick an aggregation rule:
# e.g., take the first, or the mean. Here we take the first non-null pvalue.
# If your data has unique pairs already, pivot_table will just pass values through.
pivot_df_p = (p5
    .pivot_table(index='Statistic', columns='ID', values='pvalue')#, aggfunc='first')
    # enforce the desired row order
    .reindex(index=list(stats.keys()))
)

# Optional: sort patient ID columns for consistent left-to-right ordering
pivot_df_p = pivot_df_p.reindex(sorted(pivot_df_p.columns), axis=1)

# Map the index from internal stat keys to your LaTeX labels for display
pivot_df_p.index = [stats[k] for k in pivot_df_p.index]

# --- Plot the heatmap
plt.figure(figsize=(30,5))  # scale width with #patients
# Make a copy of the base colormap and set an "over" color (choose one that stands out)
cmap = mpl.cm.get_cmap('Oranges_r').copy()
cmap.set_over([0.9,0.9,0.9,1])   # e.g., a contrasting purple; pick any color you like (hex or named)
ax = sns.heatmap(
    pivot_df_p,
    cmap=cmap,
    linewidths=0.5,
    linecolor='white',
    # cbar_kws={'label': 'p-value'},
    cbar_kws={'label': 'p-value', 'extend': 'max'},
    vmin=0.0, vmax=0.05       # p-values are typically in [0,1]
)

# Cosmetic tweaks
ax.set_xlabel('Patient ID')
ax.set_ylabel('Statistic')
ax.set_title('p-values for cross-PCF summary statistics; Neutrophil-SMA')

# Make x tick labels readable
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(f'{outpath}pvals_all_patients.png')
plt.savefig(f'{outpath}pvals_all_patients.svg')   
    

# Now do percentage change in median statistic
# If there can be multiple rows per (ID, stat), pick an aggregation rule:
# e.g., take the first, or the mean. Here we take the first non-null pvalue.
# If your data has unique pairs already, pivot_table will just pass values through.
pivot_df1 = (p5
    .pivot_table(index='Statistic', columns='ID', values='median_adenoma')#, aggfunc='first')
    # enforce the desired row order
    .reindex(index=list(stats.keys()))
)

pivot_df2 = (p5
    .pivot_table(index='Statistic', columns='ID', values='median_carcinoma')#, aggfunc='first')
    # enforce the desired row order
    .reindex(index=list(stats.keys()))
)

# Make this a percentage change in starting value
pivot_df = 100*(pivot_df2 - pivot_df1)/pivot_df1

# Optional: sort patient ID columns for consistent left-to-right ordering
pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

# Map the index from internal stat keys to your LaTeX labels for display
pivot_df.index = [stats[k] for k in pivot_df.index]

# Drop columns where tests couldn't be conducted (i.e., patients with all ad or all car)
x = list(pivot_df_p.keys())
pivot_df = pivot_df[x]

# --- Plot the heatmap
plt.figure(figsize=(30,5))  # scale width with #patients
ax = sns.heatmap(
    pivot_df,
    cmap='PRGn_r',          # 'coolwarm' or 'magma' are also common
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '% change in metric'},
    vmin=-100, vmax=100       # p-values are typically in [0,1]
)

# Cosmetic tweaks
ax.set_xlabel('Patient ID')
ax.set_ylabel('Statistic')
ax.set_title('Change in cross-PCF summary statistics; Neutrophil-SMA')

# Make x tick labels readable
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(f'{outpath}scalar_change_all_patients.png')
plt.savefig(f'{outpath}scalar_change_all_patients.svg')  


# Now plot statistically significant ones only 
pdf = pivot_df[pivot_df_p < 0.05]
# --- Plot the heatmap
plt.figure(figsize=(30,5))  # scale width with #patients
ax = sns.heatmap(
    pdf,
    cmap='PRGn_r',          # 'coolwarm' or 'magma' are also common
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '% change in metric'},
    vmin=-100, vmax=100       # p-values are typically in [0,1]
)

# Cosmetic tweaks
ax.set_xlabel('Patient ID')
ax.set_ylabel('Statistic')
ax.set_title('Change in cross-PCF summary statistics; Neutrophil-SMA')

# Make x tick labels readable
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(f'{outpath}scalar_change_all_patients_significant.png')
plt.savefig(f'{outpath}scalar_change_all_patients_significant.svg')  

#%% Now plot PCFs for patient 28717 and 25914
path_to_points = 'path/to/AllPointcloudsAsCSVs_central/'
path_to_ROI_coords = 'path/to/AllPointcloudsAsCSVs_central_coordinates.csv'
df_coords = pd.read_csv(path_to_ROI_coords)
(ct1, ct2) = ('Neutrophil','SMA')

for ID in [28717, 25914]:
    minidf = df[df['sampleID']==ID]
    
    gs = []
    diseasetypes = []
    colarray = []
    nct1s = []
    nct2s = []
    skipped = []
    
    
    for i in range(len(minidf)):
        print(i, len(minidf))
        # Read in the ROI
        the_file = minidf.iloc[i].Name
        ROI = the_file.split('.')[0]
        xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
        ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
        df_patient = pd.read_csv(f'{path_to_points}{the_file}.csv')
        
        outline = {'Name':ROI,
                   'sampleID':ROI.split('_')[0],
                   'disease':ROI.split('_')[1],
                   'quadratLabel':int(ROI.split('_')[2].split('-')[1]),
                   'xMin':xmin,
                   'yMin':ymin}
        
        domain = ms.domain(ROI)
        # First add only non-Epithelium points
        epi_mask = np.array([v.startswith('E') for v in df_patient['Celltype']])
        domain.add_points(np.asarray([df_patient['x'][~epi_mask],df_patient['y'][~epi_mask]]).T,collection_name='Cells')
        domain.add_labels('Celltype',df_patient['Celltype'][~epi_mask])
        cols = {v:celltypes[v] for v in celltypes if v in np.unique(df_patient.Celltype)}
        domain.update_colors(cols,label_name='Celltype')
        
        # Add epithelium points as a second population, just in case we want them later for vis
        domain.add_points(np.asarray([df_patient['x'][epi_mask],df_patient['y'][epi_mask]]).T,collection_name='Epithelium',zorder=0)
        # Reset the boundary
        bdy = [[xmin,ymin],[xmin+1000,ymin],[xmin+1000,ymin+1000],[xmin,ymin+1000]]
        domain.estimate_boundary(method='specify',specify_boundary_coords=bdy)
    
        # Cross PCF
        population_A = ms.query.query(domain,('label','Celltype'),'is',ct2)
        population_B = ms.query.query(domain,('label','Celltype'),'is',ct1)
        nct1 = len(ms.query.interpret_query(population_B))
        nct2 = len(ms.query.interpret_query(population_A))
        if (nct1 < 20) or (nct2 < 20):
            # Skip ROIs with low cell counts
            skipped.append(i)
            continue
        r, g = ms.spatial_statistics.cross_pair_correlation_function(domain, population_A, population_B,max_R=150,annulus_step=5,annulus_width=10,visualise_output=False)
        gs.append(g)
        diseasetypes.append(ROI.split('_')[1])
        if ROI.split('_')[1] == 'adenoma':
            c = 'b'
        else:
            c = 'r'
        colarray.append(c)
        nct1s.append(nct1)
        nct2s.append(nct2)
    #%
    gs = np.asarray(gs)
    diseasetypes = np.array(diseasetypes)
    nct1s = np.array(nct1s)
    nct2s = np.array(nct2s)
    
    
    # Bootstrap
    plt.figure(figsize=(10,5))
    plt.gca().axhline(1,c='k',linestyle=':',lw=2)
    admask = diseasetypes == 'adenoma'
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
    plt.ylim([0,2.3])
    plt.tight_layout()
    plt.savefig(f'{outpath}PCF_{ID}.png')
    plt.savefig(f'{outpath}PCF_{ID}.svg')



#%%
cm = plt.cm.bone_r
minidf = stats_df[stats_df.sampleID == ID]
patient_df = []
for i in range(len(minidf)):
    the_file = minidf.iloc[i].Name
    ROI = the_file.split('.')[0]
    xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
    ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
    patient_dict = {'disease': ROI.split('_')[1],
                    'x' : xmin,
                    'y' : ymin,
                    'isROI' : ROI in to_plot}
    patient_df.append(patient_dict)
patient_df = pd.DataFrame(patient_df)

plt.figure()
diseasecols = {'adenoma':'b','cancer':'r'}
rectangles = []

ecs = []
zs = []
for i in range(len(patient_df)):
    row = patient_df.iloc[i]
    if row['isROI']:
        ecs.append('k')
        z = 10000
    else:
        ecs.append(diseasecols[row['disease']])
        z = 0
    rect = Rectangle((row['x'], row['y']), 1000, 1000, zorder=z,edgecolor='k')
    rectangles.append(rect)
    
norm = mpl.colors.Normalize(vmin=0,vmax=1)
coll = PatchCollection(
rectangles, cmap=cm, array=patient_df['isROI']
)
coll.set_edgecolors(ecs)
coll.set_linewidths(3)
plt.gca().add_collection(coll)
plt.xlim([patient_df.x.min(),patient_df.x.max()])
plt.ylim([patient_df.y.min(),patient_df.y.max()])
plt.gca().axis('equal')
plt.gca().axis(False)
plt.title(ID)

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
scalebar = AnchoredSizeBar(plt.gca().transData,
                           5000, '5000 pixels', 'lower left', 
                           pad=-0.5,
                           color='black',
                           frameon=False,
                           size_vertical=500)

plt.gca().add_artist(scalebar)
plt.savefig(f'{outpath}ID_{ID}.png')
plt.savefig(f'{outpath}ID_{ID}.svg')
