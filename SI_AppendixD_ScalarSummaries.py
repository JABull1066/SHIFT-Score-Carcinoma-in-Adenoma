import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", font_scale=2)
import muspan as ms
import os
import pandas as pd

outpath = './SI_AppendixD_ScalarSummaries/'
#%% Load data
path_to_points = 'path/to/folder/AllPointcloudsAsCSVs_central/'
path_to_ROI_coords = 'path/to/folder/AllPointcloudsAsCSVs_central_coordinates.csv'
path_to_summary_statistics = 'path/to/SummaryStatistics_Final.csv'
summary_stats_df = pd.read_csv(path_to_summary_statistics)

files = os.listdir(path_to_points)
df_coords = pd.read_csv(path_to_ROI_coords)
colors = {'CD146':plt.cm.tab10(0),
          'CD34':plt.cm.tab10(1),
          'Cytotoxic T Cell':plt.cm.tab10(2),
          'Macrophage':plt.cm.tab10(9),
          'Neutrophil':plt.cm.tab10(4),
          'Periostin':plt.cm.tab10(6),
          'Podoplanin':plt.cm.tab10(5),
          'SMA':plt.cm.tab10(8),
          'T Helper Cell':plt.cm.tab10(3),
          'Treg Cell':plt.cm.tab20b(0)}

cells = [v for v in colors if not v.startswith('Epi')] # Exclude epithelial cells from pairwise analysis

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


#%% TDA: n Loops
ct1 = 'Periostin'
stat_to_text = get_stat_to_text(ct1,ct1)
stat = f'PH_{ct1}_nLoops'
legend = stat_to_text['PH_nLoops']


order = np.argsort(summary_stats_df[stat].dropna())
tocheck = [v for v in order.iloc[0:10]]
tocheck.extend([v for v in order.iloc[-10:]])
roi_indices = [summary_stats_df[stat].dropna().index[v] for v in tocheck]
rois = [summary_stats_df.loc[v].Name for v in roi_indices]
nloops = [summary_stats_df.loc[v][stat] for v in roi_indices]

to_plot = ['19606_cancer_ID-243','23559_cancer_ID-333']

for i, ROI in enumerate(to_plot):
    the_file = f'{ROI}.csv'
    xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
    ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
    df = pd.read_csv(f'{path_to_points}{the_file}')

    domain = ms.domain(ROI,'pixels')
    # First add only non-Epithelium points
    epi_mask = np.array([v.startswith('E') for v in df['Celltype']])
    if np.sum(~epi_mask) < 5:
        continue
    if np.sum(epi_mask) < 5:
        continue
    domain.add_points(np.asarray([df['x'][~epi_mask],df['y'][~epi_mask]]).T,collection_name='Cells')
    domain.add_labels('Celltype',list(df['Celltype'][~epi_mask]))
    cols = {v:colors[v] for v in colors if v in np.unique(df.Celltype)}
    domain.update_colors(cols,label_name='Celltype')
    
    # Add epithelium points as a second population, just in case we want them later for vis
    domain.add_points(np.asarray([df['x'][epi_mask],df['y'][epi_mask]]).T,collection_name='Epithelium',zorder=0)
    
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
    population_A = ('Celltype',ct1)
    ms.visualise.visualise(domain,'Celltype',objects_to_plot=population_A,ax=plt.gca(),marker_size=50,add_scalebar=True,show_boundary=True,scalebar_kwargs={'pad':-1})
    # plt.title(ROI)
    plt.savefig(f'{outpath}PH_example_{ROI}_{stat}.png')
    plt.savefig(f'{outpath}PH_example_{ROI}_{stat}.svg')
    plt.close()

plt.figure(figsize=(8,6))
mask = ~np.isnan(summary_stats_df[stat])
cols = summary_stats_df['disease'][mask]
vs = []
for j, i in enumerate(['adenoma','cancer']):
    v = summary_stats_df[stat][mask][cols == i]
    vs.append(v)
plt.hist(vs,bins=np.arange(summary_stats_df[stat].dropna().max()),color=['b','r'],stacked=True)
plt.savefig(f'{outpath}PH_example_{stat}_histogram_c.png')
plt.savefig(f'{outpath}PH_example_{stat}_histogram_c.svg')
plt.close()
#%% Wasserstein Distance
ct1 = 'Neutrophil'
ct2 = 'SMA'

stat_to_text = get_stat_to_text(ct1,ct2)
stat = f'WassersteinDistance_{ct1}-{ct2}'
legend = stat_to_text['WassersteinDistance']

order = np.argsort(summary_stats_df[stat].dropna())
tocheck = [v for v in order.iloc[0:10]]
tocheck.extend([v for v in order.iloc[-10:]])
roi_indices = [summary_stats_df[stat].dropna().index[v] for v in tocheck]
rois = [summary_stats_df.loc[v].Name for v in roi_indices]
statval = [summary_stats_df.loc[v][stat] for v in roi_indices]

to_plot = ['5531_adenoma_ID-38', '25914_cancer_ID-129']

for i, ROI in enumerate(to_plot):
    the_file = f'{ROI}.csv'
    xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
    ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
    df = pd.read_csv(f'{path_to_points}{the_file}')

    domain = ms.domain(ROI,'pixels')
    # First add only non-Epithelium points
    epi_mask = np.array([v.startswith('E') for v in df['Celltype']])
    if np.sum(~epi_mask) < 5:
        continue
    if np.sum(epi_mask) < 5:
        continue
    domain.add_points(np.asarray([df['x'][~epi_mask],df['y'][~epi_mask]]).T,collection_name='Cells')
    domain.add_labels('Celltype',list(df['Celltype'][~epi_mask]))
    cols = {v:colors[v] for v in colors if v in np.unique(df.Celltype)}
    domain.update_colors(cols,label_name='Celltype')
    
    # Add epithelium points as a second population, just in case we want them later for vis
    domain.add_points(np.asarray([df['x'][epi_mask],df['y'][epi_mask]]).T,collection_name='Epithelium',zorder=0)
    
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
    population_A = ('Celltype',ct1)
    population_B = ('Celltype',ct2)
    q = ms.query.query_container(population_A,'OR',population_B,domain=domain)
    ms.visualise.visualise(domain,'Celltype',objects_to_plot=q,ax=plt.gca(),marker_size=50,add_scalebar=True,show_boundary=True,scalebar_kwargs={'pad':-1})
    # plt.title(ROI)
    plt.savefig(f'{outpath}Wasserstein_example_{ROI}_{stat}.png')
    plt.savefig(f'{outpath}Wasserstein_example_{ROI}_{stat}.svg')
    plt.close()




plt.figure(figsize=(8,6))
mask = ~np.isnan(summary_stats_df[stat])
cols = summary_stats_df['disease'][mask]
vs = []
for j, i in enumerate(['adenoma','cancer']):
    v = summary_stats_df[stat][mask][cols == i]
    vs.append(v)
plt.hist(vs,bins=np.arange(0,summary_stats_df[stat].dropna().max(),10),color=['b','r'],stacked=True)
plt.savefig(f'{outpath}Wasserstein_example_{stat}_histogram_c.png')
plt.savefig(f'{outpath}Wasserstein_example_{stat}_histogram_c.svg')
plt.close()
#%% Cross-PCF
ct2 = 'Treg Cell'
ct1 = 'Cytotoxic T Cell'
stat_to_text = get_stat_to_text(ct1,ct2)
stat = f'PCF_{ct1}-{ct2}_int0_20'
legend = stat_to_text['PCF_int0_20']

order = np.argsort(summary_stats_df[stat].dropna())
tocheck = [v for v in order.iloc[0:15]]
tocheck.extend([v for v in order.iloc[-15:]])
roi_indices = [summary_stats_df[stat].dropna().index[v] for v in tocheck]
rois = [summary_stats_df.loc[v].Name for v in roi_indices]
statval = [summary_stats_df.loc[v][stat] for v in roi_indices]

to_plot = ['25914_adenoma_ID-159', '5531_cancer_ID-118']

for i, ROI in enumerate(to_plot):
    the_file = f'{ROI}.csv'
    xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
    ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
    df = pd.read_csv(f'{path_to_points}{the_file}')

    domain = ms.domain(ROI,'pixels')
    # First add only non-Epithelium points
    epi_mask = np.array([v.startswith('E') for v in df['Celltype']])
    if np.sum(~epi_mask) < 5:
        continue
    if np.sum(epi_mask) < 5:
        continue
    domain.add_points(np.asarray([df['x'][~epi_mask],df['y'][~epi_mask]]).T,collection_name='Cells')
    domain.add_labels('Celltype',list(df['Celltype'][~epi_mask]))
    cols = {v:colors[v] for v in colors if v in np.unique(df.Celltype)}
    domain.update_colors(cols,label_name='Celltype')
    
    # Add epithelium points as a second population, just in case we want them later for vis
    domain.add_points(np.asarray([df['x'][epi_mask],df['y'][epi_mask]]).T,collection_name='Epithelium',zorder=0)
    
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
    population_A = ('Celltype',ct1)
    population_B = ('Celltype',ct2)
    q = ms.query.query_container(population_A,'OR',population_B,domain=domain)
    ms.visualise.visualise(domain,'Celltype',objects_to_plot=q,ax=plt.gca(),marker_size=50,add_scalebar=True,show_boundary=True,scalebar_kwargs={'pad':-1})
    # plt.title(ROI)
    plt.savefig(f'{outpath}PCF_example_{ROI}.png')
    plt.savefig(f'{outpath}PCF_example_{ROI}.svg')
    plt.close()


plt.figure(figsize=(8,6))
mask = ~np.isnan(summary_stats_df[stat])
cols = summary_stats_df['disease'][mask]
vs = []
for j, i in enumerate(['adenoma','cancer']):
    v = summary_stats_df[stat][mask][cols == i]
    vs.append(v)
plt.hist(vs,bins=np.arange(0,summary_stats_df[stat].dropna().max(),10),color=['b','r'],stacked=True)
plt.savefig(f'{outpath}PCF_example_{stat}_histogram_c.png')
plt.savefig(f'{outpath}PCF_example_{stat}_histogram_c.svg')
plt.close()

#%% QCM
ct2 = 'Periostin'
ct1 = 'Macrophage'
stat_to_text = get_stat_to_text(ct1,ct2)
stat = f'QCM_{ct1}-{ct2}'
legend = stat_to_text['QCM']

order = np.argsort(summary_stats_df[stat].dropna())
tocheck = [v for v in order.iloc[0:5]]
tocheck.extend([v for v in order.iloc[-5:]])
roi_indices = [summary_stats_df[stat].dropna().index[v] for v in tocheck]
rois = [summary_stats_df.loc[v].Name for v in roi_indices]
statval = [summary_stats_df.loc[v][stat] for v in roi_indices]

to_plot = ['25914_cancer_ID-292','25914_cancer_ID-386']
for i, ROI in enumerate(to_plot):
    the_file = f'{ROI}.csv'
    xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
    ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
    df = pd.read_csv(f'{path_to_points}{the_file}')

    domain = ms.domain(ROI,'pixels')
    # First add only non-Epithelium points
    epi_mask = np.array([v.startswith('E') for v in df['Celltype']])
    if np.sum(~epi_mask) < 5:
        continue
    if np.sum(epi_mask) < 5:
        continue
    domain.add_points(np.asarray([df['x'][~epi_mask],df['y'][~epi_mask]]).T,collection_name='Cells')
    domain.add_labels('Celltype',list(df['Celltype'][~epi_mask]))
    cols = {v:colors[v] for v in colors if v in np.unique(df.Celltype)}
    domain.update_colors(cols,label_name='Celltype')
    
    # Add epithelium points as a second population, just in case we want them later for vis
    domain.add_points(np.asarray([df['x'][epi_mask],df['y'][epi_mask]]).T,collection_name='Epithelium',zorder=0)
    
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
    population_A = ('Celltype',ct1)
    population_B = ('Celltype',ct2)
    q = ms.query.query_container(population_A,'OR',population_B,domain=domain)
    ms.visualise.visualise(domain,'Celltype',objects_to_plot=q,ax=plt.gca(),marker_size=50,add_scalebar=True,show_boundary=True,scalebar_kwargs={'pad':-1})
    plt.title(ROI)
    plt.savefig(f'{outpath}QCM_example_{ROI}.png')
    plt.savefig(f'{outpath}QCM_example_{ROI}.svg')
    plt.close()


plt.figure(figsize=(8,6))
mask = ~np.isnan(summary_stats_df[stat])
cols = summary_stats_df['disease'][mask]
vs = []
for j, i in enumerate(['adenoma','cancer']):
    v = summary_stats_df[stat][mask][cols == i]
    vs.append(v)
plt.hist(vs,bins=np.arange(summary_stats_df[stat].dropna().min(),summary_stats_df[stat].dropna().max(),0.25),color=['b','r'],stacked=True)
plt.savefig(f'{outpath}QCM_example_{stat}_histogram_c.png')
plt.savefig(f'{outpath}QCM_example_{stat}_histogram_c.svg')
plt.close()




