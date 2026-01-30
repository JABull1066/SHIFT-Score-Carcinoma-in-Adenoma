import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import muspan as ms
sns.set_theme(style='white',font_scale=2)

path_to_points = 'path/to/AllPointcloudsAsCSVs_central/'
path_to_ROI_coords = 'path/to/AllPointcloudsAsCSVs_central_coordinates.csv'
path_to_save_outputs = './outputs/'

files = os.listdir(path_to_points)
df_coords = pd.read_csv(path_to_ROI_coords)
#%%
colors = {'CD146':plt.cm.tab10(0),
          'CD34':plt.cm.tab10(1),
          'Cytotoxic T Cell':plt.cm.tab10(2),
          'Macrophage':plt.cm.tab10(3),
          'Neutrophil':plt.cm.tab10(4),
          'Periostin':plt.cm.tab10(5),
          'Podoplanin':plt.cm.tab10(6),
          'SMA':plt.cm.tab10(8),
          'T Helper Cell':plt.cm.tab10(9),
          'Treg Cell':plt.cm.tab20b(0)}

def get_n_loops(dgm, do_plot=False):
    rs = dgm[:,1]-dgm[:,0]
    threshold_radius = 20
    
    # Critical r
    critical_r = threshold_radius*np.sqrt(3)
    plotmask = rs > critical_r
    n_loops = np.sum(plotmask)
        
    if do_plot:
        plt.figure(figsize=(10,8))
        ax = plt.gca()
        ax.scatter(dgm[plotmask,0],dgm[plotmask,1],edgecolor='black',linewidth=0.5,s=50,alpha=0.7,label='Is a loop')
        ax.scatter(dgm[~plotmask,0],dgm[~plotmask,1],edgecolor='black',linewidth=0.5,s=50,alpha=0.7,label='Isn\'t a loop')
        ax.axis('equal')
        ax.plot([0,np.max(dgm)],[0,np.max(dgm)],linestyle='--',color=[0.7,0.7,0.7,1],linewidth=2,zorder=0)
        ax.set_ylabel('Death')
        ax.set_xlabel('Birth')
        ax.legend(loc='lower right')

    return n_loops

def get_stats_for_file(the_file_index):
    the_file = files[the_file_index]
    ROI = the_file.split('.')[0]
    xmin = int(df_coords[df_coords['name'] == ROI]['lower x'].iloc[0])
    ymin = int(df_coords[df_coords['name'] == ROI]['lower y'].iloc[0])
    df = pd.read_csv(f'{path_to_points}{the_file}')
    
    outline = {'Name':ROI,
               'sampleID':ROI.split('_')[0],
               'disease':ROI.split('_')[1],
               'quadratLabel':int(ROI.split('_')[2].split('-')[1]),
               'xMin':xmin,
               'yMin':ymin}
    
    domain = ms.domain(ROI)
    # First add only non-Epithelium points
    epi_mask = np.array([v.startswith('E') for v in df['Celltype']])
    domain.add_points(np.asarray([df['x'][~epi_mask],df['y'][~epi_mask]]).T,collection_name='Cells')
    domain.add_labels('Celltype',df['Celltype'][~epi_mask])
    cols = {v:colors[v] for v in colors if v in np.unique(df.Celltype)}
    domain.update_colors(cols,label_name='Celltype')
    
    # Add epithelium points as a second population, just in case we want them later for vis
    domain.add_points(np.asarray([df['x'][epi_mask],df['y'][epi_mask]]).T,collection_name='Epithelium',zorder=0)
    
    # Boundary - first find the area of the alpha shape to estimate amount of tissue in domain
    domain.estimate_boundary(method='alpha shape',alpha_shape_kwargs={'alpha':500})
    PolygonArea = domain.boundary.area
    outline['PolygonArea'] = PolygonArea
    outline['DomainArea'] = 1000000 # 1000*1000
    outline['DomainTissueFraction'] = PolygonArea / 1000000.0
    if outline['DomainTissueFraction'] < 0.8:
        # This should never be reached as we've dropped ROIs with less than 80% tissue coverage
        return outline
    # Reset the boundary
    bdy = [[xmin,ymin],[xmin+1000,ymin],[xmin+1000,ymin+1000],[xmin,ymin+1000]]
    domain.estimate_boundary(method='specify',specify_boundary_coords=bdy)
    
    # ms.visualise.visualise(domain,'Celltype',objects_to_plot=('collection','Cells'),figure_kwargs={'figsize':(12,8)},show_boundary=True)
    fig, ax = plt.subplots(figsize=(12,8))
    ms.visualise.visualise(domain,'Celltype',ax=ax,show_boundary=True)
    plt.savefig(f'{path_to_save_outputs}ROI_images/{ROI}.png')
    plt.close()
    
    
    # Run pipeline
    # SUMMARY STATS
    count_dict = {v:0.0 for v in colors}
    counts, cts = ms.summary_statistics.label_counts(domain,'Celltype',normalised=False)
    
    # CELL COUNTS
    for i in range(len(cts)):
        count_dict[cts[i]] = float(counts[i])
    for ct in colors:
        outline[f'Count_{ct}'] = count_dict[ct]
        
    # PERSISTENT HOMOLOGY
    for i in range(len(cts)):        
        if float(counts[i]) > 20:
            # Standard PH on this point population alone
            TDA_dict = ms.topology.vietoris_rips_filtration(domain,population=('Celltype',cts[i]))
            vec, stats = ms.topology.vectorise_persistence(TDA_dict)
            for j in range(len(vec)):
                statname = '-'.join(stats[j].split(' '))
                outline[f'PH_{cts[i]}_{statname}'] = vec[j]
            outline[f'PH_{cts[i]}_nLoops'] = get_n_loops(TDA_dict['dgms'][1])
            
        
    # QCM
    SES, A, cats = ms.region_based.quadrat_correlation_matrix(domain,'Celltype',population=('Collection','Cells'),region_kwargs={'side_length':100},low_observation_bound=20)
    # Stats which are symmetric
    for i, cta in enumerate(cats):
        for j, ctb in enumerate(cats):
            if i <= j:
                population_A = ms.query.query(domain,('label','Celltype'),'is',cta)
                population_B = ms.query.query(domain,('label','Celltype'),'is',ctb)
                if i != j:
                    # Don't do QCM or Wasserstein for same cell pops
                    outline[f'QCM_{cta}-{ctb}'] = SES[i,j]
                    # WASSERSTEIN
                    w = ms.distribution.sliced_wasserstein_distance(domain, population_A, population_B)
                    outline[f'WassersteinDistance_{cta}-{ctb}'] = w
        
                # CROSS PCF
                r, g = ms.spatial_statistics.cross_pair_correlation_function(domain, population_A, population_B,max_R=150,annulus_step=5,annulus_width=10,visualise_output=False)
        
                outline[f'PCF_{cta}-{ctb}_gmax'] = g[np.nanargmax(g)]
                outline[f'PCF_{cta}-{ctb}_rpeak'] = r[np.nanargmax(g)]
                outline[f'PCF_{cta}-{ctb}_gmin'] = g[np.nanargmin(g)]
                outline[f'PCF_{cta}-{ctb}_rtrough'] = r[np.nanargmin(g)]
                outline[f'PCF_{cta}-{ctb}_int0_20'] = np.trapz(g[0:5],r[0:5])
                outline[f'PCF_{cta}-{ctb}_int0_40'] = np.trapz(g[0:9],r[0:9])
                outline[f'PCF_{cta}-{ctb}_int0_100'] = np.trapz(g[0:21],r[0:21])
                
    
    # New loop for TCMs, as they aren't symmetric
    cats = [v for v in cols]
    for cta in cats:
        population_A = ms.query.query(domain,('label','Celltype'),'is',cta)
        na = len(ms.query.interpret_query(population_A))
        if na < 20:
            continue
        for ctb in cats:
            # TCM
            population_B = ms.query.query(domain,('label','Celltype'),'is',ctb)
            nb = len(ms.query.interpret_query(population_B))
            if nb < 20:
                continue
            TCM = ms.spatial_statistics.topographical_correlation_map(domain, population_A, population_B,radius_of_interest=50, kernel_radius=150, kernel_sigma=50,visualise_output=False)
            lsf = ms.topology.level_set_filtration(TCM,visualise_output=False)
            vec, stats = ms.topology.vectorise_persistence(lsf)
            for j in range(len(vec)):
                statname = '-'.join(stats[j].split(' '))
                outline[f'TCM_{cta}-{ctb}_{statname}'] = vec[j]
    plt.close('all')
    return outline
    
#%%
n_files_per_loop = 16
failed_batches = []
vals = range(int(len(files)/n_files_per_loop))
for j in vals:
    if not os.path.isfile(f'{path_to_save_outputs}temp_data_to_collate/Batch_{j}.csv'):
        print(f'\nBeginning Loop {j}')
        results = []
        for i in range(j*n_files_per_loop,(j+1)*n_files_per_loop):
            print(i, (j+1)*n_files_per_loop, files[i])
            tocheck = files[i].split('.')[0]
            # Only repeat files which we haven't done before
            # if not os.path.isfile(f'{path_to_save_outputs}ROI_images/{tocheck}.png'):
            try:
                out = get_stats_for_file(i) 
                results.append(out)
            except:
                print(f'Failed {files[i]}')
            plt.close('all')
        df_temp = pd.DataFrame(results)
        df_temp.to_csv(f'{path_to_save_outputs}temp_data_to_collate/Batch_{j}.csv')
        print(f'Loop {j} (index {j*n_files_per_loop} to {(j+1)*n_files_per_loop-1}) complete')
    plt.close('all')

#%%
# Now stick together all the Batch_ files
batch_files = os.listdir(f'{path_to_save_outputs}temp_data_to_collate/')
dfs = []
for batch in batch_files:
    df_temp = pd.read_csv(f'{path_to_save_outputs}temp_data_to_collate/{batch}')
    dfs.append(df_temp)
#%%
df = pd.concat(dfs,axis=0)
# Drop the first column of indices
df = df.drop('Unnamed: 0',axis=1)
df.to_csv('SummaryStatistics_Final.csv')

    
    
    
    
    

