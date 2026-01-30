import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import muspan as ms
sns.set(style="white",font_scale=2)

#%% Helpers
colors = {'A':plt.cm.tab10(0),
          'B':plt.cm.tab10(1),
          'C':plt.cm.tab10(2),
          'D':[0.7,0.7,0.7,1]}

outpath = './Fig1_Components/'

#%% Set up the point clouds
np.random.seed(130357725) #fix seed
# Density
n=500

PointsA=1000*np.random.rand(n,2)
PointsB=1000*np.random.rand(n,2)
PointsC=1000*np.random.rand(n,2)
PointsD=1000*np.random.rand(n,2)

points=np.vstack([PointsA,PointsB,PointsC,PointsD])

typesA=['A']*n
typesB=['B']*n
typesC=['C']*n
typesD=['D']*n

labels1=typesA+typesB+typesC+typesD

pc1 = ms.domain('Unstructured')
pc1.add_points(points)
pc1.add_labels('Celltype', labels1)
pc1.update_colors(colors,label_name='Celltype')


# Aggregation
seeds = [[200,200],[800,200],[500,500],[200,800],[800,800]]
RoiNoise= 1000*np.random.rand(n,2)

sigma=50
labels2=['D']*n

points2=RoiNoise
ns = {'A':[50,50,100,150,150],'B':[50,50,100,150,150],'C':[150,150,100,50,50]}
for label in ['A','B','C']:
    for i in range(len(seeds)):
        n_temp = ns[label][i]
        tempPoints = np.random.randn(n_temp,2)*sigma + seeds[i]
        tempLabs = [label]*n_temp
        labels2 = labels2 + tempLabs
        points2 = np.vstack([points2,tempPoints])
    
pc2 = ms.domain('Aggregation')
pc2.add_points(points2)
pc2.add_labels('Celltype', labels2)
pc2.update_colors(colors,label_name='Celltype')

# Exclusion
#some functions to define the stromal river
def upperLine(x):
    return -x+1300+100*np.cos(0.01*x)

def lowerLine(x):
    return -x+700 +100*np.sin(0.01*x)

def upperLineImm(x):
    return -x+1200+100*np.cos(0.01*x)

def lowerLineImm(x):
    return -x+840+100*np.sin(0.01*x)

# Sample a huge volume of points at random
all_pts = 1000*np.random.rand(10000,2)
mask = np.zeros(len(all_pts))

maskEpi = (all_pts[:,1] < upperLineImm(all_pts[:,0])) & (all_pts[:,1] > lowerLineImm(all_pts[:,0]))
# First 1000 inside and outside
epi_ind = np.array([v for v in range(len(maskEpi)) if maskEpi[v]])[0:1000]
stro_ind = np.array([v for v in range(len(maskEpi)) if not maskEpi[v]])[0:500]
rand = 1000*np.random.rand(500,2)
points3=all_pts[np.hstack((epi_ind,stro_ind)),:]
points3 = np.vstack((points3,rand))
labels3=['A']*500 + ['B']*500 + ['C']*500 + ['D']*500

pc3 = ms.domain('Exclusion')
pc3.add_points(points3)
pc3.add_labels('Celltype', labels3)
pc3.update_colors(colors,label_name='Celltype')


# Architecture
cryptCentersX=[100,300, 450, 600,700,650,500,300,850,900,825,120,500,175,520,900,300,90,620,60]
cryptCentersY=[200,300, 450, 600,800,400,250,100,550,775,930,60,825,450,65,380,945,900,920,360]
npercrypt = int(n/len(cryptCentersX))
nperblob = 250

Xpoints=np.array([])
Ypoints=np.array([])
labels = []
for i in range(len(cryptCentersX)):
    r=30 + 30*np.random.rand(2*npercrypt)
    theta=2*np.pi*np.random.rand(2*npercrypt)
    Xpoints=np.append(Xpoints,[r*np.cos(theta) +cryptCentersX[i]])
    Ypoints=np.append(Ypoints,[r*np.sin(theta)+cryptCentersY[i]])
    labels.extend(['A']*npercrypt)
    labels.extend(['B']*npercrypt)

# Make big blobs, filter down
blob1X = 500*np.random.rand(20*nperblob)+50
blob1Y = 500*np.random.rand(20*nperblob)+400
mask1=np.zeros(20*nperblob,dtype=bool)
for i in range(20*nperblob):
    mask1[i] = (np.sqrt(((blob1X[i]-275)**2 + (blob1Y[i]-680)**2))<(120 + 80*np.random.rand(1)))
Xpoints=np.append(Xpoints,blob1X[mask1][0:nperblob])
Ypoints=np.append(Ypoints,blob1Y[mask1][0:nperblob])
labels.extend(['C']*nperblob)
# labels.extend(['D']*nperblob)

blob1X = 500*np.random.rand(20*nperblob)+600
blob1Y = 500*np.random.rand(20*nperblob)+0
mask1=np.zeros(20*nperblob,dtype=bool)
for i in range(20*nperblob):
    mask1[i] = (np.sqrt(((blob1X[i]-800)**2 + (blob1Y[i]-150)**2))<(120 + 80*np.random.rand(1)))
Xpoints=np.append(Xpoints,blob1X[mask1][0:nperblob])
Ypoints=np.append(Ypoints,blob1Y[mask1][0:nperblob])
labels.extend(['C']*nperblob)
# labels.extend(['D']*nperblob)

Xpoints=np.append(Xpoints,1000*np.random.rand(500))
Ypoints=np.append(Ypoints,1000*np.random.rand(500))
labels.extend(['D']*500)

points4 = np.array((Xpoints,Ypoints)).T
pc4 = ms.domain('Architecture')
pc4.add_points(points4)
pc4.add_labels('Celltype', labels)
pc4.update_colors(colors,label_name='Celltype')


#%%
for pc in [pc1,pc2,pc3,pc4]:
    ms.visualise.visualise(pc,'Celltype',add_cbar=False,add_scalebar=True)
    # plt.gca().set_axis_off()
    plt.savefig(f'{outpath}I-pointcloud_{pc.name}.png')
    plt.savefig(f'{outpath}I-pointcloud_{pc.name}.svg')
    plt.close()
    #%
    #% Counts
    plt.figure(figsize=(4,8))
    labvals = ['A','B','C','D']
    labs = pc.labels['Celltype']['labels']
    vals = [np.sum(labs == v)/1 for v in labvals]
    plt.gca().bar(labvals,vals,color=[colors[v] for v in colors],linestyle='-',edgecolor='k')
    plt.gca().set_ylabel('Count',labelpad=1)
    plt.gca().set_ylim(0,600)
    plt.tight_layout()
    plt.savefig(f'{outpath}II-counts_{pc.name}.png')
    plt.savefig(f'{outpath}II-counts_{pc.name}.svg')
    plt.close()
    
    #% PCF
    plt.figure(figsize=(8,8))
    for v in ['A','B','C','D']:
        ms.spatial_statistics.cross_pair_correlation_function(pc,('Celltype','A'),('Celltype',v),max_R=500,annulus_step=5,annulus_width=20,visualise_output=True,visualise_spatial_statistic_kwargs={'line_kwargs':{'c':colors[v],'linewidth':8,'label':f'$g_{{A,{v}}}$'},'ax':plt.gca()})
    # plt.ylabel('$g_{A,\cdot}$')
    plt.ylim([0,8])
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{outpath}III-PCF_{pc.name}.png')
    plt.savefig(f'{outpath}III-PCF_{pc.name}.svg')
    # plt.close()
    
    #% Wass
    WassDistMat = np.zeros(shape=(4,4))
    cmaps = ['Blues','Oranges','Greens','bone_r']
    for i, v in enumerate(labvals):
        for j, w in enumerate(labvals):
            out = ms.distribution.sliced_wasserstein_distance(pc, ('Celltype',v),('Celltype',w))
            WassDistMat[i,j] = out
        ms.distribution.kernel_density_estimation(pc, ('Celltype',v),visualise_output=True,visualise_heatmap_kwargs={'heatmap_cmap':cmaps[i],'colorbar_limit':[0,5e-6],'visualise_kwargs':{'add_scalebar':True},'add_cbar':False})
        plt.gca().set_axis_off()
        plt.savefig(f'{outpath}IV-dens_{pc.name}_{v}.png')
        plt.savefig(f'{outpath}IV-dens_{pc.name}_{v}.svg')
        plt.close()
        #%
        # Save a copy of the colorbar too
        a = np.array([[0,1]])
        plt.figure(figsize=(8, 0.5))
        img = plt.imshow(a, cmap=cmaps[i], vmin=0, vmax=5e-6)
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.2, 0.8, 0.6])
        plt.colorbar(orientation="horizontal", cax=cax,label=f'{v}')
        plt.tight_layout()
        plt.savefig(f'{outpath}IV-colorbar_{v}.png')
        plt.savefig(f'{outpath}IV-colorbar_{v}.svg')
        plt.close()
    #% QCM
    SES, A, lab = ms.region_based.quadrat_correlation_matrix(pc, 'Celltype', region_kwargs={'side_length':50})
    
    #% Plot SES and QCM on the same axes
    mask = 1 - np.tri(SES.shape[0], k=-1)
    mask2 = np.tri(WassDistMat.shape[0])
    WassPlotMat=np.ma.array(WassDistMat, mask=mask2,fill_value=np.nan)
    A_plot = np.ma.array(SES, mask=mask,fill_value=np.nan)
    
    plt.figure(figsize=(8,8))
    sns.heatmap(A_plot,vmax=15,vmin=-15,yticklabels=labvals,xticklabels=labvals,linewidths=0, linecolor='k',
                ax=plt.gca(),square='True',cmap='RdBu_r',cbar_kws = dict(use_gridspec=False,location="bottom",label='QCM',pad=0.05),mask=mask)
    sns.heatmap(WassPlotMat,vmax=0,vmin=200,yticklabels=labvals,xticklabels=labvals,linewidths=0, linecolor='k',
                ax=plt.gca(),square='True',cmap='Greens',cbar_kws = dict(use_gridspec=False,location="right",label='Wasserstein Distance'),mask=mask2)
    plt.gca().plot([0,len(labvals)],[0,len(labvals)],linestyle='-',color='k')
    plt.gca().tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
    plt.gca().tick_params(length=5)
    plt.savefig(f'{outpath}V-heatmap_{pc.name}.png')
    plt.savefig(f'{outpath}V-heatmap_{pc.name}.svg')
    plt.close()
    
    #% PH
    plt.figure(figsize=(8,8))
    for i, v in enumerate(labvals):
        out = ms.topology.vietoris_rips_filtration(pc,('Celltype',v))
        loops = out['dgms'][1]
        plt.scatter(loops[:,0],loops[:,1],edgecolors=colors[v],marker='o',facecolors='none',s=500)
    plt.plot([0,300],[0,300],c='k',linestyle=':',lw=5,zorder=-1000)
    plt.xlim([0,300])
    plt.ylim([0,300])
    plt.savefig(f'{outpath}VI-PH_{pc.name}.png')
    plt.savefig(f'{outpath}VI-PH_{pc.name}.svg')
    plt.close()
        
    #% TCM
    TCM = ms.spatial_statistics.topographical_correlation_map(pc, ('Celltype','A'), ('Celltype','C'),radius_of_interest=25,kernel_sigma=25)
    ms.visualise.visualise_topographical_correlation_map(pc, TCM,colorbar_limit=20)
    plt.savefig(f'{outpath}VII-TCM_{pc.name}.png')
    plt.savefig(f'{outpath}VII-TCM_{pc.name}.svg')
    plt.close()
    
    out = ms.topology.level_set_filtration(TCM)
    #%
    plt.figure(figsize=(8,8))
    cc = out['dgms'][0]
    loops = out['dgms'][1]
    plt.scatter(cc[:,0],cc[:,1],marker='o',edgecolors='k',facecolors='none',label='$H_0$',s=500,lw=5)
    plt.scatter(loops[:,0],loops[:,1],facecolors=colors['C'],edgecolors=colors['C'],marker='x',label='$H_1$',s=500,lw=10)
    plt.plot([-15,30],[-15,30],c='k',linestyle=':',lw=5,zorder=-1000)
    plt.xlim([-15,30])
    plt.ylim([-15,30])
    plt.axhline(0,c=[0.7,0.7,0.7,1],linestyle=':',lw=3)
    plt.axvline(0,c=[0.7,0.7,0.7,1],linestyle=':',lw=3)
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.legend(loc='upper right')
    plt.savefig(f'{outpath}VIII-TCM-PH_{pc.name}.png')
    plt.savefig(f'{outpath}VIII-TCM-PH_{pc.name}.svg')
    plt.close()

