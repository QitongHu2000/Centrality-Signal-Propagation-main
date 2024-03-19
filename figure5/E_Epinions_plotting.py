# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')
import scipy.stats
from matplotlib.ticker import MaxNLocator

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

B=0.8

our_simulation = load_dict('E_Epinions_our_simulation_'+str(int(100*B))).tolist()[0]
degree_simulation = load_dict('E_Epinions_degree_simulation_'+str(int(100*B))).tolist()[0]
laplacian_simulation = load_dict('E_Epinions_laplacian_simulation_'+str(int(100*B))).tolist()[0]

colors = ['#EECB8E', '#DC8910','#83272E']

fig=plt.figure(figsize=(6.5,6))
ax=fig.add_subplot(111)

ax.plot(range(len(our_simulation)), our_simulation,c=colors[1],linewidth=2.5,linestyle='-', alpha = 0.7)
ax.scatter(range(len(our_simulation)), our_simulation,s=150,marker = 's',c='none', edgecolors=colors[1],alpha =0.7,label='Ours')

ax.plot(range(len(degree_simulation)), degree_simulation,c=colors[0],linewidth=2.5,linestyle='-', alpha = 0.7)
ax.scatter(range(len(degree_simulation)), degree_simulation,s=150,marker = 'o',c='none', edgecolors=colors[0],alpha =0.7,label='Degree')

ax.plot(range(len(laplacian_simulation)), laplacian_simulation,c=colors[2],linewidth=2.5,linestyle='-', alpha = 0.7)
ax.scatter(range(len(laplacian_simulation)), laplacian_simulation,s=150,marker = '^',c='none', edgecolors=colors[2],alpha =0.7,label='Laplacian')

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))

ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)\
    
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlabel(r"$|E|$",fontsize=35)
plt.ylabel(r"$\mathcal{E}(m,E)$",fontsize=35)
# plt.title('Epinions',fontsize=40)
# plt.legend(fontsize=20,loc=1,bbox_to_anchor=(1.6,1))
plt.tight_layout()
# plt.xscale('log')
# plt.yscale('log')
# plt.axis('equal')
plt.savefig('E_Epinions_'+str(int(100*B))+'.pdf',dpi=300)
plt.show()