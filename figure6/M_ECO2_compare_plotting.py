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

B=0.2
C=0.1
a=1
simulation_indexs = load_dict('M_ECO2_simulation_'+str(int(100*B))+'_'+str(int(100*C))).tolist()[0]
theory_indexs = load_dict('M_ECO2_theory_'+str(int(100*B))+'_'+str(int(100*C))).tolist()[0]

colors = ['#8DDFBF','#328CA0', '#006837']

fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111)

# ax.plot(simulation_indexs,simulation_indexs,c=colors[2],linewidth=2.5,linestyle='-', alpha = 0.7)

# theory_indexs=[i/theory_indexs[0]*simulation_indexs[0] for i in theory_indexs]
ax.scatter(theory_indexs, simulation_indexs,s=150,marker = 's',c='none', edgecolors=colors[1],alpha =0.7)

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))

ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
    
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlabel(r"$\mathcal{S}(m,E)$",fontsize=35)
plt.ylabel(r"$\Delta \mathcal{E}(m,E)$",fontsize=35)
# plt.title('ATN',fontsize=40)
# plt.legend(fontsize=20,loc=1,bbox_to_anchor=(1.6,1))
plt.tight_layout()
plt.xscale('log')
plt.yscale('log')
# plt.axis('equal')
plt.savefig('M_ECO2_compare_'+str(int(100*B))+'_'+str(int(100*C))+'.pdf',dpi=300,bbox_inches='tight')
plt.show()