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

xs_our=load_dict('E_UCIonline_xs_our_'+str(int(100*B)))
xs_our=np.mean(xs_our,axis=1).T.tolist()[0]
ts_our=load_dict('E_UCIonline_ts_our_'+str(int(100*B))).tolist()[0]
ts_our=[i-200 for i in ts_our]
xs_degree=load_dict('E_UCIonline_xs_degree_'+str(int(100*B)))
xs_degree=np.mean(xs_degree,axis=1).T.tolist()[0]
ts_degree=load_dict('E_UCIonline_ts_degree_'+str(int(100*B))).tolist()[0]
ts_degree=[i-200 for i in ts_degree]
xs_laplacian=load_dict('E_UCIonline_xs_laplacian_'+str(int(100*B)))
xs_laplacian=np.mean(xs_laplacian,axis=1).T.tolist()[0]
ts_laplacian=load_dict('E_UCIonline_ts_laplacian_'+str(int(100*B))).tolist()[0]
ts_laplacian=[i-200 for i in ts_laplacian]

# Deltax=xs_our[-1,:]-xs_our[0,:]
# index=np.where(Deltax==np.max(Deltax))[1][0]
colors = ['#EECB8E', '#DC8910','#83272E']

fig=plt.figure(figsize=(6.25,6))
ax=fig.add_subplot(111)

ax.plot(ts_our,xs_our,c=colors[1],linewidth=2.5,linestyle='-', alpha = 0.7)
ax.plot(ts_degree,xs_degree,c=colors[0],linewidth=2.5,linestyle='--', alpha = 0.7)
ax.plot(ts_laplacian,xs_laplacian,c=colors[2],linewidth=2.5,linestyle='--', alpha = 0.7)

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
plt.xlabel(r"$t$",fontsize=35)
plt.ylabel(r"$\langle \mathbf{x}(t)\rangle$",fontsize=35)
# plt.title('UCIonline',fontsize=40)
# plt.legend(fontsize=20,loc=1,bbox_to_anchor=(1.6,1))
plt.tight_layout()
# plt.xscale('log')
# plt.yscale('log')
# plt.axis('equal')
plt.savefig('E_UCIonline_state_'+str(int(100*B))+'.pdf',dpi=300)
plt.show()