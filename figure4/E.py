# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings('ignore')
gamma=0.1
alpha = 0.3
t_0=1000
t_1=1
t_2=1000
B=0.8
from scipy.io import loadmat,savemat

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def simulation(A, A1):
    def F(A,x):
        return np.mat(-B*x+alpha*np.multiply(1-x,A*x))
        # return np.mat(-B*np.multiply(1-x,x)+np.multiply(x,A*(1-1/x)))
    
    def Fun(t,x):
        x=np.mat(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        return dx
    
    def Fun_1(t,x,source):
        x=np.mat(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        dx[source]=0
        return dx
    
    def Fun_2(t,x,source):
        x=np.mat(x).T
        dx=F(A1,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        dx[source]=0
        return dx
    
    def sim_first(A):
        x_0=np.zeros(np.shape(A)[0])
        sol=solve_ivp(Fun, [0,t_0], x_0, rtol=1e-10, atol=1e-10) #odeint(Fun,x_0,t,args=(A,a,b))
        xs=sol.y.T
        ts=sol.t
        x=xs[-1,:].tolist()
        t=ts[-1]
        return x, t, xs, ts
    
    def sim_second(A,x,t_0,source):
        x[source]+=gamma
        sol=solve_ivp(Fun_1, [t_0,t_0+t_1], x, rtol=1e-10, atol=1e-10, args=(source,))#odeint(Fun_1,x,t,args=(A,a,b,source),atol=1e-13,rtol=1e-13)
        xs1=sol.y.T
        ts1=sol.t
        x1=xs1[-1,:].tolist()
        sol=solve_ivp(Fun_2, [t_0+t_1,t_0+t_1+t_2], x1, rtol=1e-10, atol=1e-10, args=(source,))#odeint(Fun_1,x,t,args=(A,a,b,source),atol=1e-13,rtol=1e-13)
        xs2=sol.y.T
        ts2=sol.t
        x=xs2[-1,:].tolist()
        return x, xs1, ts1, xs2, ts2
    
    x1,t, xs1, ts1=sim_first(A)
    x2, xs2, ts2, xs3, ts3=sim_second(A,x1.copy(),t,source)
    xs=np.vstack((xs1,xs2,xs3))
    ts=np.hstack((ts1,ts2,ts3))
    Deltax=[i-j for (i,j) in zip(x1,x2)]
    return x2, np.linalg.norm(Deltax), xs, ts

def create_G():
    G=nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_node(4)
    G.add_node(5)
    G.add_node(6)
    G.add_node(7)
    
    G.add_edge(0,1)
    G.add_edge(0,2)
    G.add_edge(1,2)
    G.add_edge(2,3)
    # G.add_edge(2,5)
    G.add_edge(3,4)
    G.add_edge(3,5)
    G.add_edge(3,6)
    # G.add_edge(4,5)
    G.add_edge(5,6)
    G.add_edge(6,7)
    return G
    
def draw_graph(G, x2, value, arrows = None, count=0):
    colors = ['#EECB8E', '#DC8910','#83272E']
    fig=plt.figure(figsize=(9,4))
    ax=fig.add_subplot(111)
    pos=dict()
    pos[0]=(0,0)
    pos[1]=(1,0)
    pos[2]=(0.5,-0.8)
    pos[3]=(2,0)
    pos[4]=(3,0)
    pos[5]=(3,-0.8)
    pos[6]=(2,-0.8)
    pos[7]=(1.2,-0.8)
    r1=0.08
    r2=0.11
    for (u,v) in G.edges:
        if(arrows == None):
            ax.plot([pos[u][0],pos[v][0]], [pos[u][1], pos[v][1]], linewidth=2.5, color='black', zorder=1, alpha=0.75)
        else:
            if((u,v) not in arrows and (v,u) not in arrows):
                ax.plot([pos[u][0],pos[v][0]], [pos[u][1], pos[v][1]], linewidth=2.5, color='black', zorder=1, alpha=0.75)
    theta=0
    if(arrows != None):
        for arrow in arrows:
            p,q=arrow
            x_1,y_1=pos[p]
            x_2,y_2=pos[q]
            dy = (pos[q][1]-pos[p][1])
            dx = (pos[q][0]-pos[p][0])
            if (x_2>x_1 and y_2>y_1):
                theta = np.arctan (dy/dx)
            else:
                if(x_2<x_1 and y_2>=y_1):
                    theta = np.pi + np.arctan (dy/dx)
                else:
                    if (x_2<x_1 and y_2<y_1):
                        theta = np.arctan (dy/dx)+np.pi
                    else:
                        if (x_2>x_1 and y_2<=y_1):
                            theta = np.pi*2+np.arctan (dy/dx)
                        else:
                            if(y_2>y_1):
                                theta = np.pi/2
                            else:
                                theta = -np.pi/2
            ax.arrow(pos[p][0], pos[p][1], pos[q][0]-pos[p][0]-r2*np.cos(theta), pos[q][1]-pos[p][1]-r2*np.sin(theta), linewidth=2.5, color = colors[1], alpha=0.75, length_includes_head=True, head_width=0.1,shape='full')
    for i in range(len(pos)):
        if(i!=source):
            ax.scatter(pos[i][0],pos[i][1],s=800,c=colors[1],linewidth=2.5,edgecolor='black', zorder=2)
    ax.scatter(pos[source][0],pos[source][1],s=800,c=colors[2],linewidth=2.5,edgecolor='black', zorder=2)
    # nx.draw_networkx(G,pos=pos,ax=ax)
    for i in range(8):
        if(x2[i]<1e-10):
            continue
        else:
            ax.vlines(pos[i][0],pos[i][1]+r1,pos[i][1]+x2[i]*2+r1,linewidth=15,color=colors[0], alpha=0.9)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(value,fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-1,0.25])
    plt.tight_layout()
    plt.savefig('E_'+str(int(100*B))+'_'+str(source)+'_'+str(count)+'.pdf',dpi=300)
    plt.show()
    
# A_edge=np.mat(loadmat('../Networks/UCIonline.mat')['A'])
G_edge=create_G()
A_edge=nx.to_numpy_matrix(G_edge)
degrees=np.sum(A_edge,axis=1)

source=7
edges=[(5,6),(3,6)]

epsilon=1
for j,(u,v) in enumerate(edges):
    B_edge=A_edge.copy()
    B_edge[u,v]-=epsilon
x1, value_ori, xs1, ts1 = simulation(A_edge, B_edge)
index1=np.where(ts1==t_0)[0][-1]
x1=xs1[index1,:].tolist()
value1='Initial State'
draw_graph(G_edge, x1, value1, arrows = None,count=0)

index2=np.where(ts1==(t_0+t_1))[0][-1]
x2=xs1[index2,:].tolist()
value2='State when observed'
draw_graph(G_edge, x2, value2, arrows = None,count=1)

index3=np.where(ts1==(t_0+t_1+t_2))[0][-1]
x3=xs1[index3,:].tolist()
value3='Restricted'
draw_graph(G_edge, x3, value3, arrows = edges,count=2)

x4, value_ori, xs2, ts2 = simulation(A_edge, A_edge)
value4='Unrestricted'
draw_graph(G_edge, x4, value4, arrows = None,count=3)