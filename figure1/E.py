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
t_1=1000
B=0.8
from scipy.io import loadmat,savemat

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def simulation(A):
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
    
    def sim_first(A):
        x_0=np.zeros(np.shape(A)[0])
        sol=solve_ivp(Fun, [0,t_0], x_0, rtol=1e-15, atol=1e-15) #odeint(Fun,x_0,t,args=(A,a,b))
        xs=sol.y.T
        t=sol.t
        x=xs[-1,:].tolist()
        return x, t[-1]
    
    def sim_second(A,x,t_0,source):
        x[source]+=gamma
        sol=solve_ivp(Fun_1, [t_0,t_0+t_1], x, rtol=1e-15, atol=1e-15, args=(source,))#odeint(Fun_1,x,t,args=(A,a,b,source),atol=1e-13,rtol=1e-13)
        xs=sol.y.T
        x=xs[-1,:].tolist()
        return x
    
    x1,t=sim_first(A)
    x2=sim_second(A,x1.copy(),t,source)
    Deltax=[i-j for (i,j) in zip(x1,x2)]
    return x1, x2, np.linalg.norm(Deltax)

def Jac(A,x):
    J_=np.diag((-B/(1-x)).T.tolist()[0])
    S_=np.diag((1-x).T.tolist()[0])
    T_=alpha
    return np.mat(J_+S_*A*T_)

def calculate_weight(G):
    k=source
    H=G.copy()
    H=nx.DiGraph(H)
    for (u,v) in H.edges:
        H.edges[u,v]['weight']=np.abs(J1[v,k]*(J1[k,u]/J1[k,k]-J2[k,u]/J2[k,k]))
    return H

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

def draw_graph_ori(G):
    colors = ['#EECB8E', '#DC8910','#83272E']
    fig=plt.figure(figsize=(8,4))
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
    r1=0.15
    for (u,v) in G.edges:
        ax.plot([pos[u][0],pos[v][0]], [pos[u][1], pos[v][1]], linewidth=2.5, color='black', zorder=1, alpha=0.75)
    for i in range(len(pos)):
        ax.scatter(pos[i][0],pos[i][1],s=800,c=colors[1],linewidth=2.5,edgecolor='black', zorder=2)
        ax.text(pos[i][0]-0.08/np.sqrt(2),pos[i][1]-0.08/np.sqrt(2),i,fontsize=25)
    # nx.draw_networkx(G,pos=pos,ax=ax)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    # plt.title(value)
    plt.ylim([-1,0.5])
    plt.savefig('E_'+str(int(100*B))+'.pdf',dpi=300)
    plt.show()
    
def draw_graph(G, x2, value, arrow = None, count=0):
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
    r2=0.1
    for (u,v) in G.edges:
        if((u,v) != arrow and (v,u) != arrow):
            ax.plot([pos[u][0],pos[v][0]], [pos[u][1], pos[v][1]], linewidth=2.5, color='black', zorder=1, alpha=0.75)
    theta=0
    if(arrow != None):
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
    plt.xticks([])
    plt.yticks([])
    if(value<1e-10):
        plt.title('Energy value is '+str(np.round(np.linalg.norm(x2),2)),fontsize=25)
    else:
        plt.title('Energy decreases '+str(np.round(value*100,2))+'%',fontsize=25)
    plt.ylim([-1,0.25])
    plt.tight_layout()
    plt.savefig('E_'+str(int(100*B))+'_'+str(count)+'_'+str(source)+'.pdf',dpi=300, bbox_inches='tight')
    plt.show()
    
# A_edge=np.mat(loadmat('../Networks/UCIonline.mat')['A'])
G_edge=create_G()
A_edge=nx.to_numpy_matrix(G_edge)
D_edge=nx.laplacian_matrix(G_edge).todense()
G_edge=G_edge.copy()
G_edge=nx.DiGraph(G_edge)
eigs=np.linalg.eigh(D_edge)[1]
xi2=eigs[:,1].T.tolist()[0]
degrees=np.sum(A_edge,axis=1)

source=7
edges=G_edge.edges
simulation_indexs=list()
theory_indexs=list()

count=0
x1, x2, value_ori =simulation(A_edge)
draw_graph(G_edge, x2, 0, count = 0)
draw_graph_ori(G_edge)
x1=np.mat(x1).T
J1=np.mat(np.linalg.inv(Jac(A_edge, x1)))
J2=np.mat(J1.T*J1)
H_edge=calculate_weight(G_edge)

epsilon=1
for j,(u,v) in enumerate(edges):
    B_edge=A_edge.copy()
    B_edge[u,v]-=epsilon
    _, x2, value_per = simulation(B_edge)
    error=np.abs((value_per-value_ori)/value_ori)
    draw_graph(G_edge, x2, error, arrow = (u,v), count = j+1)
    print('type1',(u,v),np.abs((value_per-value_ori)/value_ori))