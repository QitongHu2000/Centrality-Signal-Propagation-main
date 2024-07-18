# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import warnings
import pickle
warnings.filterwarnings('ignore')
gamma=0.3
alpha = 0.01
t_0=200
t_1=200
B=0.2
C=0.1
a=1
from scipy.io import loadmat,savemat

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def simulation(A):
    def F(A,x):
        return np.mat(B*np.multiply(x,1-np.power(x,a)/C)+alpha*np.multiply(x,A*x))
    
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
        x_0=np.ones(np.shape(A)[0])*1e-10
        sol=solve_ivp(Fun, [0,t_0], x_0, rtol=1e-10, atol=1e-10) #odeint(Fun,x_0,t,args=(A,a,b))
        xs=sol.y.T
        t=sol.t
        x=xs[-1,:].tolist()
        return x, t[-1]
    
    def sim_second(A,x,t_0,source):
        x[source]+=gamma
        sol=solve_ivp(Fun_1, [t_0,t_0+t_1], x, rtol=1e-10, atol=1e-10, args=(source,))#odeint(Fun_1,x,t,args=(A,a,b,source),atol=1e-13,rtol=1e-13)
        xs=sol.y.T
        x=xs[-1,:].tolist()
        return x
    
    x1,t=sim_first(A)
    x2=sim_second(A,x1.copy(),t,source)
    Deltax=[i-j for (i,j) in zip(x1,x2)]
    return x1, np.linalg.norm(Deltax)

def Jac(A,x):
    J_=-B/C*a*np.identity(len(A))#np.diag(np.power(x,a).T.tolist()[0])
    S_=1#np.diag(x.T.tolist()[0])
    T_=alpha#*np.diag((1/np.power(1+x,2)).T.tolist()[0])
    return np.mat(J_+S_*A*T_)

def calculate_weight(G):
    k=source
    H=G.copy()
    H=nx.DiGraph(H)
    for (u,v) in H.edges:
        H.edges[u,v]['weight']=np.abs(J3[u,v]*J1[v,k]*(J1[k,u]/J1[k,k]-J2[k,u]/J2[k,k]))
    return H

A_edge=np.mat(loadmat('../Networks/ECO1.mat')['A'])
G_edge=nx.from_numpy_matrix(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
G_edge=nx.convert_node_labels_to_integers(G_edge)
A_edge=nx.to_numpy_matrix(G_edge)
A_edge=(A_edge>0.3)
G_edge=nx.from_numpy_matrix(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
G_edge=nx.convert_node_labels_to_integers(G_edge)
A_edge=nx.to_numpy_matrix(G_edge)
D_edge=nx.normalized_laplacian_matrix(G_edge).todense()
G_edge=G_edge.copy()
G_edge=nx.DiGraph(G_edge)
eigs=np.linalg.eigh(D_edge)[1]
xi2=eigs[:,1].T.tolist()[0]
print('max',np.where(xi2==np.max(xi2)),'min',np.where(xi2==np.min(xi2)))
degrees=np.sum(A_edge,axis=1)

k1=1
k2=100
sources=[68, ] #np.random.choice(G_edge.nodes,k1) #[196, 55, 107]
indexs=[k for(k,i) in list(G_edge.edges)]
edges=np.random.choice(indexs,k2,replace=False)
edges=[list(G_edge.edges)[i] for i in edges]

simulation_indexs=list()
theory_indexs=list()

count=0
for source in sources:
    print(source)
    x1, value_ori =simulation(A_edge)
    x1=np.mat(x1).T
    J3=np.mat(Jac(A_edge, x1))
    J1=np.mat(np.linalg.inv(J3))
    J2=np.mat(J1.T*J1)
    H_edge=calculate_weight(G_edge)
    
    simulation_index=list()
    theory_index=list()
    for u,v in edges:
        print(source, (u,v))
        B_edge=A_edge.copy()
        B_edge[u,v]=2
        _, value_per = simulation(B_edge)
        simulation_index.append(np.abs((value_per-value_ori)/value_ori))
    
    for u,v in edges:
        theory_index.append(H_edge.edges[u,v]['weight'])
    simulation_indexs.extend(simulation_index)
    theory_indexs.extend(theory_index)
    
save_dict(np.mat(simulation_indexs), 'M_ECO1_simulation_'+str(int(100*B))+'_'+str(int(100*C)))
save_dict(np.mat(theory_indexs), 'M_ECO1_theory_'+str(int(100*B))+'_'+str(int(100*C)))