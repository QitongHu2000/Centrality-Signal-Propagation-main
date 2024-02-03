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
        H.edges[u,v]['weight']=np.abs(J3[u,v]*J1[v,k]*(J1[k,u]/J1[k,k]-J2[k,u]/J2[k,k]))
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
degree_indexs=list()
laplace_indexs=list()

count=0
x1, x2, value_ori =simulation(A_edge)
x1=np.mat(x1).T
J3=np.mat(Jac(A_edge, x1))
J1=np.mat(np.linalg.inv(J3))
J2=np.mat(J1.T*J1)
H_edge=calculate_weight(G_edge)

epsilon=1
simulation_index=list()
theory_index=list()
degree_index=list()
laplace_index=list()
for u,v in edges:
    B_edge=A_edge.copy()
    B_edge[u,v]-=epsilon
    _, x2, value_per = simulation(B_edge)
    error=np.abs((value_per-value_ori)/value_ori)
    error=np.round(error,3)
    print('type1',(u,v),np.abs((value_per-value_ori)/epsilon*value_ori))
    simulation_index.append(np.abs((value_per-value_ori)/(epsilon*value_ori)))

for u,v in edges:
    theory_index.append(H_edge.edges[u,v]['weight'])

for u,v in edges:
    laplace_index.append((xi2[u]-xi2[v])**2)
    print('type2', (u,v), (xi2[u]-xi2[v])**2)

for u,v in edges:
    degree_index.append((degrees[u,0]*degrees[v,0])/np.sum(degrees))
    print('type3', (u,v), (degrees[u,0]*degrees[v,0])/np.sum(degrees))
    
simulation_indexs.extend(simulation_index)
theory_indexs.extend(theory_index)
degree_indexs.extend(degree_index)
laplace_indexs.extend(laplace_index)

save_dict(np.mat(simulation_indexs), 'E_simulation_'+str(int(100*B))+'_'+str(source))
save_dict(np.mat(theory_indexs), 'E_theory_'+str(int(100*B))+'_'+str(source))
save_dict(np.mat(degree_indexs), 'E_degree_'+str(int(100*B))+'_'+str(source))
save_dict(np.mat(laplace_indexs), 'E_laplace_'+str(int(100*B))+'_'+str(source))