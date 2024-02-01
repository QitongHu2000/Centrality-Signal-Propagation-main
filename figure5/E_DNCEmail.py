# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import warnings
import pickle
warnings.filterwarnings('ignore')
gamma=0.1
alpha = 0.01
t_0=200
t_1=10
t_2=100
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
        return x, t
    
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
    
    x1,t=sim_first(A)
    x2, xs1, ts1, xs2, ts2=sim_second(A,x1.copy(),t,source)
    xs=np.vstack((xs1,xs2))
    ts=np.hstack((ts1,ts2))
    Deltax=[i-j for (i,j) in zip(x1,x2)]
    return x1, np.linalg.norm(Deltax), xs, ts

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
        H.edges[u,v]['weight_ours']=np.abs(J1[v,k]*(J1[k,u]/J1[k,k]-J2[k,u]/J2[k,k]))
        H.edges[u,v]['weight_degree']=np.abs(degrees[u,0]*degrees[v,0])
        H.edges[u,v]['weight_laplacian']=np.abs((xi2[u]-xi2[v])**2)
    return H

A_edge=np.mat(loadmat('../Networks/DNCEmail.mat')['A'])
G_edge=nx.from_numpy_matrix(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
A_edge=nx.to_numpy_matrix(G_edge)
D_edge=nx.laplacian_matrix(G_edge).todense()
G_edge=G_edge.copy()
G_edge=nx.DiGraph(G_edge)
eigs=np.linalg.eigh(D_edge)[1]
xi2=eigs[:,1].T.tolist()[0]
print('max',np.where(xi2==np.max(xi2)),'min',np.where(xi2==np.min(xi2)))
degrees=np.sum(A_edge,axis=1)

source=52 #np.random.choice(G_edge.nodes,1)[0]
print(source,degrees[source,0])
x1, value_ori, xs1, ts1 =simulation(A_edge, A_edge)
x1=np.mat(x1).T
J1=np.mat(np.linalg.inv(Jac(A_edge, x1)))
J2=np.mat(J1.T*J1)
H_edge=calculate_weight(G_edge)

simulation_index=list()
theory_index=list()

edges_list=list()
our_value=list()
degree_value=list()
laplacian_value=list()
for k,(u,v) in enumerate(H_edge.edges):
    print('calculate',k,u,v)
    edges_list.append((u,v))
    our_value.append(H_edge.edges[u,v]['weight_ours'])
    degree_value.append(H_edge.edges[u,v]['weight_degree'])
    laplacian_value.append(H_edge.edges[u,v]['weight_laplacian'])

our_index=np.argsort(our_value)
our_index=our_index[::-1]
degree_index=np.argsort(degree_value)
degree_index=degree_index[::-1]
laplacian_index=np.argsort(laplacian_value)
laplacian_index=laplacian_index[::-1]

k_max=50 #int(len(H_edge.edges)/20)
our_selected=our_index[0:k_max]
degree_selected=degree_index[0:k_max]
laplacian_selected=laplacian_index[0:k_max]

our_simulation=[value_ori,]
degree_simulation=[value_ori,]
laplacian_simulation=[value_ori,]
B_edge=A_edge.copy()
for (k,i) in enumerate(our_selected):
    u,v=edges_list[i]
    print('our',k,u,v)
    B_edge[u,v]=0
    _, value_per, xs_our, ts_our = simulation(A_edge, B_edge)
    our_simulation.append(value_per)

B_edge=A_edge.copy()
for (k,i) in enumerate(degree_selected):
    u,v=edges_list[i]
    print('degree',k,u,v)
    B_edge[u,v]=0
    _, value_per, xs_degree, ts_degree = simulation(A_edge, B_edge)
    degree_simulation.append(value_per)

B_edge=A_edge.copy()
for (k,i) in enumerate(laplacian_selected):
    u,v=edges_list[i]
    print('laplacian',k,u,v)
    B_edge[u,v]=0
    _, value_per, xs_laplacian, ts_laplacian = simulation(A_edge, B_edge)
    laplacian_simulation.append(value_per)
    
save_dict(np.mat(our_simulation), 'E_DNCEmail_our_simulation_'+str(int(100*B)))
save_dict(np.mat(degree_simulation), 'E_DNCEmail_degree_simulation_'+str(int(100*B)))
save_dict(np.mat(laplacian_simulation), 'E_DNCEmail_laplacian_simulation_'+str(int(100*B)))

save_dict(np.mat(xs_our), 'E_DNCEmail_xs_our_'+str(int(100*B)))
save_dict(np.mat(ts_our), 'E_DNCEmail_ts_our_'+str(int(100*B)))
save_dict(np.mat(xs_degree), 'E_DNCEmail_xs_degree_'+str(int(100*B)))
save_dict(np.mat(ts_degree), 'E_DNCEmail_ts_degree_'+str(int(100*B)))
save_dict(np.mat(xs_laplacian), 'E_DNCEmail_xs_laplacian_'+str(int(100*B)))
save_dict(np.mat(ts_laplacian), 'E_DNCEmail_ts_laplacian_'+str(int(100*B)))