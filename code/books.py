import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import scipy as sp
import numpy as np
from math import log

df1 = pd.read_csv('nodes_list.csv', sep=',')
df2 = pd.read_csv('edge_list.csv', sep=',')

# generate graph
G2 = nx.DiGraph()
for idx, row in df1.iterrows():
    G2.add_node(row['Language'])
for idx, row in df2.iterrows():
    G2.add_edge(row['TargetLanguageCode'], row['SourceLanguageCode'], weight=row['Coocurrences'])

G = nx.Graph()
df3 = pd.read_csv('gdp_by_language.csv', sep=',')
gdp = {}
pop = {}

for idx, row in df3.iterrows():
    G.add_node(row['lang'])
    gdp[row['lang']] = row['gdp_pc']
    pop[row['lang']] = row['actual_speakers_m']

nx.set_node_attributes(G, 'gdp', gdp)
nx.set_node_attributes(G, 'pop', pop)

#eigenvector centrality
centrality = nx.eigenvector_centrality(G2, weight='weight')
nx.set_node_attributes(G2, 'centrality', centrality)

# to print sorted list of eigenvector centralities
#for key,values in sorted(G2.node.items(), key=lambda (k,v):(v,k), reverse=True):
#	print('%s %0.6f'%(key, values['centrality']))


#converting directed to undirected network
#adjacency matrix
A = nx.to_numpy_matrix(G2)
B = np.zeros((A.shape[0], A.shape[1]))
for i in range(0,A.shape[0]):
    for j in range(0,i+1):
        B[i,j] = A[i,j] + A[j,i]
        B[j,i] = B[i,j]
G4 = nx.from_numpy_matrix(B)
centrality = nx.eigenvector_centrality(G4, weight='weight')
nx.set_node_attributes(G4, 'centrality', centrality)

# find correlation between centrality and famous people
shared_items = set(G2.nodes()) & set(G.nodes())
gdp_val = []
pop_val = []
centrality_val = []

for i in shared_items:
    pop_val.append(G.node[i]['pop'])
    gdp_val.append(G.node[i]['gdp'])
    centrality_val.append(G2.node[i]['centrality'])

G1 = nx.Graph()
df4 = pd.read_csv('famous_wikipedia.csv', sep=',')
fpeople = {}

for idx, row in df4.iterrows():
    G1.add_node(row['lang'])
    fpeople[row['lang']] = row['exports_1800_1950']

nx.set_node_attributes(G1, 'fpeople', fpeople)

shared_items = set(G2.nodes()) & set(G1.nodes())
famousp_val = []
centrality_val = []
label = []

for i in shared_items:
    if G2.node[i]['centrality'] != 0:
        famousp_val.append(log(G1.node[i]['fpeople'],10))
        centrality_val.append(log(G2.node[i]['centrality'],10))
        label.append(i)

coefs = np.polynomial.polynomial.polyfit(centrality_val,famousp_val,1)
ffit = np.polynomial.polynomial.polyval(centrality_val, coefs)
plt.plot(centrality_val, ffit, 'r')

plt.scatter(centrality_val, famousp_val, s=300, c=[str(x/np.amax(gdp_val)) for x in gdp_val], cmap = cm.get_cmap('Greys'))
m = cm.ScalarMappable(cmap=cm.get_cmap('Greys'))
m.set_array([gdp_val])
cbar = plt.colorbar(m)
cbar.set_label("GDP per capita", rotation=90, fontsize=24)

for l,x,y in zip(label,centrality_val, famousp_val):
    plt.annotate(l, xy = (x, y), xytext = (-5, 5),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.ylabel("log10(Wikipedia 26+ famous people)", fontsize=24)
plt.xlabel("log10(Book Trans. Eigenvector Centrality)", fontsize=24)
plt.show()

# correlation between famous people and centrality
r = np.corrcoef(centrality_val, famousp_val)[0, 1]
print('correlation coefficient: ',r)