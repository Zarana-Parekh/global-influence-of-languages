import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import scipy as sp
import numpy as np
from math import log

df1 = pd.read_csv('dataset_stats_books.csv', sep=',')
df2 = pd.read_csv('dataset_stats_twitter.csv', sep=',')
df3 = pd.read_csv('dataset_stats_wikipedia.csv', sep=',')

trans_val = {}
users_val = {}
editors_val = {}

G1 = nx.DiGraph()
for idx, row in df1.iterrows():
    G1.add_node(row['Language'])
    if row['TranslationsFrom'] or row['TranslationsTo']:
		trans_val[row['Language']] = row['TranslationsFrom']

G2 = nx.DiGraph()
for idx, row in df2.iterrows():
    G2.add_node(row['Language'])
    users_val[row['Language']] = row['Users']
G3 = nx.DiGraph()
for idx, row in df3.iterrows():
    G3.add_node(row['Language'])
    editors_val[row['Language']] = row['Editors']

nx.set_node_attributes(G1, 'translations', trans_val)
nx.set_node_attributes(G2, 'users', users_val)
nx.set_node_attributes(G3, 'editors', editors_val)

common12 = set(G1.nodes()) & set(G2.nodes())

trans = []
users = []
editors = []
label = []

for i in common12:
    trans.append(log(G1.node[i]['translations'],10))
    users.append(log(G2.node[i]['users'],10))
    label.append(i)

plt.figure()
plt.scatter(trans, users, s=100)

for l,x,y in zip(label, trans, users):
    plt.annotate(l, xy = (x, y), xytext = (-5, 5),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.1),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.ylabel("log10(Book Translations)", fontsize=24)
plt.xlabel("log10(Twitter Users)", fontsize=24)
plt.show()

common23 = set(G2.nodes()) & set(G3.nodes())

trans = []
users = []
editors = []
label = []

for i in common23:
    editors.append(log(G3.node[i]['editors'],10))
    users.append(log(G2.node[i]['users'],10))
    label.append(i)

plt.figure()
plt.scatter(editors, users, s=100)

for l,x,y in zip(label, editors, users):
    plt.annotate(l, xy = (x, y), xytext = (-5, 5),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.1),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.ylabel("log10(Twitter Users)", fontsize=24)
plt.xlabel("log10(Wikipedia Editors)", fontsize=24)
plt.show()

common13 = set(G1.nodes()) & set(G3.nodes())

trans = []
users = []
editors = []
label = []

for i in common13:
	if G1.node[i]['translations'] != 0:
	    trans.append(log(G1.node[i]['translations'],10))
	    editors.append(log(G3.node[i]['editors'],10))
	    label.append(i)

plt.figure()
plt.scatter(trans, editors, s=100)

for l,x,y in zip(label, trans, editors):
    plt.annotate(l, xy = (x, y), xytext = (-5, 5),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.1),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.ylabel("log10(Book Translations)", fontsize=24)
plt.xlabel("log10(Wikipedia Editors)", fontsize=24)
plt.show()