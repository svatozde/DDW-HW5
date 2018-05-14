from itertools import combinations
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import time
from collections import Counter
import itertools
from random import randint
import math
import pickle
import os.path
from pathlib import Path



def frequentItems(transactions, support):
    counter = Counter()
    for trans in transactions:
        counter.update(frozenset([t]) for t in trans)
    return set(item for item in counter if counter[item] / len(transactions) >= support), counter


def generateCandidates(L, k):
    candidates = set()
    for a in L:
        for b in L:
            union = a | b
            if len(union) == k and a != b:
                candidates.add(union)
    return candidates


def filterCandidates(transactions, itemsets, support):
    counter = Counter()
    for trans in transactions:
        subsets = [itemset for itemset in itemsets if itemset.issubset(trans)]
        counter.update(subsets)
    return set(item for item in counter if counter[item] / len(transactions) >= support), counter


def apriori(transactions, support):
    result = list()
    resultc = Counter()
    candidates, counter = frequentItems(transactions, support)
    result += candidates
    resultc += counter
    k = 2
    while candidates:
        candidates = generateCandidates(candidates, k)
        candidates, counter = filterCandidates(transactions, candidates, support)
        result += candidates
        resultc += counter
        k += 1
    resultc = {item: (resultc[item] / len(transactions)) for item in resultc}
    return result, resultc



def findsubsets(S,m):
    return set(itertools.combinations(S, m))

def genereateRules(frequentItemsets, supports, minConfidence):
    ret = []
    for itemset in frequentItemsets:
        for i in range(1,len(itemset)):
          for s in itertools.combinations(itemset, i):
              ss = set()
              for si in s:
                  ss.add(si)
              if supports[itemset]/supports[itemset - ss] > minConfidence:
                #print(str(ss) + ' => ' +str(itemset - ss) + ' with confidence:' + str(supports[itemset]/supports[itemset - ss]) + ' and support' + str(supports[itemset]))
                yield ss,itemset

def create_graph(clicks):
    """
     Hits and pagerank is build in algorythm in networx we can benefit from that fact so lets build oriented grpah from input

    :param clicks: Padast dataframe
    :return: Graph representing transition from page to page
    """
    G = nx.DiGraph()
    prew_row = None
    for index, row in clicks.iterrows():
        pId = row['PageID']
        #print(str(row['PageID']) + ':' + str(row['PageName']))
        G.add_node(pId,PageName=row['PageName'])
        if prew_row is not None and prew_row['VisitID']== row['VisitID']:
            G.add_edge(prew_row['PageID'],pId)
            e_dict = G.get_edge_data(prew_row['PageID'],pId)
            if 'count' not in e_dict:
                e_dict['count']=1
            else:
                e_dict['count']+=1
        prew_row = row

def create_baskets(clicks):
    ret = {}
    ret_row = []
    prew_row = None

    for index, row in clicks.iterrows():
        pId = row['PageID']
        # print(str(row['PageID']) + ':' + str(row['PageName']))
        page_name = row['PageName']
        page_id = row['PageID']
        if page_name == 'APPLICATION':
            page_id = 0
        order = row['SequenceNumber']
        if prew_row is not None and prew_row['VisitID'] == row['VisitID']:
            ret_row.append((page_id,page_name))
        else:
            if len(ret_row) > 0:
                ret[prew_row['VisitID']]=ret_row
            ret_row = []
            ret_row.append((page_id,page_name))

        prew_row = row

    return ret

def create_vistiors_bakets(visits_basket, visitors):
    ret={}
    for index, row in visitors.iterrows():
        visit_id = row['VisitID']
        key = (row['Referrer'],row['Day'],row['Hour'])
        if not key in ret:
            ret[key] = []
        ret[key]+=visits_basket[visit_id]
    return ret



    return G
def find_all_paths(graph, start, end):
    path  = []
    paths = []
    queue = [(start, end, path)]
    while queue:
        start, end, path = queue.pop()
        print('PATH'+ str(path))

        path = path + [start]
        if start == end:
            paths.append(path)
        for node in set(graph[start]).difference(path):
            queue.append((node, end, path))
    return paths

def show_graph_application_patterns(g):
    labels={}
    sizes=[]

    app_nodes = []
    for n, d in g.nodes.items():
        if d['PageName'] == 'APPLICATION':
            app_nodes.append(n)

    patter_sizes = {}
    edge_widths={}
    for app_n in app_nodes:
        try:
            paths_dict = nx.shortest_path(g, source=None, target=app_n)
            for source,path in paths_dict.items():
                prev_node = None
                increment=1
                for p_node in path:
                    if p_node not in patter_sizes:
                        patter_sizes[p_node]=1
                    else:
                        patter_sizes[p_node]+=1
                    if prev_node is not None:
                        tup = (prev_node,p_node)
                        if tup not in edge_widths:
                            edge_widths[tup]=increment
                        else:
                            edge_widths[tup]+=increment
                        increment+=1
                    prev_node = p_node
        except nx.NetworkXNoPath:
            continue

    sizes = []
    n_list=[]
    for n, d in g.nodes.items():
        if d['PageName'] == 'APPLICATION':
            app_nodes.append(n)
            labels[n] = 'APPLICATION'
            sizes.append(patter_sizes[n])
            n_list.append(n)
        elif n in patter_sizes:
            sizes.append(patter_sizes[n])
            n_list.append(n)

    m = max(edge_widths.values())

    e_list = []
    for e in g.edges:
        if e in edge_widths:
            w = edge_widths[e]/m
            if w > 0.1:
                e_list.append(e)

    pos_a = nx.spring_layout(g, iterations=25, scale=100,k=0.8)
    plt.figure(num=None, figsize=(100, 100), dpi=80, facecolor='w', edgecolor='k')
    fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    nx.draw_networkx(g, pos=pos_a, arrows=True, with_labels=True,width=0.5, style='dashed',labels=labels,node_size=sizes,cmap="viridis_r", node_color=sizes,edgelist=e_list,nodelist=n_list)
    fig.savefig('fig/base_graph'+str(int(round(time.time() * 1000))))
    fig.clear()

def filter_baskets(baskets,must_contain):
    ret = []
    for b in baskets:
        for id,name in b:
            if name in must_contain:
                ret.append(b)
    return ret

def contains(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return False


def set_contains(set,value):
    for v in set:
        if v[1] == value:
            return True
    return False

#names=['LocalID','PageID','VisitID','PageName','CatName','CatID','ExtCatName','ExtCatID','TopicName','TopicID','TimeOnPage','PageScore','SequenceNumber']
clicks = pd.read_csv('clicks.csv', sep=',', header=0)
visitors = pd.read_csv('visitors.csv', sep=',', header=0)
print(list(clicks.columns.values))
print('There is ' + str(len(clicks['PageID'].unique())) + ' unique pages')
print('There is ' + str(len(visitors['Referrer'].unique())) + ' unique users')
print(str(len(clicks['PageName'].unique())))


visits_basket = None
users_baskets = None
if os.path.isfile('visits_basket.pickle'):
    with open('visits_basket.pickle', 'rb') as f:
        visits_basket = pickle.load(f)
    with open('users_baskets.pickle', 'rb') as f:
        users_baskets = pickle.load(f)
else:
    visits_basket = create_baskets(clicks)
    users_baskets = create_vistiors_bakets(visits_basket, visitors)
    with open('visits_basket.pickle', 'wb') as f:
        pickle.dump(visits_basket, f)
    with open('users_baskets.pickle', 'wb') as f:
        pickle.dump(users_baskets, f)


sum = 0
u_sum=0
users_baskets_unique = []
for k,v in users_baskets.items():
    sum+=len(v)
    us = set(v)
    u_sum+=len(us)
    ta= list(us)
    ta.insert(0,k[1])
    ta.insert(0,k[0])
    users_baskets_unique.append(ta)

print('Average number of visits per user per day: ' + str(sum/len(users_baskets)))
print('Average number of unique pages visited by user per day: ' + str(u_sum/len(users_baskets)))

print('Number of visits: ' + str(len(visits_basket)))

sum =0
for k,v in visits_basket.items():
    sum+=len(v)

for k,a in users_baskets.items():
    for b in a:
        if (b[1] == 'APPLICATION'):
            print(str(k) + ' ' + str(a))

print('Average number of pages visited during visit: ' + str(sum/len(visits_basket)))

result = None
resultc = None
if os.path.isfile('result.pickle'):
    with open('result.pickle', 'rb') as f:
        result = pickle.load(f)
    with open('resultc.pickle', 'rb') as f:
        resultc = pickle.load(f)
else:
    result, resultc = apriori(users_baskets_unique, 0.01)
    with open('result.pickle', 'wb') as f:
        pickle.dump(result, f)
    with open('resultc.pickle', 'wb') as f:
        pickle.dump(resultc, f)

print('tested sessions: ' + str(len(users_baskets_unique)))

for conversion in [('APPLICATION',0.0,0.01),('CATALOG',0.025,0.85),('DISCOUNT',0.005,0.9), ('HOWTOJOIN',0.01,0.8), ('INSURANCE',0.01,0.8), ('WHOWEARE',0.01,0.8)]:
    G = nx.DiGraph()

    print()
    print(conversion)
    print()
    rules = genereateRules(result,resultc,conversion[2])
    root_node = None
    already_in = set()
    for ss, rule in rules:
        if len(ss) == 1 and set_contains(ss, conversion[0]) and resultc[rule] >= conversion[1]  and  resultc[rule] / resultc[rule - ss] >= conversion[2]:
            print(' confidence: %0.2f' % (resultc[rule] / resultc[rule - ss]) + ' support: %0.2f' % (resultc[rule]) + ' : ' + str(ss) + ' => ' + str(rule - ss))
            for index,b in enumerate(visits_basket.values()):
                if rule.issubset(set(b)) and index not in already_in:
                    already_in.add(index)
                    for i,_ in enumerate(b[0:-1]):
                        n1 = b[i][0]
                        n2 = b[i+1][0]

                        if G.has_node(n1):
                            G.node[n1]['rule_count'] +=1
                        else:
                            G.add_node(n1,rule_count=1,name=b[i][1])

                        if G.has_node(n2):
                            G.node[n2]['rule_count'] +=1
                        else:
                            G.add_node(n2,rule_count=1,name=b[i+1][1])

                        if b[i+1][1] == conversion[0]:
                            root_node = n2

                        if G.has_edge(n1,n2):
                            ed = G.get_edge_data(n1, n2)
                            ed['counter']+=1
                        else:
                            G.add_edge(n1, n2, w=1,counter=1)


    for n in G.nodes:
        sum = 0
        for en in G.in_edges(n):
            ed = G.get_edge_data(en[0], en[1])
            sum += ed['counter']
        G.node[n]['in_count'] = sum

    sum_r = 0
    for n in G.nodes:
        sum_r+=G.node[n]['rule_count']

    avg = sum_r/max([G.number_of_nodes(),1])

    node_to_label = []
    nodes_to_remove = []
    for n in G.nodes:
        if G.node[n]['rule_count'] > (1.25*avg):
            node_to_label.append(n)
        if G.number_of_nodes() > 50:
            if G.node[n]['rule_count'] < (0.85 * avg):
                nodes_to_remove.append(n)

    for n in nodes_to_remove:
        G.remove_node(n)

    for e in G.edges:
        ac = G.node[e[0]]['rule_count']
        bc = G.node[e[1]]['rule_count']
        ed = G.get_edge_data(e[0], e[1])
        ed['w'] = ((ac + bc) * (ac - bc)) / ed['counter']
        #ed['w'] =1000/((G.in_degree(e[0])+G.in_degree(e[1]))+(G.out_degree(e[0])+G.out_degree(e[1])))


    for n in G.nodes:
        sum = 0
        for en in  G.in_edges(n):
            ed = G.get_edge_data(e[0], e[1])
            sum+=ed['counter']
        G.node[n]['in_count'] = sum



    sizes = []
    targets = []
    labels = {}
    for n in G.nodes:
        if G.nodes[n]['name'] == conversion[0]:
            sizes.append(800)
            targets.append(n)
        else:
            sizes.append(max([G.node[n]['rule_count']*3,150]))

    for n in node_to_label:
        labels[n] = str(G.nodes[n]['name']) + '\n' + str(G.node[n]['rule_count'])

    pos_a = nx.spring_layout(G,scale=10,weight='w')
    print(G.number_of_nodes())
    fig = plt.figure(num=None, figsize=(20, 20), dpi=100, facecolor='w', edgecolor='k')
    fig.tight_layout()
    nx.draw_networkx(G, pos=pos_a, arrows=True, with_labels=True,node_size=sizes,cmap = plt.get_cmap("viridis"),node_color=sizes,width=0.3,labels=labels,style='dashed')


    edge_labels = {}
    for a,b in G.in_edges(root_node):
        d = G.get_edge_data(a,b)
        edge_labels[(a,b)] = d['counter']

    nx.draw_networkx_edge_labels(G, pos_a,edge_labels,label_pos=0.8,font_size=10, alpha=0.3)
    nx.draw_networkx_edge_labels(G, pos_a,edge_labels,label_pos=0.2,font_size=10, alpha=0.3)

    fig.savefig('fig/'+str(conversion[0])+'_'+str(int(round(time.time() * 1000))), bbox_inches='tight')
    fig.clear()



g = create_graph(clicks)
pr = nx.pagerank(g, alpha=0.9)
print(nx.number_connected_components(g))
show_graph_application_patterns(g)



