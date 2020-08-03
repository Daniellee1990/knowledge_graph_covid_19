import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

def draw_kg(pairs,c1='red',c2='blue',c3='orange'):
    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',
            create_using=nx.MultiDiGraph())
  
    node_deg = nx.degree(k_graph)
    # print('node degree: {}'.format(node_deg))
    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(10, 10), dpi=80)
    
    nx.draw_networkx(
        k_graph,
        node_size=400,
        arrowsize=30,
        linewidths=3,
        pos=layout,
        edge_color=c1,
        edgecolors=c2,
        node_color=c3,
    )

    labels = dict(zip(list(zip(pairs.subject, pairs.object)),
                  pairs['relation'].tolist()))
    
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,
                                 font_color='red')
    
    plt.axis('off')
    plt.show()


relation_file = 'data/kg-triplet/entity_relations.csv'
type_file = 'data/kg-triplet/entity_types.csv'

relation_df = pd.read_csv(relation_file)
type_df = pd.read_csv(type_file)
df = pd.concat([relation_df, type_df], keys=['subject', 'relation', 'object'], ignore_index=True)

#print(df['relation'])

draw_kg(df)

