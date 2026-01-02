# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 08:31:57 2025

@author: hanna
"""
import numpy as np
import os
import networkx as nx
import re

def read_texts_from_folder(folder_path):
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
    return texts
                
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def network_entropy(G):
    edge_counts = np.array(
        [data['count'] for _, _, data in G.edges(data=True)],
        dtype=float
    )

    if edge_counts.sum() == 0:
        return 0.0

    p = edge_counts / edge_counts.sum()
    H = -np.sum(p * np.log2(p))
    return H

def word_entropy(G, word):
    if word not in G:
        return None

    out_edges = G.out_edges(word, data=True)
    counts = np.array(
        [data['count'] for _, _, data in out_edges],
        dtype=float
    )

    if counts.size == 0 or counts.sum() == 0:
        return 0.0

    p = counts / counts.sum()
    H = -np.sum(p * np.log2(p))
    return H

def all_word_entropies(G):
    return {w: word_entropy(G, w) for w in G.nodes()}

Folder_name="2023"
Output_filename=Folder_name+"_individual.gexf"

texts = read_texts_from_folder(Folder_name)
tokenized_docs = [clean_text(t).split() for t in texts]  
words = [w for doc in tokenized_docs for w in doc]       

dG = nx.DiGraph()

for doc in tokenized_docs:
    for w in doc:
        if w in dG:
            dG.nodes[w]['count'] += 1
        else:
            dG.add_node(w, count=1)

    for w, nw in zip(doc, doc[1:]):
        if not dG.has_node(nw):
            dG.add_node(nw, count=0)
        if dG.has_edge(w, nw):
            dG[w][nw]['count'] += 1
            dG[w][nw]['weight'] = 1 / (dG[w][nw]['count'])
        else:
            dG.add_edge(w, nw, count=1, weight=1.0)

nx.write_gexf(dG, Output_filename)

entropies = all_word_entropies(dG)
top_flexible = sorted(entropies.items(), key=lambda x: x[1], reverse=True)

print("Network entropy:{:.3f}\n".format(network_entropy(dG)))

print(word_entropy(dG,"health"))
print(word_entropy(dG,"people"))
print(word_entropy(dG,"care"))
print(word_entropy(dG,"children"))
print(word_entropy(dG,"community"))
print(word_entropy(dG,"hospital"))
print(word_entropy(dG,"hse"))
print(word_entropy(dG,"patients"))
print(word_entropy(dG,"staff"))
print(word_entropy(dG,"nurses"))
print(word_entropy(dG,"services"))












