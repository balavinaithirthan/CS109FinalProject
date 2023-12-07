import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import numpy as np
import math
import copy

def expectation_of_trust(distribution):
  probs = list(distribution.values())
  possible_values = np.arange(0, 1, 0.01)
  expectation = sum(x * p for x, p in zip(possible_values, probs))
  return expectation


def update_edges(graph):
    for source, target, data in graph.edges(data=True):
        dict_values = copy.deepcopy(graph[source][target]['values'])
        trust_values = list(dict_values.values())
        probability_of_success = expectation_of_trust(dict_values)
        observation = np.random.choice([0, 1], p=[1 - probability_of_success, probability_of_success])
        new_distribution = calculate_new_dist(dict_values, observation)
        graph[source][target]['values'] = new_distribution


def calculate_new_dist(belief, observation):
    # loop over every value in the support of the belief RV
    for a in belief:
        prior_a = belief[a]
        likelihood = calc_likelihood(a, observation)
        belief[a] = prior_a * likelihood
    return normalize(belief)
    

def normalize(belief):
    total = belief_sum(belief)
    sum = 0
    for key in belief:
        belief[key] /= total
    return belief

def belief_sum(belief):
    total = 0
    for key in belief:
        total += belief[key]
    return total


def calc_likelihood(trust_value, obs):
    # returns P(obs | A = a) using Item Response Theory
    p_correct_true = sigmoid(trust_value + 1)
    if obs == 1:
        return p_correct_true
    else:
        return 1 - p_correct_true

def sigmoid(x):
    # the classic squashing function. All outputs are [0,1]
    return 1 / (1 + math.exp(-x))


def initialize_user_trust_graphs(users, nodes, intial_prior):
    graphs = []
    for user in users:
        G = nx.DiGraph()
        G.name = user
        i = 1
        for node in nodes:
          if user == str(i):
            i += 1
            continue
          G.add_edge(user, node, values=initial_prior)
          i += 1
        graphs.append(G)
    return graphs

def time_updates(graphs):
  for graph in graphs:
      update_edges(graph)
  return graphs


def printGraphInfo(graphs):
  for graph in graphs:
    for source, target, data in graph.edges(data=True):
      print(source, target, data['values'])
    break

def anaylze_data(graphs):
  user_trusts = {}
  for graph in graphs:
    user_trusts[graph.name] = 0
  for graph in graphs:
    user = graph.name
    for source, target, data in graph.edges(data=True):
      dict_values = data['values']
      trust_values = list(data['values'].values())
      trust_val = expectation_of_trust(dict_values)
      user_trusts[target] += trust_val
  for user in user_trusts:
    user_trusts[user] /= (len(user_trusts) - 1)
  print(user_trusts)
