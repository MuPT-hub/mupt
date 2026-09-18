"""For converting back-and-forth between anytree and networkx DiGraphs"""

from anytree.node import NodeMixin
from networkx import DiGraph


def tree_to_networkx(node : NodeMixin) -> DiGraph:
    '''Convert a tree into a directed NetworkX graph'''
    raise NotImplementedError

def networkx_to_tree(digraph : DiGraph) -> NodeMixin:
    '''Convert a tree into a directed NetworkX graph'''
    raise NotImplementedError