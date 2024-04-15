from rtree import index
import numpy as np

from .vptree import VPTree

class Node:
    def __init__(self, coords, idx = -1):
        self.p = np.array(coords)
        self.parent = None
        self.cost = np.inf
        self.idx = idx

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.p[i]

    def __repr__(self):
        return 'Node({}, {})'.format(self.p,self.cost)

def euclidean(p1, p2):
    return np.linalg.norm(p2.p - p1.p)

class VpTree:
    def __init__(self, dim):
        self.dim = dim
        self.node_list = []
        self.idx = VPTree([Node([0.0, 0.0])], euclidean)
        self.len = 0

    def add(self, new_node):
        '''add nodes to tree'''
        if new_node.cost != np.inf:
            new_node.idx = self.len
            self.node_list.append(new_node)

            self.idx = VPTree(self.node_list, euclidean)
            self.len += 1

    def k_nearest(self, node, k):
        '''Returns k-nearest nodes to the given node'''
        near_ids = self.idx.get_n_nearest_neighbors(node, k)
        for r in near_ids:
            yield r[1]

    def nearest(self, node):
        '''Returns nearest node to the given node'''
        r = self.idx.get_nearest_neighbor(node)
        return r[1]

    def all(self):
        return self.node_list


#Rtree to store Nodes
class Rtree:
    def __init__(self,dim):
        self.dim = dim
        self.node_list = []
        self.idx = self.get_tree(dim)
        self.len = 0

    @staticmethod
    def get_tree(dim):
        '''Initialise the tree'''
        p = index.Property()
        p.dimension = dim
        p.dat_extension = 'data'
        p.idx_extension = 'index'
        return index.Index(properties=p)

    def add(self,new_node):
        '''add nodes to tree'''
        self.node_list.append(new_node)
        self.idx.insert(self.len,(*new_node.p,))
        self.len += 1

    def k_nearest(self,node,k):
        '''Returns k-nearest nodes to the given node'''
        near_ids = self.idx.nearest((*node.p,),k)
        for i in near_ids:
            yield self.node_list[i]

    def nearest(self,node):
        '''Returns nearest node to the given node'''
        near_ids = self.idx.nearest((*node.p,),1)
        id = list(near_ids)[0]
        return self.node_list[id]

    def all(self):
        return self.node_list
