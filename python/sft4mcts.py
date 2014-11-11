import networkx as nx

class SFT4mcts(object):
    def __init__(self,name,state,backgroundrates,p,duration=0,external=False):
        self.name = name
        self.state = state
        self.infect_duration = duration
        self.sends_to = backgroundrates.keys()
        self.backgroundrates = backgroundrates
        self.infectionrate = p
        self.external = external

class SFTnet4mcts(object):
    def __init__(self,name,nodes,draw=False):
        self.name = name
        self.allnodenames = tuple([n.name for n in nodes])
        self.nodes = dict(zip(self.allnodenames,nodes))
        self.states0 = dict(zip(self.allnodenames,[nd.state for nd in self.nodes.values()]))
        self.externals = tuple([n for n in nodes if n.external == True])
        self.internals = tuple([n for n in nodes if n.external == False])
        self.graph = self.create_network(draw=draw)
    def create_network(self,draw):
        """Returns directed network.
        """
        nodes = self.nodes
        G = nx.DiGraph()
        G.add_nodes_from(nodes.keys())
        edges = []
        for nd in nodes.values():
            for outnode in nd.sends_to:
                edges.append((nd.name,outnode))
        G.add_edges_from(edges)
        if draw == True:
            colors = []
            nds = []
            for (key,val) in node_s0.iteritems():
                nds.append(key)
                if val == 'Infected':
                    colors.append('r')
                elif val == 'Normal':
                    colors.append('g')
            assert len(nds) == len(colors)
            nx.draw(G, nodelist = nds, node_color = colors)
        return G
