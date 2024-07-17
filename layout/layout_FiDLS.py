from collections import defaultdict
from copy import deepcopy
from queue import PriorityQueue

from qiskit.circuit import Qubit, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target

import networkx as nx
from FiDLS.FiDLS_utils import graph_of_circuit, is_embeddable

class Layout_FiDLS(AnalysisPass):
    def __init__(self, coupling_map):
        super().__init__()

        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map

        if self.coupling_map is not None:
            if not self.coupling_map.is_symmetric:
                if isinstance(coupling_map, CouplingMap):
                    self.coupling_map = deepcopy(self.coupling_map)
                self.coupling_map.make_symmetric()

    def run(self, dag: DAGCircuit):
        """Run the SabreLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        print(f"In Layout FiDLS")

        # layout = Layout()
        mapping_option = "wgt"
        if mapping_option == "wgt":
            layout = self._tau_bsg_()
        elif mapping_option == "top":
            layout = self._tau_bstg()


        for qreg in dag.qregs.values():
            layout.add_register(qreg)

        # Map idle physical qubits to ancilla qubits
        idle_p = [p for p in self.coupling_map.physical_qubits if p not in layout.get_physical_bits()]

        if idle_p:
            qreg = QuantumRegister(len(idle_p), name="ancilla")
            layout.add_register(qreg)
            dag.add_qreg(qreg)
        
            for idx, idle_q in enumerate(idle_p):
                layout[idle_q] = qreg[idx]

        self.property_set['layout'] = layout

    def _tau_bsg_(self, C, G, anchor, stop) -> Layout:
        """Return the weighted subgraph initial mapping

        Args:
            C (list): the input circuit
            G (graph): the architecture graph

        Returns:
            tau (list): the weighted subgraph initial mapping
        """

        def best_wtg_o_ini_mapping(C, G, anchor, stop): #'o' for original
            ''' Return a graph g which is isomorphic to a subgraph of G
                    while maximizing the number of CNOTs in C that correspond to edges in g
                Method: sort the edges according to their weights (the number of CNOTs in C corresponding to each edge);
                        construct a graph by starting with the edge with the largest weight; then consider the edge with the second large weight, ...
                        if in any step the graph is not isomorphic to a subgraph of G, skip this edge and consider the next till all edges are considered.
            
            Args:
                C (list): the input circuit
                G (graph): the architecture graph
                
            Returns:
                g (graph)
                map (dict)
            '''    
            g_of_c = graph_of_circuit(C)
            test = is_embeddable(g_of_c, G, anchor, stop)
            if test[0]:
                #print('The graph of the circuit is embeddable in G')
                return g_of_c, test[1]
            
            edge_wgt_list = list([C.count([e[0],e[1]]) + C.count([e[1],e[0]]), e] for e in g_of_c.edges())
            edge_wgt_list.sort(key=lambda t: t[0], reverse=True) # q[0] weight, q[1] edge
            
            '''Sort the edges reversely according to their weights''' 
            EdgeList = list(item[1] for item in edge_wgt_list)    
            #edge_num = len(EdgeList)
            
            '''We search backward, remove the first edge that makes g not embeddable, 
                    and continue till all edges are evaluated in sequence. '''
                    
            #Hard_Edge_index = 0 # the index of the first hard edge
            g = nx.Graph()
            result = dict()
            # add the first edge into g
            edge = EdgeList[0]
            g.add_edge(edge[0], edge[1])

            #EdgeList_temp = EdgeList[:]
            for edge in EdgeList:           
                g.add_edge(edge[0], edge[1])           
                test = is_embeddable(g, G, anchor, stop)
                if not test[0]:
                    g.remove_edge(edge[0], edge[1])
                    if nx.degree(g, edge[0]) == 0: g.remove_node(edge[0])
                    if nx.degree(g, edge[1]) == 0: g.remove_node(edge[1])
                else:
                    result = test[1]
            return g, result

        result = best_wtg_o_ini_mapping(C, G, anchor, stop)[1] # o in {o,x}

        return result

    def _tau_bstg(self, circuit, ag, anchor, stop) -> Layout:
        return

