from collections import defaultdict
from copy import copy, deepcopy
from numpy import Infinity
from queue import Queue

from qiskit.circuit import QuantumRegister, Qubit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target

from utils import (
    coupling_longest_shortest_distance,
    coupling_qubit_neighborhood,
    dag_qubit_interactivity,
    dag_qubit_distance_matrix
)

class Layout_0872(AnalysisPass):
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

        self.dist_matrix = self.coupling_map.distance_matrix

    def run(self, dag: DAGCircuit):
        """Run the SabreLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        print(f"In Layout 087")

        best_operations = list()
        best_layout = None
        reps = 2
        layout = self._mapping_2(dag)
        # idle_p = [p for p in self.coupling_map.physical_qubits if p not in layout.get_physical_bits()]
        # if idle_p:
        #     qreg = QuantumRegister(len(idle_p), name="q")
        #     layout.add_register(qreg)
        #     dag.add_qreg(qreg)
        
        #     for idx, p in enumerate(idle_p):
        #         layout[p] = qreg[idx]

        # for mapping_option in [1, 2]:
        #     if mapping_option == 1:
        #         layout = self._mapping_1(dag)
        #     elif mapping_option == 2:
        #         layout = self._mapping_2(dag)

        #     for qreg in dag.qregs.values():
        #         layout.add_register(qreg)
            
        #     # Map idle physical qubits to ancilla qubits
        #     idle_p = [p for p in self.coupling_map.physical_qubits if p not in layout.get_physical_bits()]

        #     if idle_p:
        #         qreg = QuantumRegister(len(idle_p), name="ancilla")
        #         layout.add_register(qreg)
        #         dag.add_qreg(qreg)
            
        #         for idx, p in enumerate(idle_p):
        #             layout[p] = qreg[idx]
            
        #     for _ in range(reps):
        #         trial_operations = list()
        #         trial_layout = layout.copy()

        #         front_layer = dag.front_layer()
        #         self.applied_predecessors = defaultdict(int)

        #         for _, input_node in dag.input_map.items():
        #             for successor in self._successors(input_node, dag):
        #                 self.applied_predecessors[successor] += 1

        #         while front_layer:
        #             curr_gate = front_layer[0]
        #             v0, v1 = curr_gate.qargs
        #             p0, p1 = (trial_layout._v2p[v] for v in curr_gate.qargs)

        #             if self.coupling_map.graph.has_edge(p0, p1):
        #                 assert self.dist_matrix[p0][p1] == 1
        #                 trial_operations.append({"name": curr_gate.op.name, "qargs": curr_gate.qargs})
        #                 front_layer.remove(curr_gate)

        #                 for successor in self._successors(curr_gate, dag):
        #                     self.applied_predecessors[successor] += 1
        #                     if self._is_resolved(successor):
        #                         front_layer.append(successor)
        #             else:
        #                 assert self.dist_matrix[p0][p1] != 1

        #                 # Find the path from `p0` to `mid_p` and the path from `p1` to `mid_p`
        #                 best_mid = None
        #                 mini = Infinity
        #                 max_shortest_dist = coupling_longest_shortest_distance(self.coupling_map)
        #                 shortest_path = self.coupling_map.shortest_undirected_path(p0, p1)
        #                 mid_center = shortest_path[len(shortest_path) // 2]
        #                 mid_neighbors = coupling_qubit_neighborhood(self.coupling_map, mid_center, range=3)
        #                 mid_search_list = (p for p in [mid_center] + mid_neighbors if p not in [p0, p1])

        #                 for _p in mid_search_list:
        #                     if (
        #                         abs(self.dist_matrix[p0][_p] - self.dist_matrix[p1][_p]) <= 3 and
        #                         max_shortest_dist[_p] < mini
        #                     ):
        #                         mini = max_shortest_dist[_p]
        #                         best_mid = _p
                        
        #                 assert best_mid != None
        #                 path_p0 = self.coupling_map.shortest_undirected_path(p0, best_mid)
        #                 path_p1 = self.coupling_map.shortest_undirected_path(p1, best_mid)
                        
        #                 # Consecutively swap `p0` to `mid_p` (included)
        #                 for i in range(len(path_p0)-1):
        #                     _v0, _v1 = (trial_layout._p2v[p] for p in (path_p0[i], path_p0[i+1]))
        #                     trial_operations.append({"name": "swap", "qargs": (_v0, _v1)})
        #                     trial_layout.swap(_v0, _v1)

        #                 # Consecutively swap `p1` to `mid_p` (excluded)
        #                 for i in range(len(path_p1)-2):
        #                     _v0, _v1 = (trial_layout._p2v[p] for p in (path_p1[i], path_p1[i+1]))
        #                     trial_operations.append({"name": "swap", "qargs": (_v0, _v1)})
        #                     trial_layout.swap(_v0, _v1)

        #                 assert self.dist_matrix[trial_layout._v2p[v0]][trial_layout._v2p[v1]] == 1
        #                 trial_operations.append({"name": curr_gate.op.name, "qargs": curr_gate.qargs})
        #                 front_layer.remove(curr_gate)

        #                 for successor in self._successors(curr_gate, dag):
        #                     self.applied_predecessors[successor] += 1
        #                     if self._is_resolved(successor):
        #                         front_layer.append(successor)
                
        #         # If there are swap gate before the first CX gate, then swaps
        #         while trial_operations:
        #             op = trial_operations[0]

        #             if op["name"] == "swap":
        #                 trial_operations.pop(0)
        #                 layout.swap(*op["qargs"])
        #             else:
        #                 break
                
        #         # If this is the first trial
        #         if len(best_operations) == 0:
        #             best_operations = copy(trial_operations)
        #             best_layout = layout.copy()
        #         # If this trial result has less swap gates than the best record
        #         elif len(trial_operations) < len(best_operations):
        #             best_operations = copy(trial_operations)
        #             best_layout = layout.copy()
                
        #         # Update current `initial_layout` with `trial_layout`
        #         layout = trial_layout
                
        self.property_set['layout'] = layout

    def _successors(self, node, dag):
        for _, successor, edge_data in dag.edges(node):
            if not isinstance(successor, DAGOpNode):
                continue
            if isinstance(edge_data, Qubit):
                yield successor
    
    def _is_resolved(self, node):
        """Return True if all of a node's predecessors in dag are applied."""
        return self.applied_predecessors[node] == len(node.qargs)
        
    def _mapping_1(self, dag: DAGCircuit):
        """The first mapping method 
        
        Args:
            dag (DAGCircuit): DAG to find layout for.
        """
        # Get the physical qubit with smallest maximum value of shortest distance 
        best_p = self._sort_coupling_qubits()[0]
        # Get the logical qubit with the most interactivity
        best_v = self._sort_dag_qubits(dag)[0]
        layout = Layout({best_v: best_p})
        v_interact = dag_qubit_interactivity(dag)
        v_ranks = self._get_dag_qubit_rank(dag)
        log_dist_matrix = dag_qubit_distance_matrix(dag)

        # For each dag qubit, sort its neighbors based on their ranks
        for v in dag.qubits:
            neighbor_ranks = list()
            neighbor_idx = list()

            for neighbor in v_interact[v]:
                neighbor_ranks.append(v_ranks[neighbor])
                neighbor_idx.append(dag.qubits.index(neighbor))
            
            v_interact[v] = [
                dag.qubits[idx] for _, idx in sorted(zip(neighbor_ranks, neighbor_idx))
            ]

        p_que = Queue()
        visited_p = dict.fromkeys(self.coupling_map.physical_qubits, False)
        visited_v = dict.fromkeys(dag.qubits, False)

        p_que.put(best_p)
        visited_p[best_p] = True
        visited_v[best_v] = True

        while not p_que.empty():
            curr_p = p_que.get()

            # Find unmapped neighbor of current physical qubit
            for p_neighbor in self.coupling_map.neighbors(curr_p):
                if not visited_p[p_neighbor]:
                    curr_v = layout._p2v[curr_p]

                    # Find unmapped interactivity of current logical qubit
                    # If found, map it to the current p_neighbor
                    for v_neighbor in v_interact[curr_v]:
                        if not visited_v[v_neighbor]:
                            visited_v[v_neighbor] = True
                            visited_p[p_neighbor] = True
                            layout.add(v_neighbor, p_neighbor)
                            p_que.put(p_neighbor)
                            break
        
        # print(f"after bfs: {layout}")

        # Sort unmapped logical and physical qubits by their distance to `best_v` and `best_p` respectively
        unmapped_v_idx = (dag.qubits.index(v) for v in visited_v.keys() if visited_v[v] == False)
        unmapped_p = (p for p in visited_p.keys() if visited_p[p] == False)
        dist_to_best_v = (log_dist_matrix[v_idx][dag.qubits.index(best_v)] for v_idx in unmapped_v_idx)
        sorted_unmapped_v_idx = (v_idx for _, v_idx in sorted(zip(dist_to_best_v, unmapped_v_idx)))
        dist_to_best_p = (self.dist_matrix[p][best_p] for p in unmapped_p)
        sorted_unmapped_p = (p for _, p in sorted(zip(dist_to_best_p, unmapped_p)))

        for v_idx, p in zip(sorted_unmapped_v_idx, sorted_unmapped_p):
            layout.add(dag.qubits[v_idx], p)

        return layout
    
    def _mapping_2(self, dag: DAGCircuit):
        v_idx = list()
        weights = list()
        ranks = self._get_dag_qubit_rank(dag)
        v_interacts = dag_qubit_interactivity(dag)

        for v in dag.qubits:
            v_idx.append(dag.qubits.index(v))
            weights.append(len(v_interacts[v]) / ranks[v])
        
        sorted_v = [dag.qubits[idx] for _, idx in sorted(zip(weights, v_idx), reverse=True)]
        layout = Layout()
        best_p = self._sort_coupling_qubits()[0]
        curr_v_idx = 0
        q = Queue()
        visited_p = dict.fromkeys(self.coupling_map.physical_qubits, False)
        q.put(best_p)
        visited_p[best_p] = True

        while not q.empty():
            curr_p = q.get()
            layout.add(sorted_v[curr_v_idx], curr_p)
            curr_v_idx += 1
            if curr_v_idx > len(sorted_v)-1:
                break

            for neighbor in self.coupling_map.neighbors(curr_p):
                if not visited_p[neighbor]:
                    visited_p[neighbor] = True
                    q.put(neighbor)
        
        return layout

    def _get_dag_qubit_rank(self, dag: DAGCircuit):
        rank = 1
        ranks = dict.fromkeys(dag.qubits, 1)
        front_layer = dag.front_layer()
        self.applied_predecessors = defaultdict(int)

        for _, input_node in dag.input_map.items():
            for successor in self._successors(input_node, dag):
                self.applied_predecessors[successor] += 1

        while front_layer:
            for gate in front_layer:
                v0, v1 = gate.qargs

                if ranks[v0] == 1:
                    ranks[v0] = rank
                if ranks[v1] == 1:
                    ranks[v1] = rank
                
                front_layer.remove(gate)
                for successor in self._successors(gate, dag):
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        front_layer.append(successor)
            
            rank += 1
        
        return ranks
    
    def _sort_dag_qubits(self, dag: DAGCircuit) -> list[Qubit]:
        """Sort logical qubits of `dag` according to each qubit's interactivity.
        The first element is a logical qubit with the most interactivity.
        
        Args:
            dag (DAGCircuit): DAG to provide logical qubits and gate infos.
        """
        interact = dict.fromkeys(dag.qubits, 0)

        for gate in dag.op_nodes():
            if gate.op.num_qubits == 2:
                v0, v1 = gate.qargs
                interact[v0] += 1
                interact[v1] += 1

        return [q for q, _ in sorted(interact.items(), key=lambda x: x[1], reverse=True)]
    
    def _sort_coupling_qubits(self) -> list[int]:
        """Sort physical qubits of coupling map according to each qubit's maximum
        value of shortest distances between itself and all the other qubits.
        The first element is a physical qubit with the smallest maximum value.
        """
        max_shortest_dist = dict.fromkeys(self.coupling_map.physical_qubits, 0)

        for p0 in self.coupling_map.physical_qubits:
            for p1 in self.coupling_map.physical_qubits:
                max_shortest_dist[p0] = max(max_shortest_dist[p0], int(self.dist_matrix[p0][p1]))

        return [q for q, _ in sorted(max_shortest_dist.items(), key=lambda x: x[1])]