from collections import defaultdict
from copy import copy, deepcopy
from numpy import Infinity

from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError, CouplingError
from qiskit.transpiler.target import Target

from utils import (
    coupling_longest_shortest_distance,
    coupling_qubit_neighborhood
)

class Routing_087(TransformationPass):
    def __init__(self, coupling_map, fake_run=False):
        super().__init__()

        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.coupling_map = coupling_map
            self.target = None

        if self.coupling_map is not None and not self.coupling_map.is_symmetric:
            if isinstance(coupling_map, CouplingMap):
                self.coupling_map = deepcopy(self.coupling_map)
            self.coupling_map.make_symmetric()

        self.fake_run = fake_run
        self.dist_matrix = None
    
    def run(self, dag: DAGCircuit):
        if self.coupling_map is None:
            raise TranspilerError("SabreSwap cannot run with coupling_map=None")

        # if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
        #     raise TranspilerError("Sabre swap runs on physical circuits only.")

        num_dag_qubits = len(dag.qubits)
        num_coupling_qubits = self.coupling_map.size()
        if num_dag_qubits < num_coupling_qubits:
            raise TranspilerError(
                f"Fewer qubits in the circuit ({num_dag_qubits}) than the coupling map"
                f" ({num_coupling_qubits})."
                " Have you run a layout pass and then expanded your DAG with ancillas?"
                " See `FullAncillaAllocation`, `EnlargeWithAncilla` and `ApplyLayout`."
            )
        if num_dag_qubits > num_coupling_qubits:
            raise TranspilerError(
                f"More qubits in the circuit ({num_dag_qubits}) than available"
                f" in the coupling map ({num_coupling_qubits})."
                " This circuit cannot be routed to this device."
            )

        print(f"In Routing 087")

        self.dist_matrix = self.coupling_map.distance_matrix
        layout = self.property_set['layout']

        # Start routing algorithm
        best_mapped_dag = dag.copy_empty_like()
        best_layout = None
        swap_option = 1 # better result: 1
        reps = 1

        for _ in range(reps):
            trial_mapped_dag = dag.copy_empty_like()
            trial_layout = layout.copy()
            

            front_layer = dag.front_layer()
            self.applied_predecessors = defaultdict(int)

            for _, input_node in dag.input_map.items():
                for successor in self._successors(input_node, dag):
                    self.applied_predecessors[successor] += 1

            while front_layer:
                curr_gate = front_layer[0]
                v0, v1 = curr_gate.qargs
                p0, p1 = (trial_layout._v2p[v] for v in curr_gate.qargs)

                if self.coupling_map.graph.has_edge(p0, p1):
                    assert self.dist_matrix[p0][p1] == 1

                    self._apply_gate(trial_mapped_dag, curr_gate)
                    front_layer.remove(curr_gate)

                    for successor in self._successors(curr_gate, dag):
                        self.applied_predecessors[successor] += 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)
                else:
                    assert self.dist_matrix[p0][p1] != 1

                    # Find the path from `p0` to `mid_p` and the path from `p1` to `mid_p`
                    best_mid = None
                    mini = Infinity
                    max_shortest_dist = coupling_longest_shortest_distance(self.coupling_map)
                    shortest_path = self.coupling_map.shortest_undirected_path(p0, p1)
                    mid_center = shortest_path[len(shortest_path) // 2]
                    mid_neighbors = coupling_qubit_neighborhood(self.coupling_map, mid_center, range=3)
                    mid_search_list = (p for p in [mid_center] + mid_neighbors if p not in [p0, p1])

                    for _p in mid_search_list:
                        if (
                            abs(self.dist_matrix[p0][_p] - self.dist_matrix[p1][_p]) <= 3 and
                            max_shortest_dist[_p] < mini
                        ):
                            mini = max_shortest_dist[_p]
                            best_mid = _p
                    
                    assert best_mid != None

                    path_p0 = self.coupling_map.shortest_undirected_path(p0, best_mid)
                    path_p1 = self.coupling_map.shortest_undirected_path(p1, best_mid)
                    
                    # Consecutively swap `p0` to `mid_p` (included)
                    for i in range(len(path_p0)-1):
                        _v0 = trial_layout._p2v[path_p0[i]]
                        _v1 = trial_layout._p2v[path_p0[i+1]]
                        swap_node = DAGOpNode(op=SwapGate(), qargs=(_v0, _v1))
                        self._apply_gate(trial_mapped_dag, swap_node)
                        trial_layout.swap(_v0, _v1)

                    # Consecutively swap `p1` to `mid_p` (excluded)
                    for i in range(len(path_p1)-2):
                        _v0 = trial_layout._p2v[path_p1[i]]
                        _v1 = trial_layout._p2v[path_p1[i+1]]
                        swap_node = DAGOpNode(op=SwapGate(), qargs=(_v0, _v1))
                        self._apply_gate(trial_mapped_dag, swap_node)
                        trial_layout.swap(_v0, _v1)

                    assert self.dist_matrix[trial_layout._v2p[v0]][trial_layout._v2p[v1]] == 1

                    self._apply_gate(trial_mapped_dag, curr_gate)
                    front_layer.remove(curr_gate)

                    for successor in self._successors(curr_gate, dag):
                        self.applied_predecessors[successor] += 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)
                
            # If this is the first trial
            if len(best_mapped_dag.op_nodes()) == 0:
                best_mapped_dag = copy(trial_mapped_dag)
                best_layout = trial_layout.copy()
            # If current trial result has less swap gates than the best record, update the record
            elif len(trial_mapped_dag.op_nodes()) < len(best_mapped_dag.op_nodes()):
                best_mapped_dag = copy(trial_mapped_dag)
                best_layout = trial_layout.copy()

        self.property_set['final_layout'] = best_layout
        return best_mapped_dag
        
    def _apply_gate(self, mapped_dag: DAGCircuit, node: DAGOpNode):
        mapped_dag.apply_operation_back(node.op, node.qargs, node.cargs)

    def _successors(self, node, dag):
        for _, successor, edge_data in dag.edges(node):
            if not isinstance(successor, DAGOpNode):
                continue
            if isinstance(edge_data, Qubit):
                yield successor
    
    def _is_resolved(self, node):
        """Return True if all of a node's predecessors in dag are applied."""
        return self.applied_predecessors[node] == len(node.qargs)