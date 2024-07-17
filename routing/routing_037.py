from collections import defaultdict
from copy import copy, deepcopy

from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target

class Routing_037(TransformationPass):
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
                f"More qubits in the circuit ({num_dag_qubits}) than available in the coupling map"
                f" ({num_coupling_qubits})."
                " This circuit cannot be routed to this device."
            )

        print(f"In Routing 037")

        self.dist_matrix = self.coupling_map.distance_matrix
        layout = self.property_set['layout']

        # Start routing algorithm
        mapped_dag = dag.copy_empty_like()
        front_layer = dag.front_layer()
        self.applied_predecessors = defaultdict(int)

        for _, input_node in dag.input_map.items():
            for successor in self._successors(input_node, dag):
                self.applied_predecessors[successor] += 1

        while front_layer:
            execute_gate_list = []

            for gate in front_layer:
                if len(gate.qargs) == 2:
                    v0, v1 = gate.qargs
                    p0, p1 = (layout._v2p[v] for v in gate.qargs)

                    if self.coupling_map.graph.has_edge(p0, p1):
                        execute_gate_list.append(gate)

            if execute_gate_list:

                for gate in execute_gate_list:
                    self._apply_gate(mapped_dag, gate)
                    front_layer.remove(gate)

                    for successor in self._successors(gate, dag):
                        self.applied_predecessors[successor] += 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)

                continue

            # After all executable gates are exhausted, heuristically find
            # the min cost to execute a gate and swap along the shortest path.
            swap_candidates = list()

            for gate in front_layer:
                p0, p1 = (layout._v2p[v] for v in gate.qargs)

                if self.dist_matrix[p0][p1] > 1:
                    swap_candidates.append((self.dist_matrix[p0][p1], gate))
            
            min_cost_candidate = sorted(swap_candidates)[0][1]
            p0, p1 = (layout._v2p[v] for v in min_cost_candidate.qargs)
            path = self.coupling_map.shortest_undirected_path(p0, p1)

            for i in range(len(path)-2):
                _v0, _v1 = (layout._p2v[p] for p in (path[i], path[i+1]))
                swap_node = DAGOpNode(op=SwapGate(), qargs=(_v0, _v1))
                self._apply_gate(mapped_dag, swap_node)
                layout.swap(_v0, _v1)
            
            self._apply_gate(mapped_dag, min_cost_candidate)
            front_layer.remove(min_cost_candidate)

            for successor in self._successors(min_cost_candidate, dag):
                self.applied_predecessors[successor] += 1
                if self._is_resolved(successor):
                    front_layer.append(successor)

        self.property_set['final_layout'] = layout
        return mapped_dag
        
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