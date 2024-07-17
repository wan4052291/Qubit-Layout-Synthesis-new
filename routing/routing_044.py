from collections import defaultdict
from copy import copy, deepcopy

from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target

from utils import dag_qubit_interactivity

class Routing_044(TransformationPass):
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

        print(f"In Routing 044")

        self.dist_matrix = self.coupling_map.distance_matrix
        layout = self.property_set['layout']

        # Start routing algorithm
        mapped_dag = dag.copy_empty_like()
        front_layer = dag.front_layer()
        self.applied_predecessors = defaultdict(int)
        v_interacts = dag_qubit_interactivity(dag)

        for _, input_node in dag.input_map.items():
            for successor in self._successors(input_node, dag):
                self.applied_predecessors[successor] += 1

        while front_layer:
            curr_gate = front_layer[0]
            v0, v1 = curr_gate.qargs
            p0, p1 = (layout._v2p[v] for v in curr_gate.qargs)

            if self.coupling_map.graph.has_edge(p0, p1):
                self._apply_gate(mapped_dag, curr_gate)
                front_layer.remove(curr_gate)

                for successor in self._successors(curr_gate, dag):
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        front_layer.append(successor)

            else:
                path = self.coupling_map.shortest_undirected_path(p0, p1)
                rev_path = list(reversed(path))

                if len(v_interacts[v0]) > len(v_interacts[v1]):
                    # Swap `v1` to `v0`
                    for i in range(len(path)-2):
                        _v0, _v1 = (layout._p2v[p] for p in (rev_path[i], rev_path[i+1]))
                        swap_node = DAGOpNode(op=SwapGate(), qargs=(_v0, _v1))
                        self._apply_gate(mapped_dag, swap_node)
                        layout.swap(_v0, _v1)
                else:
                    # Swap `v0` to `v1`
                    for i in range(len(path)-2):
                        _v0, _v1 = (layout._p2v[p] for p in (path[i], path[i+1]))
                        swap_node = DAGOpNode(op=SwapGate(), qargs=(_v0, _v1))
                        self._apply_gate(mapped_dag, swap_node)
                        layout.swap(_v0, _v1)
                
                self._apply_gate(mapped_dag, curr_gate)
                front_layer.remove(curr_gate)

                for successor in self._successors(curr_gate, dag):
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