from collections import defaultdict
from copy import copy, deepcopy

from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target

from utils import dag_qubit_interactivity

class Routing_023(TransformationPass):
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

        print(f"In Routing 023")

        self.dist_matrix = self.coupling_map.distance_matrix
        layout = self.property_set['layout']

        # Start routing algorithm
        mapped_dag = dag.copy_empty_like()
        sorted_front_layer = self._sort_front_layer(dag.front_layer(), layout)
        self.applied_predecessors = defaultdict(int)
        v_interacts = dag_qubit_interactivity(dag)

        for _, input_node in dag.input_map.items():
            for successor in self._successors(input_node, dag):
                self.applied_predecessors[successor] += 1

        while sorted_front_layer:
            closest_gate = sorted_front_layer[0]
            v0, v1 = closest_gate.qargs
            p0, p1 = (layout._v2p[v] for v in closest_gate.qargs)

            if self.coupling_map.graph.has_edge(p0, p1):
                self._apply_gate(mapped_dag, closest_gate)
                sorted_front_layer.remove(closest_gate)

                for successor in self._successors(closest_gate, dag):
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        sorted_front_layer.append(successor)

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

                self._apply_gate(mapped_dag, closest_gate)
                sorted_front_layer.remove(closest_gate)

                for successor in self._successors(closest_gate, dag):
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        sorted_front_layer.append(successor)
                    
            sorted_front_layer = self._sort_front_layer(sorted_front_layer, layout)

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

    def _sort_front_layer(self, front_layer: list[DAGOpNode], layout: Layout) -> list[DAGOpNode]:
        """Sort gates in `front_layer` based on the distance between its corresponding physical qubits.

        Args:
            front_layer (list[DAGOpNode]): the front layer to be sorted.
            layout (Layout): a mapping of virtual qubits to physical qubits.

        Returns:
            list[DAGOpNode]: The sorted front layer. The first element 
        """
        front_layer_dist = list()

        for gate in front_layer:
            p0, p1 = (layout._v2p[v] for v in gate.qargs)
            front_layer_dist.append(int(self.dist_matrix[p0][p1]))

        return [gate for _, gate in sorted(zip(front_layer_dist, front_layer))]