from collections import defaultdict
from copy import deepcopy
from queue import Queue, PriorityQueue

from qiskit.circuit import QuantumRegister, Qubit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target

class Layout_037(AnalysisPass):
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
        
        print(f"In Layout 037")

        layout = Layout.generate_trivial_layout(dag.qregs['q'])
        
        # Map idle physical qubits to ancilla qubits
        idle_p = [p for p in self.coupling_map.physical_qubits if p not in layout.get_physical_bits()]

        if idle_p:
            qreg = QuantumRegister(len(idle_p), name="ancilla")
            layout.add_register(qreg)
            dag.add_qreg(qreg)
        
            for idx, p in enumerate(idle_p):
                layout[p] = qreg[idx]

        # Start layout algorithm
        front_layer = dag.front_layer()
        self.applied_predecessors = defaultdict(int)
        operations = list()

        for _, input_node in dag.input_map.items():
            for successor in self._successors(input_node, dag):
                self.applied_predecessors[successor] += 1

        while front_layer:
            curr_gate = front_layer[0]
            p0, p1 = (layout._v2p[v] for v in curr_gate.qargs)

            if self.coupling_map.graph.has_edge(p0, p1):
                front_layer.remove(curr_gate)

                for successor in self._successors(curr_gate, dag):
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        front_layer.append(successor)
            else:
                for neighbor in self.coupling_map.neighbors(p0):
                    if int(self.dist_matrix[p1][neighbor]) < int(self.dist_matrix[p0][p1]):
                        layout.swap(p0, neighbor)
                        operations.append((p0, neighbor))
                        break
                
                front_layer.remove(curr_gate)

                for successor in self._successors(curr_gate, dag):
                    self.applied_predecessors[successor] += 1
                    if self._is_resolved(successor):
                        front_layer.append(successor)
        
        for _p0, _p1 in reversed(operations):
            layout.swap(_p0, _p1)

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