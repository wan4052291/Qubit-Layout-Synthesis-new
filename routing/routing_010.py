from copy import copy, deepcopy

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target

class Routing_010(TransformationPass):
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

        print(f"In Routing 010")

        self.dist_matrix = self.coupling_map.distance_matrix
        layout = self.property_set['layout']

        # Start routing algorithm
        mapped_dag = dag.copy_empty_like()

        for gate in dag.topological_op_nodes():
            v0, v1 = gate.qargs
            p0 = layout._v2p[v0]
            p1 = layout._v2p[v1]

            if p1 in self.coupling_map.neighbors(p0):
                self._apply_gate(mapped_dag, gate)
            else:
                path = self.coupling_map.shortest_undirected_path(p0, p1)

                for i in range(len(path)-2):
                    _v0 = layout._p2v[path[i]]
                    _v1 = layout._p2v[path[i+1]]
                    swap_node = DAGOpNode(op=SwapGate(), qargs=(_v0, _v1))
                    self._apply_gate(mapped_dag, swap_node)
                    layout.swap(_v0, _v1)
                
                self._apply_gate(mapped_dag, gate)

        self.property_set['final_layout'] = layout
        return mapped_dag

    def _apply_gate(self, mapped_dag: DAGCircuit, node: DAGOpNode):
        mapped_dag.apply_operation_back(node.op, node.qargs, node.cargs)