from collections import deque
from copy import deepcopy

from qiskit.circuit import QuantumRegister, Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target

class Layout_040(AnalysisPass):
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

        print(f"In Layout 040")

        layout = Layout()
        logical_qubits = deque()
        visited_v = dict.fromkeys(dag.qubits, False)

        for gate in dag.topological_op_nodes():
            v0, v1 = gate.qargs

            if not visited_v[v0]:
                logical_qubits.append(v0)
                visited_v[v0] = True

            if not visited_v[v1]:
                logical_qubits.append(v1)
                visited_v[v1] = True
        
        # Since the logical qubits operated in DAG nodes may not be whole qubits
        # in the circuit, so we add those remaining qubits into `logical_qubits`
        for v in dag.qubits:
            if not visited_v[v]:
                logical_qubits.append(v)
                visited_v[v] = True
        
        if len(logical_qubits) < len(self.coupling_map.physical_qubits):
            size = len(self.coupling_map.physical_qubits) - len(logical_qubits)
            qreg = QuantumRegister(size, name="ancilla")
            dag.add_qreg(qreg)

            for ancilla in qreg:
                logical_qubits.append(ancilla)
        
        deq = deque()
        visited_p = dict.fromkeys(self.coupling_map.physical_qubits, False)
        src = self.coupling_map.physical_qubits[0]
        deq.append(src)
        visited_p[src] = True

        while len(deq) > 0:
            curr_p = deq.pop()

            if len(logical_qubits) > 0:
                layout.add(logical_qubits.pop(), curr_p)
            
            for neighbor in self.coupling_map.neighbors(curr_p):
                if not visited_p[neighbor]:
                    deq.append(neighbor)
                    visited_p[neighbor] = True

        for qreg in dag.qregs.values():
            layout.add_register(qreg)
        
        self.property_set['layout'] = layout