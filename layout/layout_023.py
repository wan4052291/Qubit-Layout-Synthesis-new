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

class Layout_023(AnalysisPass):
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
        
        print(f"In Layout 023")

        layout = Layout()
        sorted_coupling_qubits = self._sort_coupling_qubits()
        sorted_dag_qubits = self._sort_dag_qubits(dag)
        v_interacts = self._get_dag_qubit_interactivity(dag)

        for v in sorted_dag_qubits:
            for p in sorted_coupling_qubits:
                if v not in layout.get_virtual_bits() and p not in layout.get_physical_bits():
                    layout.add(v, p)

                    unmapped_v_neighbors_idx = list()
                    unmapped_p_neighbors = list()

                    for v_neighbor in v_interacts[v]:
                        if v_neighbor not in layout.get_virtual_bits():
                            unmapped_v_neighbors_idx.append(
                                (len(v_interacts[v_neighbor]), dag.qubits.index(v_neighbor))
                            )
                    
                    for p_neighbor in self.coupling_map.neighbors(p):
                        if p_neighbor not in layout.get_physical_bits():
                            unmapped_p_neighbors.append((len(self.coupling_map.neighbors(p)), p_neighbor))

                    unmapped_v_neighbors_idx.sort(reverse=True)
                    unmapped_p_neighbors.sort(reverse=True)

                    for (_, _v_idx), (_, _p) in zip(unmapped_v_neighbors_idx, unmapped_p_neighbors):
                        layout.add(dag.qubits[_v_idx], _p)

        # Map idle physical qubits to ancilla qubits
        idle_p = [p for p in self.coupling_map.physical_qubits if p not in layout.get_physical_bits()]

        if idle_p:
            qreg = QuantumRegister(len(idle_p), name="ancilla")
            layout.add_register(qreg)
            dag.add_qreg(qreg)
        
            for idx, p in enumerate(idle_p):
                layout[p] = qreg[idx]
        

        print(layout)
        self.property_set['layout'] = layout

    def _get_dag_qubit_interactivity(self, dag: DAGCircuit) -> dict[Qubit, set]:
        """Iterate all logical qubits, and collect whom current qubit have interactivity with.

        Args:
            dag (DAGCircuit): DAG to provide gate infos.
        
        Returns:
            dict[Qubit, set]: every key is a `dag` qubit, and the corresponding value is a set of
            logical qubits that have interactivity with the qubit in key.
        """
        interactivity = dict.fromkeys(dag.qubits, set())

        for gate in dag.op_nodes():
            if gate.op.num_qubits == 2:
                v0, v1 = gate.qargs
                interactivity[v0].add(v1)
                interactivity[v1].add(v0)
                
        return interactivity

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

        return [v for v, _ in sorted(interact.items(), key=lambda x: x[1], reverse=True)]

    def _sort_coupling_qubits(self) -> list[int]:
        """Sort physical qubits of coupling map according to each qubit's degree.
        The first element is a physical qubit with the most degree.
        """
        degree = dict.fromkeys(self.coupling_map.physical_qubits, 0)

        for p in self.coupling_map.physical_qubits:
            degree[p] = len(self.coupling_map.neighbors(p))

        return [p for p, _ in sorted(degree.items(), key=lambda x: x[1], reverse=True)]