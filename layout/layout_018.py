from copy import deepcopy
from queue import PriorityQueue

from qiskit.circuit import Qubit, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target

class Layout_018(AnalysisPass):
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

        print(f"In Layout 018")

        layout = Layout()
        pq = PriorityQueue()
        visited_p = dict.fromkeys(self.coupling_map.physical_qubits, False)
        sorted_coupling_qubits = self._sort_coupling_qubits()
        sorted_dag_qubits = self._sort_dag_qubits(dag)
        v_interacts = self._get_dag_qubit_interactivity(dag)

        pq.put((1, sorted_coupling_qubits[0]))
        visited_p[sorted_coupling_qubits[0]] = True

        while not pq.empty():
            curr_p = pq.get()[1]

            if curr_p not in layout.get_physical_bits():

                for best_v in sorted_dag_qubits:
                    if best_v not in layout.get_virtual_bits():
                        layout.add(best_v, curr_p)
                        break
            else:
                # if `curr_p` is mapped to any logical qubit, then map the interactivity of 
                # that logical qubit to neighbor of `curr_p`
                curr_v = layout._p2v[curr_p]
                if len(v_interacts[curr_v]):

                    for p_neighbor in self.coupling_map.neighbors(curr_p):
                        for v_neighbor in v_interacts[curr_v]:
                            if (
                                p_neighbor not in layout.get_physical_bits() and 
                                v_neighbor not in layout.get_virtual_bits()
                            ):
                                layout.add(v_neighbor, p_neighbor)
                                break
            
            for p_neighbor in self.coupling_map.neighbors(curr_p):
                if not visited_p[p_neighbor]:
                    visited_p[p_neighbor] = True
                    priority = sorted_coupling_qubits.index(p_neighbor) + 1
                    pq.put((priority, p_neighbor))

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