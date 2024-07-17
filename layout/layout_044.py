from collections import defaultdict
from copy import deepcopy

from qiskit.circuit import QuantumRegister, Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target

class Layout_044(AnalysisPass):
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

        print(f"In Layout 044")

        sorted_dag_qubits = self._sort_dag_qubits(dag)
        sorted_coupling_qubits = self._sort_coupling_qubits()
        layout = Layout({v: p for v, p in zip(sorted_dag_qubits, sorted_coupling_qubits)})

        for qreg in dag.qregs.values():
            layout.add_register(qreg)
        
        # Map idle physical qubits to ancilla qubits
        idle_p = [p for p in self.coupling_map.physical_qubits if p not in layout.get_physical_bits()]

        if idle_p:
            qreg = QuantumRegister(len(idle_p), name="ancilla")
            layout.add_register(qreg)
            dag.add_qreg(qreg)
        
            for idx, p in enumerate(idle_p):
                layout[p] = qreg[idx]

        qubit_interactivity = self._get_dag_qubit_interactivity(dag)
        topo_sequence = dag.topological_op_nodes()

        for gate in topo_sequence:
            v0, v1 = gate.qargs
            p0, p1 = (layout._v2p[v] for v in gate.qargs)
            path = self.coupling_map.shortest_undirected_path(p0, p1)
            rev_path = list(reversed(path))

            if len(qubit_interactivity[v0]) > len(qubit_interactivity[v1]):
                # Swap `v1` to `v0`
                for i in range(len(path)-2):
                    layout.swap(rev_path[i], rev_path[i+1])
            else:
                # Swap `v0` to `v1`
                for i in range(len(path)-2):
                    layout.swap(path[i], path[i+1])
        
        for gate in reversed(list(topo_sequence)):
            v0, v1 = gate.qargs
            p0, p1 = (layout._v2p[v] for v in gate.qargs)
            path = self.coupling_map.shortest_undirected_path(p0, p1)
            rev_path = list(reversed(path))

            if len(qubit_interactivity[v0]) > len(qubit_interactivity[v1]):
                # Swap `v1` to `v0`
                for i in range(len(path)-2):
                    layout.swap(rev_path[i], rev_path[i+1])
            else:
                # Swap `v0` to `v1`
                for i in range(len(path)-2):
                    layout.swap(path[i], path[i+1])

        self.property_set['layout'] = layout
    
    def _get_dag_qubit_interactivity(self, dag: DAGCircuit):
        """Iterate all logical qubits, and collect whom current qubit have 
        interactivity with in the circuit.

        Args:
            dag (DAGCircuit): DAG to provide gate infos.
        """
        interactivity = defaultdict(list)

        for gate in dag.op_nodes():
            if gate.op.num_qubits == 2:
                v0, v1 = gate.qargs
                interactivity[v0].append(v1)
                interactivity[v1].append(v0)
                
        return interactivity
        
    def _sort_dag_qubits(self, dag: DAGCircuit) -> list[Qubit]:
        """Sort logical qubits in `dag` according to each qubit's interactivity.
        
        Args:
            dag (DAGCircuit): DAG to provide logical qubits and gate infos.
        
        Returns:
            list[Qubit]: The sorted list of logical qubits in `dag`. The first element
            is one with the most interactivity, while the last element is one with the
            least interactivity.
        """
        interact = dict.fromkeys(dag.qubits, 0)

        for gate in dag.op_nodes():
            if gate.op.num_qubits == 2:
                v0, v1 = gate.qargs
                interact[v0] += 1
                interact[v1] += 1

        return (v for v, _ in sorted(interact.items(), key=lambda x: x[1], reverse=True))
    
    def _sort_coupling_qubits(self) -> list[int]:
        """Sort physical qubits of coupling map according to each qubit's degree.
        The first element is a physical qubit with the most degree.
        """
        degree = dict.fromkeys(self.coupling_map.physical_qubits, 0)

        for p in self.coupling_map.physical_qubits:
            degree[p] = len(self.coupling_map.neighbors(p))

        return (p for p, _ in sorted(degree.items(), key=lambda x: x[1], reverse=True))