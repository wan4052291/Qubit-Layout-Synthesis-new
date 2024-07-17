from collections import defaultdict
from copy import deepcopy
from queue import Queue

from qiskit.circuit import QuantumRegister, Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target


class Layout_010(AnalysisPass):
    
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

        print(f"In Layout 010")

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
        num_unmapped_dag_qubit = len(dag.qubits)
        used_dag_qubit = dict.fromkeys(dag.qubits, False)
        locked_coupling_qubit = dict.fromkeys(self.coupling_map.physical_qubits, False)

        for gate in dag.topological_op_nodes():
            v0, v1 = gate.qargs
            p0, p1 = (layout._v2p[v] for v in gate.qargs)
            
            if used_dag_qubit[v0] == False or used_dag_qubit[v1] == False:
                used_dag_qubit[v0] = True
                used_dag_qubit[v1] = True

                if locked_coupling_qubit[p1] == False:
                    # src, dest = p0, p1
                    if locked_coupling_qubit[p0] == False:
                        num_unmapped_dag_qubit -= 2
                    else:
                        num_unmapped_dag_qubit -= 1
                    new_location = self._lock_bfs(p0, locked_coupling_qubit)
                    layout.swap(p1, new_location)
                else:
                    # src, dest = p1, p0
                    num_unmapped_dag_qubit -= 1
                    new_location = self._lock_bfs(p1, locked_coupling_qubit)
                    layout.swap(p0, new_location)
                        
            # new_physical_qubit = self._lock_bfs(src, locked_coupling_qubit)
            # layout.swap(dest, new_physical_qubit)
            
            if num_unmapped_dag_qubit == 0:
                break

        self.property_set['layout'] = layout
    
    def _lock_bfs(self, src: int, locked_coupling_qubit: dict[Qubit, bool]):
        que = Queue()
        visited = [False] * self.coupling_map.size()
        que.put(src)
        visited[src] = True

        while not que.empty():
            curr = que.get()
            if locked_coupling_qubit[curr] == False:
                locked_coupling_qubit[curr] = True
                return curr
            else:
                for neighbor in self.coupling_map.neighbors(curr):
                    if visited[neighbor] == False:
                        que.put(neighbor)
                        visited[neighbor] = True
        return None