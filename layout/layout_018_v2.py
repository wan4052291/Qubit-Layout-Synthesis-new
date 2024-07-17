from collections import defaultdict
from copy import deepcopy
from queue import PriorityQueue

from qiskit.circuit.quantumregister import Qubit
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout

class Layout_018_V2(AnalysisPass):
    def __init__(self, coupling_map: CouplingMap):
        super().__init__()
        if not coupling_map:
            raise TranspilerError("A coupling map is necessary to run the pass.")

        self.coupling_map = coupling_map
        if self.coupling_map is not None:
            if not self.coupling_map.is_symmetric:
                # deepcopy is needed here if we don't own the coupling map (i.e. we were passed it
                # directly) to avoid modifications updating shared references in passes which
                # require directional constraints
                if isinstance(coupling_map, CouplingMap):
                    self.coupling_map = deepcopy(self.coupling_map)
                self.coupling_map.make_symmetric()
        self.nodes = coupling_map.physical_qubits
        self.num_nodes = coupling_map.size()
        # print(f"coupling map: {coupling_map}")

    def run(self, dag: DAGCircuit):
        if len(dag.qubits) > self.num_nodes:
            raise TranspilerError("Number of qubits greater than device.")

        layout = Layout(self._inner_run(dag))
        for qreg in dag.qregs.values():
            layout.add_register(qreg)
        self.property_set['layout'] = layout
        
    def _inner_run(self, dag: DAGCircuit) -> dict[Qubit, int | None]:
        node_ranking = self._sort_node_by_degree()
        qubit_ranking = self._sort_qubit_by_weight(dag)
        # max_degree_node = sorted_node_degree[0]
        qubit_neighbors = self._collect_qubit_neighbors(dag)
        print(f"node ranking: {node_ranking}")
        print(f"qubit ranking: {qubit_ranking}")
        return self._bfs_from_max_degree_node(
            dag,
            node_ranking,
            qubit_ranking,
            qubit_neighbors
        )

    def _bfs_from_max_degree_node(
            self,
            dag: DAGCircuit,
            node_ranking: list[int],
            qubit_ranking: list[Qubit],
            qubit_neighbors: defaultdict[list[Qubit]]
    ) -> dict[Qubit, int | None]:

        # BFS search from source (the node with max degree)
        print("BFS search from max-degree node")
        mapping = dict.fromkeys(dag.qubits, None)
        print(f"mapping: {mapping}")
        index = 0
        pq = PriorityQueue()
        pq.put((1, node_ranking[0]))
        visited = [False] * self.num_nodes
        visited[node_ranking[0]] = True

        while not pq.empty():
            print(f"visited: {visited}")
            print(f"priority queue: {pq.queue}")
            # if current node is not mapped to any qubit
            curr_node = pq.get()[1]
            print(f"current node is {curr_node}")
            if curr_node not in mapping.values():
                print("current node is unmapped, find unmapped high-degree qubit...")
                # find unmapped high-degree qubit
                is_all_qubit_mapped = False
                while mapping[qubit_ranking[index]]:
                    print(f"check index = {index}: {qubit_ranking[index]}")
                    index += 1
                    if index >= len(qubit_ranking):
                        print("All qubits are mapped")
                        is_all_qubit_mapped = True
                        break

                # if all qubit are mapped, then break
                # else map current node to unmapped high-degree qubit
                if is_all_qubit_mapped:
                    break
                else:
                    mapping[qubit_ranking[index]] = curr_node
                    print(f"MAP {qubit_ranking[index]} to {curr_node}")

            else:
                # if current node is mapped to one qubit,
                # and the qubit is envolved in some gate
                curr_qubit = list(mapping.keys())[list(mapping.values()).index(curr_node)]
                print(f"current node ({curr_node}) is already mapped to {curr_qubit}")
                if len(qubit_neighbors[curr_qubit]):
                    print(f"iterate neighbors of its corresponding qubit...")
                    ptr = 0
                    for node_neighbor in self.coupling_map.neighbors(curr_node):
                        # if the neighbor of current node is mapped
                        if node_neighbor in mapping.values():
                            continue

                        # find unmapped qubit in current qubit's neighbors
                        print(f"neighbor of qubit: {qubit_neighbors[curr_qubit]}")
                        is_qubit_neighbor_all_mapped = False
                        while mapping[qubit_neighbors[curr_qubit][ptr]]:
                            print(f"check ptr = {ptr}: {mapping[qubit_neighbors[curr_qubit][ptr]]}")
                            ptr += 1
                            if ptr >= len(qubit_neighbors[curr_qubit]):
                                print("All neighbors of current qubit are mapped")
                                is_qubit_neighbor_all_mapped = True
                                break

                        # if current qubit has no unmapped neighbor, then break
                        # else map the node neighbor to the qubit neighbor
                        if is_qubit_neighbor_all_mapped:
                            break
                        else:
                            mapping[qubit_neighbors[curr_qubit][ptr]] = node_neighbor
                            print(f"MAP {qubit_neighbors[curr_qubit][ptr]} to {node_neighbor}")
            
            for node_neighbor in self.coupling_map.neighbors(curr_node):
                if not visited[node_neighbor]:
                    visited[node_neighbor] = True
                    priority = node_ranking.index(node_neighbor) + 1
                    pq.put((priority, node_neighbor))
            
            print(f"mapping: {mapping}")

        return mapping

    def _collect_qubit_neighbors(self, dag: DAGCircuit) -> defaultdict[list]:
        qubit_neighbors = defaultdict(list)
        circ = dag_to_circuit(dag)
        for inst in circ.data:
            if inst.operation.num_qubits == 2:
                # print(f"{inst.operation.name} ({inst.qubits[0]}, {inst.qubits[1]})")
                q1 = inst.qubits[0]
                q2 = inst.qubits[1]
                qubit_neighbors[q1].append(q2)
                qubit_neighbors[q2].append(q1)
        return qubit_neighbors

    def _sort_qubit_by_weight(self, dag: DAGCircuit) -> list[Qubit]:
        circ = dag_to_circuit(dag)
        # involved_gates = dict.fromkeys(circ.qubits, list())
        involved_gates = defaultdict(list)
        num_gate = len(circ.data)
        weights = defaultdict(int)

        for index, inst in enumerate(circ.data):
            if inst.operation.num_qubits == 2:
                involved_gates[inst.qubits[0]].append(index+1)
                involved_gates[inst.qubits[1]].append(index+1)
        # print(f"involved gates: {involved_gates}")

        for qubit in circ.qubits:
            weight = 0
            for gate_id in involved_gates[qubit]:
                # print(f"involved_gates of {qubit}: {gate_id}")
                weight += (num_gate - gate_id + 1) / num_gate
            weights[qubit] = weight
        # for qubit, weight in weights.items():
        #     print(f"weight of {qubit}: {weight}")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        # print("sorted qubit weight:")
        # print(sorted_weights)
        return [qubit for qubit, weight in sorted_weights]

    def _sort_node_by_degree(self) -> list[int]:
        degrees = {}
        for id in self.nodes:
            degrees[id] = len(self.coupling_map.neighbors(id))
        # for id, degree in deg.items():
        #     print(f"degree of n_{id}: {degree}")
        sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return [node for node, degree in sorted_degrees]