import networkx as nx
from FiDLS.vfs import Vf
from qiskit import qasm2

def CreateCircuitFromQASM(file, path):
    QASM_file = open(path + file, 'r')
    iter_f = iter(QASM_file)
    QASM = ''
    for line in iter_f: 
        QASM = QASM + line
    #print(QASM)
    # cir = QuantumCircuit.from_qasm_str(QASM)
    cir = qasm2.loads(QASM)
    QASM_file.close    
    return cir

def ReducedCircuit(cir):
    '''Return Reduced Circuit containing only [name, [p,q]], e.g., ['cx', [0,2]] '''
    C = []
    for gate in cir:
        if gate[0].name != 'cx': continue
        qubits = [q.index for q in gate[1]]
        C.append(qubits)
    return C

def qubit_in_circuit(LD, C): 
    ''' Return the set of qubits in a subcircuit D of C
    Args:
        LD (list): a sublist of CNOT gates of the input circuit C
    Returns:
        QD (set): the set of qubits in D
    '''
    QD = set()
    for i in LD:
        QD.add(C[i][0])
        QD.add(C[i][1])
    return QD

#####################################################################################
#        #GRAPHS related to circuit and NISQ device architecture#
#####################################################################################
    
def graph_of_circuit(C):
    ''' Return the induced graph of the reduced circuit C
            - node set: qubits in C
            - edge set: all pair (p,q) if CNOT [p,q] or CNOT[q,p] in C
        Args:
            C (list): the input reduced circuit
        Returns:
            g (Graph)
    '''   
    L = list(range(len(C)))
    g = nx.Graph()
    g.add_nodes_from(qubit_in_circuit(L, C))
    for gate in C:
        if len(gate) != 2: continue
        g.add_edge(gate[0],gate[1])
    return g

spl = nx.shortest_path_length

def hub(g):
    '''A hub of g is a node with maximum degree'''
    if not nx.is_connected(g): 
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy() 
        
    deg = max([g.degree(node) for node in g.nodes ])
    Hub = []
    for node in g.nodes():
        if g.degree(node) == deg: Hub.append(node)

    step = 0
    while len(Hub) > 1 and step < nx.diameter(g):
        step += 1
        Hub = [[x, len([y for y in g.nodes() if spl(g,x,y)== step+1])] for x in Hub]
        max_val = max([item[1] for item in Hub])
        Hub = [item[0] for item in Hub if item[1]==max_val]
    return Hub[0]

def is_embeddable(g, H, anchor, stop):
    '''check if a small graph g is embeddable in a large H, anchor is bool
        g, H (Graph)
        anchor (bool): whether or not mapping anchor of g to that of H
        stop (float): time limit for vf2
    '''
    vf2 = Vf()
    result = {} 
    if anchor: result[hub(g)] = hub(H)
    result = vf2.dfsMatch(g, H, result, stop)
    lng = len(nx.nodes(g))
    if len(result) == lng:
        return True, result   
    return False, result