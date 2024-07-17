from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,qasm2
import os


qasm_path = ["benchmark\\mqtbench\\qubit_16_20",
             "benchmark\\mqtbench\\qubit_50_53"]



# qasm_path = ["benchmark\\mqtbench\\qubit_16_201",
#              "benchmark\\mqtbench\\qubit_50_531"]


output_path = ["benchmark\\mqtbench\\qubit_16_20_ONLYQ",
             "benchmark\\mqtbench\\qubit_50_53_ONLYQ"]
index = 0
circuits = []

for path in output_path : 
    if not os.path.isdir(path):
        os.makedirs(path)

def keep_cx(circuit) -> QuantumCircuit:
    
    qregs = {qreg.name: qreg.size for qreg in circuit.qregs}
    cregs = {creg.name: creg.size for creg in circuit.cregs}
    new_qregs = {name: QuantumRegister(size, name) for name, size in qregs.items()}
    new_cregs = {name: ClassicalRegister(size, name) for name, size in cregs.items()}
    new_circuit = QuantumCircuit(*new_qregs.values(), *new_cregs.values())  
    '''
    for instr, qargs, cargs in circuit.data:
        if instr.num_qubits == 2:
            new_circuit.append(instr,qargs,cargs)        
    '''

    for instr, qargs, cargs in circuit.data:
        if instr.name == 'cx':
            new_circuit.append(instr,qargs,cargs)        
            
    return new_circuit

def alter(qc):
    total_qubits = sum(reg.size for reg in qc.qregs)       
    new_qreg = QuantumRegister(total_qubits, 'q')
    new_circuit = QuantumCircuit(new_qreg)
    
    for creg in qc.cregs:
        new_circuit.add_register(creg)
    qubit_map = {}
    current_qubit_index = 0
    for qreg in qc.qregs:
        for qubit in qreg:
            qubit_map[qubit] = new_qreg[current_qubit_index]
            current_qubit_index += 1

    for param in qc.parameters:
        if param not in new_circuit.parameters:
            new_circuit._parameter_table[param] = qc._parameter_table[param]

    # 將原始電路中的指令複製到新的電路
    for instr, qargs, cargs in qc:
        new_qargs = [qubit_map[qubit] for qubit in qargs]
        new_circuit.append(instr, new_qargs, cargs)
    
    return new_circuit


for path in qasm_path:
    for filename in os.listdir(path):
        if(filename.endswith(".qasm")):
            filepath = os.path.join(path,filename)
            qc = QuantumCircuit.from_qasm_file(filepath)
            print(filename)
            new_qc = alter(qc)
            new_file_path = os.path.join(output_path[index],filename)
            print(len(new_qc))
            if len(new_qc.data) != 0:
                qasm2.dump(new_qc,new_file_path)
            
    index += 1



