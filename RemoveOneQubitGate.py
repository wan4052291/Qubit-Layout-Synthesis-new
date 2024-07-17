from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,qasm2
import os


qasm_path = ["benchmark\\mqtbench\\qubit_16_20_ONLYQ",
             "benchmark\\mqtbench\\qubit_16_20_ONLYQ"]


'''
qasm_path = ["benchmark\\mqtbench\\qubit_16_201",
             "benchmark\\mqtbench\\qubit_50_531"]
'''

output_path = ["benchmark\\mqtbench\\qubit_16_20_ONLYQ_CX",
             "benchmark\\mqtbench\\qubit_50_53_ONLYQ_CX"]
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



for path in qasm_path:
    for filename in os.listdir(path):
        if(filename.endswith(".qasm")):
            filepath = os.path.join(path,filename)
            qc = QuantumCircuit.from_qasm_file(filepath)
            new_qc = keep_cx(qc)
            new_file_path = os.path.join(output_path[index],filename)
            print(new_file_path)
            if len(new_qc.data) != 0:
                qasm2.dump(new_qc,new_file_path)
    index += 1



