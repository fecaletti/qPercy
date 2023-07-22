import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np

class QPercy:
    def __init__(self, qubits, quantum_circuit, initial_weights, threshold, bias, learning_rate, activation_fn='sigmoid', encoding_mode='angle', debug=True):
        self.qubits = qubits
        self.quantum_circuit = quantum_circuit
        self.weights = np.copy(initial_weights)
        self.threshold = threshold
        self.bias = bias
        self.learning_rate = learning_rate
        self.encoding_mode = encoding_mode
        self.debug = debug
        
        if activation_fn == 'sigmoid':
            self.activation_fn = QPercy.__sigmoid_fn

    def __sigmoid_fn(input):
        return 1 / (1 + np.power(np.e, -1 * input))

    def apply_sigmoid(self, data):
        return QPercy.__sigmoid_fn(data)

    def create_circuit(self, weights):
        # qreg = QuantumRegister(len(inputs))
        # creg = ClassicalRegister(1)
        # ckt = QuantumCircuit(qreg, creg)

        # #Encode inputs
        # for i in range(len(inputs)):
        #     if inputs[i] == 1:
        #         ckt.x(qreg[i])

        # ckt.h(qreg[0])
        # ckt.h(qreg[1])

        # ckt.cnot(qreg[0], qreg[1])
        # ckt.rz(self.weights[0], qreg[0])
        # ckt.rz(self.weights[1], qreg[1])

        # ckt.h(qreg[0])
        # ckt.h(qreg[1])

        # ckt.measure(qreg[1], creg[0])

        # return ckt
        n_qubits = int(np.sqrt(len(weights)))
        if self.debug:
            print(f'Obtained => {n_qubits}')
        qreg = QuantumRegister(n_qubits)
        creg = ClassicalRegister(n_qubits)
        ckt = QuantumCircuit(qreg, creg)

        return ckt

        # for q in range(0, n_qubits, 2):
        #     ckt.rx(q, inputs[q])
        #     ckt.ry(q, inputs[q+1])
        
        # return ckt.toGate()

    def add_variational_circuit(self, circuit, weights):
        for q in range(0, circuit.num_qubits):
            circuit.rx(weights[q], q)
            circuit.ry(weights[q+1], q)

    def add_measurements(self, circuit):
        for q in range(0, circuit.num_qubits):
            circuit.measure(q, q)

    def create_angle_encoding_gate(self, inputs):
        n_qubits = int(np.sqrt(len(inputs)))
        # n_qubits = int(len(inputs))
        qreg = QuantumRegister(n_qubits)
        ckt = QuantumCircuit(qreg)

        # normalized_inputs = [ QPercy.__sigmoid_fn(norm) * (np.pi / 2) for norm in inputs ]
        # normalized_inputs = [ QPercy.__sigmoid_fn(norm) * 2 * np.pi for norm in inputs ]
        normalized_inputs = inputs

        if self.debug:
            print(f'Normalized inputs => {normalized_inputs}')

        for q in range(0, n_qubits, 2):
            ckt.rx(normalized_inputs[q], q)
            ckt.ry(normalized_inputs[q + 1], q)
    
        # print(ckt.draw(output='text'))
        return ckt.to_gate(label='angle encoding')

    def run_circuit(self, circuit):
        simulator = AerSimulator()
        compiled_circuit = transpile(circuit, simulator)
        job = simulator.run(compiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts(compiled_circuit)

        return counts, result

    def activation(self, X):
        return 1 / 1 - np.exp(X)

    def process(self, inputs, weights=None):
        _weights = weights if weights != None else self.weights

        ckt = self.create_circuit(inputs)
        if self.debug:
            print('got here')
        encoding_gate = self.create_angle_encoding_gate(inputs)

        ckt.append(encoding_gate, [range(0, ckt.num_qubits)])
        ckt.barrier()
        self.add_variational_circuit(ckt, _weights)
        ckt.barrier()

        self.add_measurements(ckt)
        if self.debug:
            print(ckt.draw())
        # cnts, res = self.run_circuit(ckt)

        return ckt

    def correct_weights(self, corrections, inputs):
        for w in range(len(self.weights)):
            self.weights[w] += self.learning_rate * corrections[w] * inputs[w]
        
        # self.bias += self.learning_rate * corrections[w]


xor_data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

xand_data = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
]

or_data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]

and_data = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
]

# initial_weights = [0.5, -0.5]

# neu = QPercy(None, None, initial_weights=initial_weights, threshold=0.5, bias=0.2, learning_rate=0.15)
# neu.process(data[0][:-1])