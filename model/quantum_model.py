import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from pennylane import broadcast
from pennylane.wires import Wires

class QModel(nn.Module):
    def __init__(self):
        super(QModel, self).__init__()
        self.n_qubits = 9  # 8 for processing, 1 for output
        self.n_layers = 2
        self.device = qml.device('default.qubit', wires=self.n_qubits)
        self.weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        self.init_method = {"weights": lambda x: torch.nn.init.uniform_(x, -1 * np.pi, np.pi)}
        self.circuitQNode = qml.QNode(self.circuit, self.device, interface='torch', diff_method='backprop')
        self.batchQNode = qml.batch_input(self.circuitQNode, argnum=0)
        self.qLayer = qml.qnn.TorchLayer(self.circuitQNode, self.weight_shapes, init_method=self.init_method)
        self.flatten = nn.Flatten()
        self.finalFlatten = nn.Flatten(start_dim=0)
    
    def __entangling_layer(self, entangler: str, pattern: str, wires:int):
        match entangler:
            case 'CNOT':
                entangling_operation = qml.CNOT
            case 'CZ':
                entangling_operation = qml.CZ
        broadcast(unitary= entangling_operation, pattern = pattern, wires = wires)
    
    def wires_pairwise(self, wires):
        """Wire sequence for the pairwise pattern."""
        sequence = []
        for layer in range(2):
            block = wires[layer : len(wires)]
            sequence += [block.subset([i, i + 1]) for i in range(0, len(block) - 1, 2)]
        return sequence

    def circuit(self, inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits - 1), normalize=True)
        qml.PauliX(self.n_qubits-1)
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                qml.RZ(weights[layer, wire, 0], wires=wire)
                qml.RX(weights[layer, wire, 1], wires=wire)
                qml.RZ(weights[layer, wire, 2], wires=wire)
            #.entangling(.ent_list)
            self.__entangling_layer(entangler = 'CNOT', pattern = self.wires_pairwise(Wires([i for i in range(self.n_qubits)])), wires = [i for i in range(self.n_qubits)])
        return qml.probs(wires=self.n_qubits - 1)  # Measure probability distribution


    def forward(self, x):
        x = self.flatten(x)
        x = self.qLayer(x)
        return x  # Return the probabilities directly
