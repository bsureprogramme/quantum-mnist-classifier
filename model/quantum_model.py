import pennylane as qml
import torch
import torch.nn as nn
import itertools

class QModel(nn.Module):
    def __init__(self):
        super(QModel, self).__init__()
        self.n_qubits = 9
        self.n_layers = 2
        self.device = qml.device('default.qubit', wires=self.n_qubits)
        self.weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        self.init_method = {"weights": lambda x: torch.nn.init.uniform_(x, -1 * np.pi, np.pi)}
        self.circuitQNode = qml.QNode(self.circuit, self.device, interface='torch', diff_method='backprop')
        self.batchQNode = qml.batch_input(self.circuitQNode, argnum=0)
        self.qLayer = qml.qnn.TorchLayer(self.batchQNode, self.weight_shapes, init_method=self.init_method)
        self.flatten = nn.Flatten()
        self.finalFlatten = nn.Flatten(start_dim=0)

    def entangling_layer(self):
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_qubits - 1, 0])

    def circuit(self, inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                qml.RZ(weights[layer, wire, 0], wires=wire)
                qml.RX(weights[layer, wire, 1], wires=wire)
                qml.RZ(weights[layer, wire, 2], wires=wire)
            self.entangling_layer()
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        x = self.flatten(x)
        x = self.qLayer(x)
        x = self.finalFlatten(x)
        return x
