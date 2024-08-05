import pennylane as qml
from pennylane import numpy as np
import torch

class HybridModel(torch.nn.Module):
    def __init__(self, n_qubits=8, n_layers=10):
        super(HybridModel, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(self.quantum_circuit, weight_shapes)
        self.fc1 = torch.nn.Linear(n_qubits, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.qlayer(x)
        x = self.fc1(x)
        return x

    @qml.qnode(self.dev, interface='torch')
    def quantum_circuit(self, inputs, weights):
        qml.templates.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
        for w in weights:
            self.layer(w)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def layer(self, weights):
        for i in range(self.n_qubits):
            qml.Rot(*weights[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
