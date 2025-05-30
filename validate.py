import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch import Tensor
import torch
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader, StackDataset
import torch.optim as optim
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
    BCEWithLogitsLoss,
    Sigmoid
)
import torch.nn.functional as F

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from timeit import default_timer as timer
from datetime import timedelta

X = np.load('train_data_reduced.npy')
Y = np.load('train_labels.npy')
from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=1)

x_train_tensor = torch.tensor(X[:200], dtype=torch.float32).cuda()
y_train_tensor = torch.tensor(Y[:200], dtype=torch.float).cuda()

train_dataset = StackDataset(data=x_train_tensor, target=y_train_tensor)


train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

# Number of input features/qubits
num_inputs = 10

feature_map = ZZFeatureMap(num_inputs)
ansatz = RealAmplitudes(num_inputs)
qc = QuantumCircuit(num_inputs)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

from qiskit.primitives import StatevectorEstimator as Estimator

estimator = Estimator()
sigm = Sigmoid()

def create_qnn():
    feature_map = ZZFeatureMap(num_inputs)
    ansatz = RealAmplitudes(num_inputs, reps=1)
    qc = QuantumCircuit(num_inputs)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=estimator,
    )
    return qnn


qnn5 = create_qnn()
model5 = TorchConnector(qnn5)
model5.load_state_dict(torch.load("model4.pt"))

optimizer = optim.LBFGS(model5.parameters(), lr=0.001)
loss_func = BCEWithLogitsLoss()


model5.eval()  # set model to evaluation mode

total_loss = []
with no_grad():

    correct = 0
    for batch_idx, data in enumerate(train_loader):
        output = sigm(model5(data["data"]))
        print(torch.round(output), data["target"])
        
        for i in range(len(output)):
            if(torch.round(output)[i] == data["target"][i]):
                correct += 1
        print(correct)

    print(
        "Performance on test data: Accuracy: {:.1f}%".format(
             correct / 2
        )
    )