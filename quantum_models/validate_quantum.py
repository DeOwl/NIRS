import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2, ZFeatureMap
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, ADAM, P_BFGS, SciPyOptimizer
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import BaseEstimatorV1

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from collections.abc import Iterable, Sequence

from sklearn.metrics import accuracy_score



algorithm_globals.random_seed = 1

qc = QNNCircuit(num_qubits=7, feature_map=ZFeatureMap(7, reps=1), ansatz=EfficientSU2(7, reps=2))
qc.decompose().draw("mpl", style="clifford")

from qiskit.primitives import StatevectorSampler as Sampler

normal_train_data = np.load('../data/normal_train_data.npy')
normal_test_data = np.load('../data/normal_test_data.npy')
anomalous_train_data = np.load('../data/anomalous_train_data.npy')
anomalous_test_data = np.load('../data/anomalous_test_data.npy')
train_data = np.load('../data/train_data.npy')
test_data = np.load('../data/test_data.npy')
train_labels = np.load('../data/train_labels.npy')
test_labels = np.load('../data/test_labels.npy')


t_d_1 = np.concatenate((normal_train_data[:500],anomalous_train_data[:500]))
y = np.concatenate((np.ones(500, dtype=int), np.zeros(500, dtype=int)))
from sklearn.decomposition import PCA
pca = PCA(n_components=7, random_state=1)  # Set number of components to 24
train_data_reduced = pca.fit_transform(test_data, y)
from qiskit.primitives import StatevectorEstimator as Estimator

estimator = Estimator()

estimator_qnn = EstimatorQNN(circuit=qc, estimator=estimator)

data6 = [
    0.95761627, 0.14935029, 1.55785771, 0.2675841, 0.0700845, -0.01276889,
    0.21977913, 0.80257391, 0.50774254, 0.6491254, 0.72011062, 0.0952225,
    0.40746078, 0.35398017, 0.132843, 1.6068075, -0.28641487, 0.97855575,
    -0.27334889, 1.43915594, 0.93280761, 0.77060616, 0.04392645, 0.95995703,
    0.55044612, -0.1466742, 0.45362373, 0.34166176, 1.55635563, 1.33242941,
    0.27241527, -0.01167743, 0.76397984, 0.53776008, -0.60833722, 0.1494149,
    0.13678915, 0.24892094, 0.38282467, 0.64907906, 0.83756376, 0.77603195
]

data = estimator_qnn.forward(train_data_reduced, data6)
print(data)
print(accuracy_score([1 if data[i] < 0 else 0 for i in range(len(data))], test_labels))