import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, ADAM, P_BFGS, SciPyOptimizer
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import BaseEstimatorV1

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_aer import AerSimulator
from collections.abc import Iterable, Sequence

from sklearn.metrics import accuracy_score

algorithm_globals.random_seed = 1

qc = QNNCircuit(num_qubits=10, feature_map=ZZFeatureMap(10), ansatz=RealAmplitudes(10))
qc.decompose().draw("mpl", style="clifford")

from qiskit.primitives import StatevectorSampler as Sampler

estimator = Sampler(default_shots=1024)
def parity(x):
    return "{:b}".format(x).count("1") % 2
estimator_qnn = SamplerQNN(circuit=qc, sampler=estimator, output_shape=2, interpret=parity)

def callback_graph(weights, obj_func_eval):
    print(weights, obj_func_eval)
    plt.clf()
    objective_func_vals.append(obj_func_eval)
    obj_func_range = range(len(objective_func_vals))
    plt.plot(obj_func_range, objective_func_vals)
    plt.draw()
    plt.pause(1)
    
estimator_classifier = NeuralNetworkClassifier(
    estimator_qnn, optimizer=SciPyOptimizer(method="COBYQA"), callback=callback_graph, loss="cross_entropy", one_hot=True)

objective_func_vals = []
R_vals = []
obj_func_range = range(0)
plt.title("Objective function value against iteration")
plt.xlabel("Iteration")
plt.ylabel("Objective function value")
plt.ion()

normal_train_data = np.load('normal_train_data.npy')
normal_test_data = np.load('normal_test_data.npy')
anomalous_train_data = np.load('anomalous_train_data.npy')
anomalous_test_data = np.load('anomalous_test_data.npy')
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')


t_d_1 = np.concatenate((normal_train_data[:500],anomalous_train_data[:500]))
y = np.concatenate((np.ones(500, dtype=int), np.zeros(500, dtype=int)))
from sklearn.decomposition import PCA
pca = PCA(n_components=10)  # Set number of components to 24
train_data_reduced = pca.fit_transform(t_d_1, y)
validation_data_reduced = pca.fit_transform(test_data, test_labels)
# fit classifier to data
estimator_classifier.fit(train_data_reduced, y)

plt.ioff()
plt.show()

# score classifier
print(estimator_classifier.score(validation_data_reduced,test_labels))