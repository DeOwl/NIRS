import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit

algorithm_globals.random_seed = 42
num_inputs = 10
objective_func_vals = []

def callback_graph(weights, obj_func_eval):
    print(weights, obj_func_eval)



X = np.load('train_data_reduced.npy')
Y = np.load('train_labels.npy')
np.random.shuffle(X)
np.random.shuffle(Y)
from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=1)
X = X[:800]
Y = Y[:800]
feature_map = ZZFeatureMap(num_inputs)
ansatz = RealAmplitudes(num_inputs)

vqr = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=100),
    callback=callback_graph,
)


# fit regressor
vqr.fit(X, Y)

# score result
print(vqr.score(X, Y))