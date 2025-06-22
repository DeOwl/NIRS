import time
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
from qiskit_aer import AerSimulator
from collections.abc import Iterable, Sequence

from sklearn.metrics import accuracy_score

algorithm_globals.random_seed = 10


from qiskit_aer.primitives import EstimatorV2

for n_qubits in range (1, 20):
    for number_of_parallel_threads in range(1, 17, 5):
        for optimizer_type in range(1, 2):
            qc = QNNCircuit(num_qubits=n_qubits, feature_map=ZFeatureMap(n_qubits, reps=1), ansatz= EfficientSU2(n_qubits, reps=3))
            print(qc.decompose().draw("text", style="clifford"))
            estimator = EstimatorV2(options={"backend_options":{"max_parallel_threads": number_of_parallel_threads, "statevector_parallel_threshold":1}})


            estimator_qnn = EstimatorQNN(circuit=qc.decompose(), estimator=estimator, weight_params=qc.weight_parameters, input_params=qc.input_parameters)
            def callback_graph(weights, obj_func_eval):
                global start_time
                times.append(time.time() - start_time)                
                if (len(times) == 2):
                    with open("my_file.txt", "a") as file:
                        t = sum(times[1:]) / len(times[1:])
                        time_d =   t * 2 if optimizer_type == 0 else t / len(qc.weight_parameters)
                        print("AAAAAAAAAAAA")
                        file.writelines(str(time_d)+ "\n")
                    raise RuntimeError()
                start_time = time.time()
            
    
            opt = COBYLA() if optimizer_type == 0 else L_BFGS_B()
            estimator_classifier = NeuralNetworkClassifier(
                estimator_qnn, optimizer=opt, callback=callback_graph)

            times = []


            n_samples = 1000  # total number of samples
            n_features = n_qubits   # number of features (parameters)
            n_classes = 2    # number of classes

            # Generate data for each class
            X_list = []
            y_list = []

            for class_label in range(n_classes):
                # For each class, generate features centered at different means
                # For simplicity, centers are spaced apart by a fixed amount
                center = np.full(n_features, fill_value=class_label * 3.0)  # e.g., 0.0 and 3.0 for two classes
                # Generate samples around the center with some noise
                X_class = np.random.normal(loc=center, scale=0.5, size=(n_samples // n_classes, n_features))
                y_class = np.full((n_samples // n_classes,), class_label)
                
                X_list.append(X_class)
                y_list.append(y_class)

            # Combine data from all classes
            X = np.vstack(X_list)
            y = np.hstack(y_list)

            # Shuffle the dataset
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            start_time = time.time()
            with open("my_file.txt", "a") as file:
                    file.writelines(f"optimizer:{'COBYLA' if optimizer_type == 0 else 'L_BGFS'} num_qubits:{n_qubits} num_parallel_threads:{number_of_parallel_threads}" + "\n")
            # fit classifier to data
            try:
                estimator_classifier.fit(X, y)
                with open("my_file.txt", "a") as file:
                        time_d = np.mean(times) * 2 if optimizer_type == 0 else np.mean(times[1:]) / len(estimator_qnn.weight_params)
                        file.writelines(str(len(times)) + " " + str(time_d)+ "\n")
            except RuntimeError:
                continue;