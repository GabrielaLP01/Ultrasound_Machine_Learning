"""
Código desenvolvido por:

Prof. Dr. Sidney Lima
Universidade Federal de Pernambuco
Departamento de Eletrônica e Sistemas
"""
from libsvm.svmutil import *
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.datasets import dump_svmlight_file
import os
import numpy as np
import pandas as pd
import argparse
import sys, string
import os

#========================================================================
class svmParameters():
    def main(self, dataset):
        # Carregar e Preparar o Conjunto de Dados
        threshold = 0.1
        y, x = svm_read_problem(dataset)
        y, x = pruningDataset(y, x, threshold)

        cost_vector = []
        cost_vector.append(10 ** -3)
        cost_vector.append(10 ** -2)
        cost_vector.append(10 ** -1)
        cost_vector.append(10 ** 0)
        cost_vector.append(10 ** 1)
        cost_vector.append(10 ** 2)
        cost_vector.append(10 ** 3)    
        gamma_vector = cost_vector

        # Criar arquivo para salvar os parâmetros
        with open("svm_parameters_results.txt", "w") as params_file:
            params_file.write("Kernel\tCost\tGamma\tMean_Acc_Train\tStd_Acc_Train\tMean_Acc_Test\tStd_Acc_Test\n")
            
            # Criar arquivo para salvar as matrizes de confusão
            with open("svm_confusion_matrices.txt", "w") as confusion_file:
                min_acc = 101
                max_acc = -1
                min_kernel = -1
                max_kernel = -1
                min_cost = 0
                max_cost = 0
                min_gamma = 0
                max_gamma = 0
                
                min_mean_accuracy_train = 101
                min_std_accuracy_train = 101
                min_mean_accuracy_test = 101
                min_std_accuracy_test = 101
                
                max_mean_accuracy_train = -1 
                max_std_accuracy_train = -1 
                max_mean_accuracy_test = -1
                max_std_accuracy_test = -1

                for t in range(4):
                    for c in range(len(cost_vector)):
                        for g in range(len(gamma_vector)):
                            mean_accuracy_train, std_accuracy_train, mean_accuracy_test, std_accuracy_test, cm_train, cm_test = \
                            svmKfold(y, x, t, cost_vector[c], gamma_vector[g])
                            
                            # Escrever os parâmetros no arquivo
                            params_file.write(f"{kernel_str(t)}\t{cost_vector[c]:.6f}\t{gamma_vector[g]:.6f}\t")
                            params_file.write(f"{mean_accuracy_train:.2f}\t{std_accuracy_train:.2f}\t")
                            params_file.write(f"{mean_accuracy_test:.2f}\t{std_accuracy_test:.2f}\n")
                            params_file.flush()  # Forçar escrita imediata
                            
                            # Escrever as matrizes de confusão no arquivo
                            confusion_file.write(f"\nKernel: {kernel_str(t)}, Cost: {cost_vector[c]:.6f}, Gamma: {gamma_vector[g]:.6f}\n")
                            confusion_file.write("Matriz de Confusão (Treino - %):\n")
                            np.savetxt(confusion_file, cm_train, fmt='%.2f', delimiter='\t')
                            confusion_file.write("\nMatriz de Confusão (Teste - %):\n")
                            np.savetxt(confusion_file, cm_test, fmt='%.2f', delimiter='\t')
                            confusion_file.write("\n" + "="*50 + "\n")
                            confusion_file.flush()  # Forçar escrita imediata

                            if(mean_accuracy_test < min_acc):
                                min_acc = mean_accuracy_test
                                min_kernel = t
                                min_cost = cost_vector[c]
                                min_gamma = gamma_vector[g]
                                min_mean_accuracy_train = mean_accuracy_train
                                min_std_accuracy_train = std_accuracy_train
                                min_mean_accuracy_test = mean_accuracy_test
                                min_std_accuracy_test = std_accuracy_test
                            if(mean_accuracy_test > max_acc):
                                max_acc = mean_accuracy_test
                                max_kernel = t
                                max_cost = cost_vector[c]
                                max_gamma = gamma_vector[g]
                                max_mean_accuracy_train = mean_accuracy_train 
                                max_std_accuracy_train = std_accuracy_train 
                                max_mean_accuracy_test = mean_accuracy_test
                                max_std_accuracy_test = std_accuracy_test

        print(f'...........................................')
        print(f"Pior Acurácia Média de Treino: {min_mean_accuracy_train:.2f}% ± {min_std_accuracy_train:.2f}%")
        print(f"Pior Acurácia Média de Teste: {min_mean_accuracy_test:.2f}% ± {min_std_accuracy_test:.2f}%")
        print('Pior kernel: ' + kernel_str(min_kernel))
        print(f'Pior conf. de cost: {min_cost:.3f}')
        print(f'Pior conf. de gamma: {min_gamma:.3f}')
        print(f'...........................................')
        print(f"Melhor Acurácia Média de Treino: {max_mean_accuracy_train:.2f}% ± {max_std_accuracy_train:.2f}%")
        print(f"Melhor Acurácia Média de Teste: {max_mean_accuracy_test:.2f}% ± {max_std_accuracy_test:.2f}%")
        print('Melhor Kernel: ' + kernel_str(max_kernel))
        print(f'Melhor conf. de Cost: {max_cost:.3f}')
        print(f'Melhor conf. de Gamma: {max_gamma:.3f}')

#========================================================================
def pruningDataset(y, x, threshold):

    # Convert LIBSVM format to dense numpy array
    max_feature = max([max(sample.keys()) if sample else 0 for sample in x])
    x_array = np.zeros((len(x), max_feature))
    
    for i, sample in enumerate(x):
        for feature, value in sample.items():
            x_array[i, feature-1] = value  # Convert 1-based to 0-based indexing

    # Calculate feature correlations with target
    correlations = np.array([np.corrcoef(x_array[:, i], y)[0, 1] 
                           for i in range(x_array.shape[1])])
    
    # Select features above correlation threshold
    selected_indices = np.where(np.abs(correlations) >= threshold)[0]
    
    # Save selected features with their correlations
    with open('selected_features.csv', 'w') as ff:
        ff.write("Original_Index,Selected_Index,Correlation\n")
        for new_idx, old_idx in enumerate(selected_indices, 1):
            ff.write(f"{old_idx+1},{new_idx},{correlations[old_idx]:.6f}\n")

    # Filter and reindex features
    x_pruned = []
    feature_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices, 1)}
    
    for sample in x:
        # Keep only selected features and reindex sequentially
        pruned_sample = {feature_mapping[k-1]: v 
                         for k, v in sample.items() 
                         if (k-1) in selected_indices}
        x_pruned.append(pruned_sample)
    
    # Save to disk if output path provided
    save_libsvm(y, x_pruned, 'globalPruned.csv')
    
    return (y, x_pruned)

#========================================================================
def save_libsvm(y, x, filename):
    """Save data in LIBSVM format"""
    with open(filename, 'w') as f:
        for label, features in zip(y, x):
            sorted_features = sorted(features.items(), key=lambda x: x[0])
            feature_str = ' '.join(f"{k}:{v}" for k, v in sorted_features)
            f.write(f"{int(label)} {feature_str}\n")
#========================================================================    
def kernel_str(t):
    if (t == 0): 
        str_kernel = 'Linear' 
    elif (t == 1): 
        str_kernel = 'Polynomial'
    elif (t == 2): 
        str_kernel = 'Radial Basis Function'
    elif (t == 3):  
        str_kernel = 'Sigmoid'
    return str_kernel

#========================================================================
def svmKfold(y, x, t, cost, gamma):        
    # Configurar o k-Fold
    k = 10
    # O parâmetro shuffle server para randomizar (embaralhar) as amostras
    np.random.seed(1)
    kf = KFold(n_splits=k, shuffle=True)
    
    accuracies_train = []
    accuracies_test = []
    
    all_y_train_true = []
    all_y_train_pred = []
    all_y_test_true = []
    all_y_test_pred = []
    
    for train_index, test_index in kf.split(x):
        x_train, x_test = [x[i] for i in train_index], [x[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        m = svm_train(y_train, x_train, '-t ' + str(t) + ' -c ' + str(cost) + ' -g ' + str(gamma))

        p_label_train, p_acc_train, p_val_train = svm_predict(y_train, x_train, m)
        p_label_test, p_acc_test, p_val_test = svm_predict(y_test, x_test, m)   
        
        accuracies_train.append(p_acc_train[0])
        accuracies_test.append(p_acc_test[0])
        
        all_y_train_true.extend(y_train)
        all_y_train_pred.extend(p_label_train)
        all_y_test_true.extend(y_test)
        all_y_test_pred.extend(p_label_test)
    
    mean_accuracy_train = np.mean(accuracies_train)
    std_accuracy_train = np.std(accuracies_train)
    mean_accuracy_test = np.mean(accuracies_test)
    std_accuracy_test = np.std(accuracies_test)
    
    # Calcular matrizes de confusão
    cm_train = confusion_matrix(all_y_train_true, all_y_train_pred)
    cm_train_percent = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis] * 100
    
    cm_test = confusion_matrix(all_y_test_true, all_y_test_pred)
    cm_test_percent = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis] * 100
    
    return mean_accuracy_train, std_accuracy_train, mean_accuracy_test, std_accuracy_test, cm_train_percent, cm_test_percent

#========================================================================
def setOpts(argv):                         
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', dest='dataset', action='store', 
        default='heart_scale', help='Filename of dataset')
        
    arg = parser.parse_args()
    return(arg.__dict__['dataset'])    

#========================================================================
if __name__ == "__main__":
    opts = setOpts(sys.argv[1:])
    ff = svmParameters()
    ff.main(opts)