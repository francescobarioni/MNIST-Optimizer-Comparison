import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import torch.nn.functional as F
import os
import pandas as pd
import csv
import ast
import random

# modulo DATA.PY
def calculate_statistics(dataset, batch_size=64):
    """
    Calcola la media, deviazione standard, valore minimo e massimo del dataset MNIST.
    Restituisce anche un istogramma della distribuzione dei valori dei pixel.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # inizializzazione contatori
    mean = 0.0
    std = 0.0
    min_val = float('inf')
    max_val = float('-inf')
    pixel_values = []  # si traccia la distribuzione dei pixel

    num_batches = 0

    for images, _ in loader:
        # calcolo la media e la deviazione standard
        mean += images.mean([0, 2, 3])  # media sui canali e su tutte le immagini del batch
        std += images.std([0, 2, 3])    # deviazione standard sui canali e su tutte le immagini del batch

        # minimo e massimo valore dei pixel
        min_val = min(min_val, images.min())
        max_val = max(max_val, images.max())

        # aggiungo i valori dei pixel per l'istogramma
        pixel_values.append(images.view(-1).numpy())  # Rende 1D ogni batch e aggiunge alla lista
        num_batches += 1

    # calcolo la media e la deviazione standard complessiva
    mean /= num_batches
    std /= num_batches

    # unisco tutti i batch per ottenere un array 1D di tutti i valori dei pixel
    pixel_values = np.concatenate(pixel_values)

    # traccio l'istogramma della distribuzione dei pixel
    plt.figure(figsize=(10, 6))
    plt.hist(pixel_values, bins=50, range=(0, 1), color='blue', alpha=0.7)
    plt.title('Distribuzione dei valori dei pixel')
    plt.xlabel('Valore del pixel')
    plt.ylabel('Frequenza')
    plt.show()

    return mean, std, min_val, max_val

# modulo MODEL.PY
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10):
        super(MLP, self).__init__()

        # primo strato nascosto
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])  # input -> primo strato
        self.relu = nn.ReLU()  # funzione di attivazione ReLU

        # secondo strato nascosto
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])  # primo -> secondo strato
        self.relu2 = nn.ReLU()  # Funzione di attivazione ReLU

        # strato di output
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)  # secondo -> output layer
        self.softmax = nn.Softmax(dim=1)  # funzione Softmax per probabilità

    def forward(self, x):
        # passaggio attraverso il primo strato e ReLU
        x = self.fc1(x)
        x = self.relu(x)

        # passaggio attraverso il secondo strato e ReLU
        x = self.fc2(x)
        x = self.relu2(x)

        # passaggio attraverso l'output layer e Softmax
        x = self.fc3(x)
        return x

# modulo OPTIMIZERS.PY
# ottimizzatore Adam
class AdamOptimizer:
    def __init__(self, params, lr, betas, eps):
        self.params = list(params) # inizializza i parametri  come lista
        self.lr = lr # imposta il learning rate
        self.betas = betas  # imposta beta per il calcolo dei momenti
        self.eps = eps # imposto precisione di macchina
        self.m = [torch.zeros_like(p) for p in self.params]  # primo momento
        self.v = [torch.zeros_like(p) for p in self.params]  # secondo momento
        self.t = 0  # passo temporale

    def step(self):
        self.t += 1 # incrementa il passo temporale
        with torch.no_grad():  # disabilita il calcolo del gradiente per l'aggiornamento
            for p, m_, v_ in zip(self.params, self.m, self.v):
                if p.grad is None:
                    continue
                # aggiorna il primo momento con la media esponenziale dei gradienti
                m_.mul_(self.betas[0]).add_((1 - self.betas[0]) * p.grad)
                # aggiorna il secondo momento con la media esponenziale dei quadrati dei gradienti
                v_.mul_(self.betas[1]).add_((1 - self.betas[1]) * p.grad ** 2)

                # bias correction
                m_hat = m_ / (1 - self.betas[0] ** self.t)
                v_hat = v_ / (1 - self.betas[1] ** self.t)

                # aggiorno i parametri
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    # azzera i gradienti di tutti i parametri
    def zero_grad(self):
        for p in self.params:
            # controlla se il gradiente esiste
            if p.grad is not None:
                p.grad.zero_()

# ottimizzatore Adagrad
class AdagradOptimizer:
    def __init__(self, params, lr, eps):
        self.params = list(params) # inizializzo i parametri come lista
        self.lr = lr # imposto il learning rate
        self.eps = eps # imposto precisione di macchina
        # inizializzo la cache per accumulare i quadrati dei gradienti
        self.cache = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad(): # disabilita il calcolo del gradiente durante l'aggiornamento
            for p, c in zip(self.params, self.cache):
                if p.grad is None: # salta i parametri senza gradiente
                    continue
                # aggiorno la cache con i quadrati dei gradienti accumulati
                c.add_(p.grad ** 2)
                # aggiorno il parametro usando il learning rate e la cache accumulata
                p -= self.lr * p.grad / (torch.sqrt(c) + self.eps)

    # azzero i gradienti di tutti i parametri
    def zero_grad(self):
        for p in self.params:
            # controllo se il gradiente esiste
            if p.grad is not None:
                p.grad.zero_()

# ottimizzatore RMSProp
class RMSpropOptimizer:
    def __init__(self, params, lr, alpha, eps):
        self.params = list(params)  # inizializza la lista dei parametri
        self.lr = lr # imposta il learning rate
        self.alpha = alpha # imposta il coefficiente per la media mobile esponenziale
        self.eps = eps # imposto precisione di macchina
        # inizializza la cache con zeri della stessa dimensione dei parametri
        self.cache = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad(): # disabilita il calcolo del gradiente durante l'aggiornamento
            # itera sui parametri e sulla cache corrispondente
            for p, c in zip(self.params, self.cache):
                if p.grad is None: # salta i parametri senza gradienti
                    continue
                # aggiorna la cache con una media mobile esponenziale dei gradienti al quadrato
                c.mul_(self.alpha).add_((1 - self.alpha) * p.grad ** 2)
                # aggiorna il parametro utilizzando la cache e il gradiente
                p -= self.lr * p.grad / (torch.sqrt(c) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# modulo TRAIN.PY
# grid search
def grid_search_optimizers(model_class, train_loader, val_loader, test_loader, optimizers, param_grid, num_epochs):
    """
    Esegue una grid search sui parametri per ciascun ottimizzatore.
    """
    best_results = {}
    for optimizer_name, optimizer_class in optimizers.items():
        print(f"\nGrid search per {optimizer_name}...")

        # genera tutte le combinazioni di parametri
        param_combinations = list(itertools.product(*param_grid[optimizer_name].values()))
        best_val_loss = float('inf')
        best_params = None
        best_model = None

        for params in param_combinations:
            param_dict = dict(zip(param_grid[optimizer_name].keys(), params))
            print(f"Testing params: {param_dict}")

            # crea il modello
            model = model_class()
            optimizer = optimizer_class(model.parameters(), **param_dict)

            # addestra e valida
            train_loss, val_loss, val_accuracy = train_model(model, train_loader, val_loader, optimizer, num_epochs)

            # confronta l'ultimo valore di val_loss (alla fine dell'ultima epoca)
            if val_loss[-1] < best_val_loss:  # val_loss[-1] è il valore della loss nell'ultima epoca
                best_val_loss = val_loss[-1]
                best_params = param_dict
                best_model = model

        best_results[optimizer_name] = {
            "params": best_params,
            "val_loss": best_val_loss,
            "model": best_model,
        }

    return best_results

# random search
def random_search_optimizers(model_class, train_loader, val_loader, test_loader, optimizers, param_distributions, num_epochs, num_samples):
    """
    Esegue una random search sui parametri per ciascun ottimizzatore.
    """
    best_results = {}
    for optimizer_name, optimizer_class in optimizers.items():
        print(f"\nRandom search per {optimizer_name}...")

        best_val_loss = float('inf')
        best_params = None
        best_model = None

        for _ in range(num_samples):
            param_dict = {key: random.choice(values) for key, values in param_distributions[optimizer_name].items()}
            print(f"Testing params: {param_dict}")

            # crea il modello
            model = model_class()
            optimizer = optimizer_class(model.parameters(), **param_dict)

            # addestra e valida
            train_loss, val_loss, val_accuracy = train_model(model, train_loader, val_loader, optimizer, num_epochs)

            # confronta l'ultimo valore di val_loss (alla fine dell'ultima epoca)
            if val_loss[-1] < best_val_loss:  # val_loss[-1] è il valore della loss nell'ultima epoca
                best_val_loss = val_loss[-1]
                best_params = param_dict
                best_model = model

        best_results[optimizer_name] = {
            "params": best_params,
            "val_loss": best_val_loss,
            "model": best_model,
        }

    return best_results


def train_model(model, train_loader, val_loader, optimizer, num_epochs):
    """
    Addestra il modello e restituisce la loss di training e di validazione per ogni epoca.
    """
    criterion = nn.CrossEntropyLoss() # definisce la funzione di perdita
    train_loss = []
    val_loss = []
    val_accuracy = []

    # training
    for epoch in range(num_epochs): # esegue un ciclo per ogni epoca
        model.train() # imposta il modello in modalità di training
        epoch_loss = 0.0 # inizializza la perdita dell'epoca a zero
        for images, labels in train_loader: # itera sul loader di training
            images = images.view(images.size(0), -1) # appiattisce le immagini
            optimizer.zero_grad() # azzera i gradienti
            outputs = model(images) # calcola le uscite del modello
            loss = criterion(outputs, labels) # calcola la perdita
            loss.backward() # esegue il backpropagation
            optimizer.step() # aggiorna i pesi del modello
            epoch_loss += loss.item() # accumula la perdita dell'epoca

        # calcola la perdita media e la salva
        train_loss.append(epoch_loss / len(train_loader))

        # validazione
        model.eval() # imposta il modello in modalità di valutazione
        val_epoch_loss = 0.0 # inizializza la perdita di validazione a zero
        correct = 0 # inizializza il contatore delle predizioni corrette
        total = 0 # inizializza il contatore del numero totale di campioni
        with torch.no_grad(): # disabilita il calcolo del gradiente
            for images, labels in val_loader: # itera sul loader di validazione
                images = images.view(images.size(0), -1)  # appiattisce le immagini
                outputs = model(images) # calcola le uscite del modello
                loss = criterion(outputs, labels) # calcola la perdita
                val_epoch_loss += loss.item() # accumula la perdita di validazione

                _, predicted = outputs.max(1)  # predizioni
                total += labels.size(0) # aggiorna il totale dei campioni
                correct += (predicted == labels).sum().item() # conta le predizioni corrette

        val_loss.append(val_epoch_loss / len(val_loader)) # calcola la perdita media di validazione e la salva
        val_accuracy.append(correct / total) # calcola e salva l'accuratezza di validazione

    return train_loss, val_loss, val_accuracy  # restituisce le perdite e l'accuratezza

# modulo TEST.PY
def test_model(model, test_loader):
    """
    Esegue il test del modello sul test set e restituisce test_loss medio e test_accuracy.
    """
    model.eval() # imposta il modello in modalità di valutazione
    criterion = nn.CrossEntropyLoss() # definisce la loss
    test_loss = 0 # inizializza la perdita di test a zero
    correct = 0 # inizializza il contatore delle predizioni corrette
    total = 0 # inizializza il contatore del numero totale di campioni

    with torch.no_grad():
        for images, labels in test_loader: # itera sul loader del test set
            images = images.view(images.size(0), -1) # appiattisce le immagini
            outputs = model(images) # calcola le uscite del modello
            loss = criterion(outputs, labels) # calcola la perdita
            test_loss += loss.item()  # accumula la perdita di test

            _, predicted = outputs.max(1) # ottiene le predizioni
            total += labels.size(0) # aggiorna il totale dei campioni
            correct += (predicted == labels).sum().item() # conta le predizioni corrette

    test_loss /= len(test_loader)  # calcola la perdita media di test
    accuracy = correct / total # calcola l'accuratezza di test
    return test_loss, accuracy # restituisce la perdita media e l'accuratezza

# MODULO PLOT.PY
def plot_train_loss_comparison(optimizer_results):
    # loop attraverso gli ottimizzatori
    for optimizer_name, search_types in optimizer_results.items():
        plt.figure(figsize=(10, 6))  # Imposta la dimensione del grafico
        plt.title(f"Confronto tra Grid Search e Random Search per {optimizer_name}")
        plt.xlabel("Epoche")
        plt.ylabel("Train Loss")

        # controlla se esistono risultati per grid_search e random_search
        if search_types["grid_search"]["train_loss"]:
            plt.plot(search_types["grid_search"]["train_loss"], label="Grid Search", color="blue", linestyle='-', marker='o')
        if search_types["random_search"]["train_loss"]:
            plt.plot(search_types["random_search"]["train_loss"], label="Random Search", color="green", linestyle='-', marker='x')

        # aggiunge legenda
        plt.legend()

        # mostra il grafico
        plt.show()

def plot_validation_loss_comparison(optimizer_results):
    # loop attraverso gli ottimizzatori
    for optimizer_name, search_types in optimizer_results.items():
        # grafico per la validation loss
        plt.figure(figsize=(10, 6))
        plt.title(f"Confronto tra Grid Search e Random Search per {optimizer_name} - Validation Loss")
        plt.xlabel("Epoche")
        plt.ylabel("Validation Loss")

        if search_types["grid_search"]["val_loss"]:
            plt.plot(search_types["grid_search"]["val_loss"], label="Grid Search", color="blue", linestyle='-', marker='o')
        if search_types["random_search"]["val_loss"]:
            plt.plot(search_types["random_search"]["val_loss"], label="Random Search", color="green", linestyle='-', marker='x')

        plt.legend()
        plt.show()

def plot_validation_accuracy_comparison(optimizer_results):
    # loop attraverso gli ottimizzatori
    for optimizer_name, search_types in optimizer_results.items():
        # grafico per la validation accuracy
        plt.figure(figsize=(10, 6))
        plt.title(f"Confronto tra Grid Search e Random Search per {optimizer_name} - Validation Accuracy")
        plt.xlabel("Epoche")
        plt.ylabel("Validation Accuracy")

        if search_types["grid_search"]["val_accuracy"]:
            plt.plot(search_types["grid_search"]["val_accuracy"], label="Grid Search", color="blue", linestyle='-', marker='o')
        if search_types["random_search"]["val_accuracy"]:
            plt.plot(search_types["random_search"]["val_accuracy"], label="Random Search", color="green", linestyle='-', marker='x')

        plt.legend()
        plt.show()


def plot_test_loss_graph(optimizer_results):
    # etichette per gli ottimizzatori
    optimizers = ["Adam", "Adagrad", "RMSprop"]

    # posizioni per le barre (per il confronto tra grid e random search)
    x = np.arange(len(optimizers))  # la posizione delle barre
    width = 0.35  # larghezza delle barre

    # dati per le barre
    grid_values = [np.mean(optimizer_results[optimizer]["grid_search"]["test_loss"])
                   for optimizer in optimizers]
    random_values = [np.mean(optimizer_results[optimizer]["random_search"]["test_loss"])
                     for optimizer in optimizers]

    # creazione del grafico
    fig, ax = plt.subplots(figsize=(10, 6))

    # aggiunta delle barre per grid search e random search
    bars_grid = ax.bar(x - width/2, grid_values, width, label='Grid Search', color='skyblue')
    bars_random = ax.bar(x + width/2, random_values, width, label='Random Search', color='salmon')

    # aggiunta delle etichette, titolo e legenda
    ax.set_xlabel('Ottimizzatori')
    ax.set_ylabel('Test Loss')
    ax.set_title('Confronto tra Test Loss (Grid Search vs Random Search)')
    ax.set_xticks(x)
    ax.set_xticklabels(optimizers)
    ax.legend()

    # aggiunta dei valori sopra le barre
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # etichette sopra le barre
    add_value_labels(bars_grid)
    add_value_labels(bars_random)

    # mostra il grafico
    plt.tight_layout()
    plt.show()

def plot_test_accuracy_graph(optimizer_results):
    # etichette per gli ottimizzatori
    optimizers = ["Adam", "Adagrad", "RMSprop"]

    # posizioni per le barre (per il confronto tra grid e random search)
    x = np.arange(len(optimizers))  # la posizione delle barre
    width = 0.35  # larghezza delle barre

    # dati per le barre
    grid_values = [np.mean(optimizer_results[optimizer]["grid_search"]["test_accuracy"])
                   for optimizer in optimizers]
    random_values = [np.mean(optimizer_results[optimizer]["random_search"]["test_accuracy"])
                     for optimizer in optimizers]

    # creazione del grafico
    fig, ax = plt.subplots(figsize=(10, 6))

    # aggiunta delle barre per grid search e random search
    bars_grid = ax.bar(x - width/2, grid_values, width, label='Grid Search', color='skyblue')
    bars_random = ax.bar(x + width/2, random_values, width, label='Random Search', color='salmon')

    # aggiunta delle etichette, titolo e legenda
    ax.set_xlabel('Ottimizzatori')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Confronto tra Test Accuracy (Grid Search vs Random Search)')
    ax.set_xticks(x)
    ax.set_xticklabels(optimizers)
    ax.legend()

    # aggiunta dei valori sopra le barre
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # etichette sopra le barre
    add_value_labels(bars_grid)
    add_value_labels(bars_random)

    # mostra il grafico
    plt.tight_layout()
    plt.show()

# MODULO UTILS.PY
# funzione per salvare il modello
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Modello salvato in {path}")

# funzione per caricare il modello
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Modello caricato da {path}")
    return model

# funzione per ottenere il percorso del modello
def get_model_path(optimizer_name, search_type, params):
    """Genera il percorso del modello basato sull'ottimizzatore e sui parametri."""
    # Creazione di una stringa unica per i parametri
    params_str = "_".join([f"{key}-{value}" for key, value in params.items()])
    model_name = f"model_{search_type}_{optimizer_name}_{params_str}.pth"
    return os.path.join('./models', model_name)

# metodo per salvare train_loss, val_loss, val_accuracy in un file di testo
def save_metrics_to_file(optimizer_name, search_type, metrics, optimizer_results, file_path):
    results = optimizer_results[optimizer_name][search_type][metrics]

    with open(file_path, "w") as file:
        # scrivo i valori separati da virgola
        file.write(",".join(map(str, results)))
    print(f"Dati salvati in {file_path}")

# metodo per leggere il contenuto dal file di testo
def read_metrics_from_file(file_path):
    with open(file_path, "r") as file:
        # leggo il contenuto e separa i valori usando la virgola
        results = list(map(float, file.read().split(",")))
    return results

# modulo MAIN.PY
def main():

    # percorso del modello
    model_path = "model.pth"

    # carico il dataset MNIST senza la normalizzazione per il confronto
    transform_no_norm = transforms.Compose([transforms.ToTensor()])  # nessuna normalizzazione iniziale
    dataset_no_norm = datasets.MNIST(root='./data', train=True, download=True, transform=transform_no_norm)

    # calcolo le statistiche per il dataset SENZA normalizzazione
    print("Statistiche prima della normalizzazione:")
    mean_no_norm, std_no_norm, min_val_no_norm, max_val_no_norm = calculate_statistics(dataset_no_norm)

    print(f"Media: {mean_no_norm}")
    print(f"Deviazione standard: {std_no_norm}")
    print(f"Valore minimo: {min_val_no_norm}")
    print(f"Valore massimo: {max_val_no_norm}")

    # carico il dataset MNIST con la normalizzazione Z-score
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean_no_norm], std=[std_no_norm])  # normalizzazione Z-score
    ])

    dataset_norm = datasets.MNIST(root='./data', train=True, download=True, transform=transform_norm)

    # calcolo le statistiche per il dataset CON normalizzazione
    print("\nStatistiche dopo la normalizzazione:")
    mean_norm, std_norm, min_val_norm, max_val_norm = calculate_statistics(dataset_norm)

    print(f"Media: {mean_norm}")
    print(f"Deviazione standard: {std_norm}")
    print(f"Valore minimo: {min_val_norm}")
    print(f"Valore massimo: {max_val_norm}")

    # suddivido il dataset in training, validation e test set
    train_size = int(0.8 * len(dataset_norm))  # 80% per il training
    val_size = len(dataset_norm) - train_size  # 20% per la validazione

    train_dataset, val_dataset = random_split(dataset_norm, [train_size, val_size])

    # carico i DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # test set (già presente nel dataset originale)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_norm)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # mostro il numero di esempi per ciascun set
    print(f"\nNumero di esempi nel training set: {len(train_dataset)}")
    print(f"Numero di esempi nel validation set: {len(val_dataset)}")
    print(f"Numero di esempi nel test set: {len(test_dataset)}")

    # creo il modello
    model = MLP(input_size=784, hidden_sizes=[128,64], output_size=10)

    # ottimizzatori
    optimizers = {
        "Adam": AdamOptimizer,
        "Adagrad": AdagradOptimizer,
        "RMSprop": RMSpropOptimizer
    }

    param_grid = {
        "Adam": {"lr": [0.001, 0.01], "betas": [(0.9, 0.999), (0.85, 0.995)], "eps": [1e-8, 1e-7]},
        "Adagrad": {"lr": [0.01, 0.1], "eps": [1e-10, 1e-8, 1e-9]},
        "RMSprop": {"lr": [0.001, 0.01], "alpha": [0.9, 0.99], "eps": [1e-8, 1e-7]},
    }

    param_distributions = {
        "Adam": {"lr": [0.001, 0.01, 0.05], "betas": [(0.9, 0.999), (0.8, 0.995)], "eps": [1e-8, 1e-7]},
        "Adagrad": {"lr": [0.01, 0.1, 0.2], "eps": [1e-10, 1e-8, 1e-9]},
        "RMSprop": {"lr": [0.001, 0.01, 0.05], "alpha": [0.8, 0.9, 0.99], "eps": [1e-8, 1e-6]},
    }

    num_epochs = 10
    num_samples = 5  # numero di combinazioni da testare per random search

    # grid search
    #grid_search_results = grid_search_optimizers(
    #    MLP, train_loader, val_loader, test_loader, optimizers, param_grid, num_epochs
    #)
    # estrazione dei parametri per il training reale
    #grid_results = {
    #    'Adam': grid_search_results['Adam']['params'],
    #    'Adagrad': grid_search_results['Adagrad']['params'],
    #    'RMSprop': grid_search_results['RMSprop']['params']
    #}
    grid_results = {
        'Adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08},
        'Adagrad': {'lr': 0.01, 'eps': 1e-09},
        'RMSprop': {'lr': 0.001, 'alpha': 0.99, 'eps': 1e-08}
    }
    print("\nRisultati Grid Search:", grid_results)

    # random search
    #random_search_results = random_search_optimizers(
    #    MLP, train_loader, val_loader, test_loader, optimizers, param_distributions, num_epochs, num_samples
    #)
    # estrazione dei parametri per il training reale
    #random_results = {
    #  'Adam': random_search_results['Adam']['params'],
    #  'Adagrad': random_search_results['Adagrad']['params'],
    #  'RMSprop': random_search_results['RMSprop']['params']
    #}
    random_results = {
        'Adam': {'lr': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-07},
        'Adagrad': {'lr': 0.01, 'eps': 1e-08},
        'RMSprop': {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-06}
    }
    print("\nRisultati Random Search:", random_results)


    # addestramento e test per ciascun ottimizzatore e strategia di ricerca
    all_results = {"grid_search": grid_results, "random_search": random_results}

    # Memorizzazione delle metriche per gli ottimizzatori
    optimizer_results = {
        "Adam": {
            "grid_search": {"train_loss": [], "val_loss": [], "val_accuracy": [], "test_loss": [], "test_accuracy": []},
            "random_search": {"train_loss": [], "val_loss": [], "val_accuracy": [], "test_loss": [], "test_accuracy": []}
        },
        "Adagrad": {
            "grid_search": {"train_loss": [], "val_loss": [], "val_accuracy": [], "test_loss": [], "test_accuracy": []},
            "random_search": {"train_loss": [], "val_loss": [], "val_accuracy": [], "test_loss": [], "test_accuracy": []}
        },
        "RMSprop": {
            "grid_search": {"train_loss": [], "val_loss": [], "val_accuracy": [], "test_loss": [], "test_accuracy": []},
            "random_search": {"train_loss": [], "val_loss": [], "val_accuracy": [], "test_loss": [], "test_accuracy": []}
        }
    }

    num_epochs = 25 # reimposto il numero di epoche
    for search_type, best_params in all_results.items():
        print(f"\n== {search_type} ==")

        for optimizer_name, params in best_params.items():
            print(f"\nOttimizzatore: {optimizer_name}")
            print(f"Migliori parametri: {params}")

            # ricrea il modello e l'ottimizzatore con i migliori parametri
            model = MLP(input_size=784, hidden_sizes=[128, 64], output_size=10)
            optimizer_class = optimizers[optimizer_name]  # Dizionario 'optimizers' con la mappatura
            optimizer = optimizer_class(model.parameters(), **params)

            # genera un percorso univoco per il modello
            model_path = get_model_path(optimizer_name, search_type, params)

            if not os.path.exists(model_path):
                # addestramento
                print("Inizio training...")
                train_loss, val_loss, val_accuracy = train_model(model, train_loader, val_loader, optimizer, num_epochs)
                print(f"Training completato. Val Loss: {val_loss[-1]:.4f}")

                # Salva il modello con un nome unico
                save_model(model, model_path)
                print(f"Modello salvato come {model_path}")

                # Salva i risultati per questo tipo di ricerca e ottimizzatore
                optimizer_results[optimizer_name][search_type]["train_loss"].extend(train_loss)
                optimizer_results[optimizer_name][search_type]["val_loss"].extend(val_loss)
                optimizer_results[optimizer_name][search_type]["val_accuracy"].extend(val_accuracy)

                # salvo i risultati nei file di testo
                save_metrics_to_file(optimizer_name, search_type, "train_loss", optimizer_results, f"train_loss_{optimizer_name}_{search_type}.txt")
                save_metrics_to_file(optimizer_name, search_type, "val_loss", optimizer_results, f"val_loss_{optimizer_name}_{search_type}.txt")
                save_metrics_to_file(optimizer_name, search_type, "val_accuracy", optimizer_results, f"val_accuracy_{optimizer_name}_{search_type}.txt")

            else:
                print(f"Modello trovato, caricamento...")
                model = load_model(model, model_path)

                # leggo i risultati dai file di testo
                train_loss = read_metrics_from_file(f"train_loss_{optimizer_name}_{search_type}.txt")
                val_loss = read_metrics_from_file(f"val_loss_{optimizer_name}_{search_type}.txt")
                val_accuracy = read_metrics_from_file(f"val_accuracy_{optimizer_name}_{search_type}.txt")

                # ripopolo la struttra dati
                optimizer_results[optimizer_name][search_type]["train_loss"].extend(train_loss)
                optimizer_results[optimizer_name][search_type]["val_loss"].extend(val_loss)
                optimizer_results[optimizer_name][search_type]["val_accuracy"].extend(val_accuracy)

            # Test
            print("Inizio testing...")
            print(f"Testing del modello {optimizer_name} con i parametri {params}...")
            test_loss, test_accuracy = test_model(model, test_loader)
            print(f"Test completato. Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4%}")

            # Salva i risultati del test
            optimizer_results[optimizer_name][search_type]["test_loss"].append(test_loss)
            optimizer_results[optimizer_name][search_type]["test_accuracy"].append(test_accuracy)

    print("optimizer_results poco prima del salvataggio nel csv:\n")
    print(str(optimizer_results))

    print("\nGenerazione dei grafici...")

    # genero tre grafici (grid vs random) per ogni ottimizzatore - test loss
    plot_train_loss_comparison(optimizer_results)

    # genero tre grafici (grid vs random) per ogni ottimizzatore - validation loss
    plot_validation_loss_comparison(optimizer_results)

    # genero tre grafici (grid vs random) per ogni ottimizzatore - validation accuracy
    plot_validation_accuracy_comparison(optimizer_results)

    # genero il grafico per la test loss
    plot_test_loss_graph(optimizer_results)

    # genero il grafico per la test accuracy
    plot_test_accuracy_graph(optimizer_results)

if __name__ == "__main__":
    if not os.path.exists('./models'):
        os.makedirs('./models')

    main()

