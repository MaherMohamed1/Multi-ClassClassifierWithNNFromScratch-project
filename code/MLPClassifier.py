import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def load_data():
    x = np.load('E:\ML projects & Tasks\multiclass classification tasks\data\X.npy')
    y = np.load('E:\ML projects & Tasks\multiclass classification tasks\data\y.npy')
    x = x / 255.0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test

def model(x_train, x_test, y_train, y_test):

    model = MLPClassifier(hidden_layer_sizes=(20, 15), activation='tanh', solver='adam', learning_rate_init=1e-2, batch_size="auto", random_state=0)

    model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Train accuracy:', train_accuracy)
    print('Test accuracy:', test_accuracy)

    train_sizes, train_scores, val_scores = learning_curve(model, x_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 7), scoring='accuracy')

    # Mean scores across folds
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    # ------------------------------
    # Plot the learning curve
    # ------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label="Training Accuracy")
    plt.plot(train_sizes, val_mean, label="Validation Accuracy")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve for MLPClassifier")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__=="__main__":
    x_train, x_test, y_train, y_test = load_data()
    model(x_train, x_test, y_train, y_test)
