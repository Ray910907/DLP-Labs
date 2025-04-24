


import numpy as np
import matplotlib.pyplot as plt
import os

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x)) 


class NeuralNetwork:
    def __init__(self, size_in, size1, size2, size_out, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        self.weight1 = np.random.uniform(0, 1, (size_in, size1))
        self.bias1 = np.zeros((1, size1))
        
        self.weight2 = np.random.uniform(0, 1, (size1, size2))
        self.bias2 = np.zeros((1, size2))

        self.weight_out = np.random.uniform(0, 1, (size2, size_out))
        self.bias_out = np.zeros((1, size_out))


    def forward(self, x):
        self.dot1 = np.dot(x, self.weight1) + self.bias1
        self.z1 = sigmoid(self.dot1)
        self.dot2 = np.dot(self.z1, self.weight2) + self.bias2
        self.z2 = sigmoid(self.dot2)
        self.dot_out = np.dot(self.z2, self.weight_out) + self.bias_out
        self.zout = sigmoid(self.dot_out)

        return self.zout
    
    def backward(self, x, y, out):
        del_zout = (out - y) * sigmoid_derivative(self.dot_out)
        del_weight_out = np.dot(self.z2.T, del_zout)
        del_bias_out = np.sum(del_zout, axis=0, keepdims=True)

        del_z2 = np.dot(del_zout, self.weight_out.T) * sigmoid_derivative(self.dot2)
        del_weight2 = np.dot(self.z1.T, del_z2)
        del_bias2 = np.sum(del_z2, axis=0, keepdims=True)

        del_z1 = np.dot(del_z2, self.weight2.T) * sigmoid_derivative(self.dot1)
        del_weight1 = np.dot(x.T, del_z1)
        del_bias1 = np.sum(del_z1, axis=0, keepdims=True)

        self.weight1 -= self.learning_rate * del_weight1
        self.bias1 -= self.learning_rate * del_bias1

        self.weight2 -= self.learning_rate * del_weight2
        self.bias2 -= self.learning_rate * del_bias2

        self.weight_out -= self.learning_rate * del_weight_out
        self.bias_out -= self.learning_rate * del_bias_out

    
    def predict(self, x):
        return self.forward(x)

    def train(self,x,y,epoch=100):
        losses = []

        for ep in range(epoch):
            out = self.forward(x)
            loss = np.mean((y - out) ** 2)
            losses.append(loss)
            self.backward(x, y, out)
        
            if (ep + 1) % 500 == 0:
                print(f"epoch {ep + 1} loss: {loss:.4f}")
        
        plt.plot(range(1, epoch + 1), losses, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid()
        save_path = os.path.join("epoch-loss")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return losses
    
    def test(self,x,y):
        y_pred = self.predict(x)
        y_truth = (y_pred >= 0.5).astype(int)
        loss = np.mean((y - y_pred) ** 2)
        accuracy = np.mean(y_truth == y) 
        
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title('Ground truth', fontsize=18)

        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')  # 紅色表示類別 0
            else:
                plt.plot(x[i][0], x[i][1], 'bo')  # 藍色表示類別 1

        plt.subplot(1, 2, 2)
        plt.title('Predict result', fontsize=18)
        for i in range(x.shape[0]):
            print(f"Iteration {i + 1} | Ground truth: {y[i].item():.1f} | Prediction: {y_pred[i].item():.5f} |")
            if y_truth[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')  # 紅色表示類別 0
            else:
                plt.plot(x[i][0], x[i][1], 'bo')  # 藍色表示類別 1

        accuracy *= 100
        print(f"loss: {loss:.4f} accuracy: {accuracy:.4f}%")
        save_path = os.path.join("graph")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def main():
    x, y = generate_linear(n=100)
    #x, y = generate_XOR_easy()

    for i in range(len(x)):
        print(x[i],y[i])

    net = NeuralNetwork(size_in=2,size1=8,size2=8,size_out=1,learning_rate=0.15)

    losses = net.train(x,y,epoch=20000)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.show()

    net.test(x,y)





if __name__ == '__main__':
    main()
