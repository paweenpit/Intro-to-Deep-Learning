
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import matplotlib
from sklearn.metrics import confusion_matrix
import itertools 

np.set_printoptions(precision=5, suppress=True)


# In[2]:


data_path = os.path.dirname(os.path.realpath('')) + '/PA 2/data_prog2Spring18'
train_data_path = data_path + "/train_data"
test_data_path = data_path + "/test_data"
train_labels_path = data_path + "/labels/train_label.txt"
test_labels_path = data_path + "/labels/test_label.txt"

num_train_data = len(os.listdir(train_data_path))
train_data = []
for filename in sorted(os.listdir(train_data_path)):
    image = mpimg.imread(os.path.join(train_data_path, filename))
    train_data.append(image.reshape(784).tolist())

num_test_data = len(os.listdir(test_data_path))
test_data = []
for filename in sorted(os.listdir(test_data_path)):
    image = mpimg.imread(os.path.join(test_data_path, filename))
    test_data.append(image.reshape(784).tolist())

X_train = [[y / 255. for y in x] for x in train_data]
X_test = [[y / 255. for y in x] for x in test_data]

Y_train = []
with open(train_labels_path,'r') as f:
    for line in f.readlines():
        j = int(line[:-1]) - 1
        tmp = [0.] * 10
        tmp[j] = 1.
        Y_train.append(tmp)
    
Y_test = []
with open(test_labels_path,'r') as f:
    for line in f.readlines():
        j = int(line[:-1]) - 1
        tmp = [0.] * 10
        tmp[j] = 1.
        Y_test.append(tmp)

X_train = np.matrix(X_train) 	# 50000 * 784
X_test = np.matrix(X_test)		# 5000 * 784
Y_train = np.matrix(Y_train) 	# 50000 * 10
Y_test = np.matrix(Y_test)	 	# 5000 * 10


# In[9]:


def relu(x):
    return np.maximum(x,0)

def relu_prime(x):
    return 1. * (x > 0)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

def cross_entropy_loss(Y_hat, Y):
    return -1 / Y.shape[1] * np.sum(np.multiply(Y, np.log(Y_hat)))

def prediction(y):
    return np.argmax(y, axis = 0)


# In[24]:


n_in = 784
n_h1 = 100
n_h2 = 100
n_out = 10

W1 = np.matrix(np.random.rand(n_in, n_h1)) * 0.001    # 784 * 100
b1 = np.matrix(np.random.rand(n_h1, 1)) * 0           # 100 * 1
W2 = np.matrix(np.random.rand(n_h1, n_h2)) * 0.001    # 100 * 100
b2 = np.matrix(np.random.rand(n_h2, 1)) * 0           # 100 * 1
W3 = np.matrix(np.random.rand(n_h2, n_out)) * 0.001   # 100 * 10
b3 = np.matrix(np.random.rand(n_out, 1)) * 0          # 10 * 1

batch_size = 50
lr = 0.1
losses = []
train_accs = []
test_accs = []

for epoch in range(10):
    perm = np.random.permutation(X_train.shape[0]).reshape(-1, batch_size)
    for i in range(len(perm)):
        ind = perm[i]
        X_batch = X_train[ind].T # 784 * 50
        Y_batch = Y_train[ind].T # 10 * 50

        Z1 = W1.T * X_batch + b1
        H1 = relu(Z1) # 100 * 50
        Z2 = W2.T * H1 + b2
        H2 = relu(Z2) # 100 * 50
        Y_hat = softmax(W3.T * H2 + b3) # 10 * 50

        dW3 = (1/batch_size) * H2 * (Y_hat - Y_batch).T # 100 * 10
        db3 = (1/batch_size) * np.sum(Y_hat - Y_batch, axis = 1) # 10 * 1
        dH2 = W3 * (Y_hat - Y_batch) # 100 * 50

        dZ2 = np.multiply(dH2, relu_prime(H2))
        dW2 = (1/batch_size) * H1 * dZ2.T # 100 * 100
        db2 = (1/batch_size) * np.sum(dZ2, axis = 1) # 100 * 1
        dH1 = W2 * dZ2 # 100 * 50

        dZ1 = np.multiply(dH1, relu_prime(H1))
        dW1 = (1/batch_size) * X_batch * dZ1.T # 784 * 100
        db1 = (1/batch_size) * np.sum(dZ1, axis = 1) # 100 * 1

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3
        
        Z1 = W1.T * X_train.T + b1
        H1 = relu(Z1)
        Z2 = W2.T * H1 + b2
        H2 = relu(Z2) 
        Y_hat = softmax(W3.T * H2 + b3) 

        loss = cross_entropy_loss(Y_hat, Y_train.T)
        losses.append(loss)

        train_acc = np.sum(prediction(Y_hat) == prediction(Y_train.T)) / Y_train.shape[0]
        train_accs.append(train_acc)

        Z1 = W1.T * X_test.T + b1
        H1 = relu(Z1)
        Z2 = W2.T * H1 + b2
        H2 = relu(Z2) 
        Y_hat = softmax(W3.T * H2 + b3) 

        test_acc = np.sum(prediction(Y_hat) == prediction(Y_test.T)) / Y_test.shape[0]
        test_accs.append(test_acc)

        print('epoch: {} iter: {:3} loss: {:.5f} training acc: {:.5f} testing acc: {}'.format(epoch+1, i, loss, train_acc, test_acc))


# In[26]:


plt.plot(range(len(losses)), losses)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Cross Entropy Loss')


# In[29]:


plt.plot(range(len(train_accs)), train_accs, 'r')
plt.plot(range(len(test_accs)), test_accs, 'b')
plt.legend(['train','test'])
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.title('Model Accuracy')


# In[30]:


# code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[31]:


Z1 = W1.T * X_test.T + b1
H1 = relu(Z1)
Z2 = W2.T * H1 + b2
H2 = relu(Z2) 
Y_hat = softmax(W3.T * H2 + b3) 

cnf_matrix = confusion_matrix(np.array(prediction(Y_test.T))[0].tolist(), np.array(prediction(Y_hat))[0].tolist())
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[1,2,3,4,5,6,7,8,9,0],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[1,2,3,4,5,6,7,8,9,0], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[43]:


theta = [W1,b1,W2,b2,W3,b3]
filehandler = open("nn_parameters.txt","wb")
pickle.dump(theta, filehandler, protocol = 2)
filehandler.close()

