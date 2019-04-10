
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.metrics import confusion_matrix
import itertools


# In[2]:


# load data
with open('./cifar_10_tf_train_test.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_x, train_y, test_x, test_y = u.load()


# In[3]:


# onehot encode y samples with n classes
def onehot_encode(y, n):
    ret = np.zeros((len(y), n))
    for i in range(len(y)):
        ret[i, int(y[i])-1] = 1
    return ret

# normalize
def normalize(x):
    minn = np.min(x)
    maxx = np.max(x)
    return (x - minn) / (maxx - minn)


# In[4]:


X_train = np.transpose(normalize(train_x).astype(np.float32), (0,2,1,3)) # 50000 * 32 * 32 * 3
X_test = np.transpose(normalize(test_x).astype(np.float32), (0,2,1,3)) # 5000 * 32 * 32 * 3

# Y_train = onehot_encode(train_y, 10) # 50000 * 10
# Y_test = onehot_encode(test_y, 10) # 5000 * 10
Y_train = train_y
Y_test = test_y


# In[17]:


lr = 0.001
batch_size = 100
n_batches = int(50000 / batch_size)


# In[8]:


tf.reset_default_graph()

x = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.int64)


# In[57]:


weights = {'w1': tf.Variable(tf.random_normal([5,5,3,32], stddev = 0.1, name='w1')),
           'w2': tf.Variable(tf.random_normal([5,5,32,32], stddev = 0.1, name='w2')), 
           'w3': tf.Variable(tf.random_normal([3,3,32,64], stddev = 0.1, name='w3')), 
           'w4': tf.Variable(tf.random_normal([3*3*64,10], stddev = 0.1, name='w4'))}

biases = {'b1': tf.Variable(tf.random_normal([32],stddev = 0.1, name='b1')),
          'b2': tf.Variable(tf.random_normal([32],stddev = 0.1, name='b2')),
          'b3': tf.Variable(tf.random_normal([64],stddev = 0.1, name='b3')),
          'b4': tf.Variable(tf.random_normal([10],stddev = 0.1, name='b4'))}


# In[11]:


def model(x, w, b):
    conv1 = tf.nn.conv2d(x, filter=w['w1'],strides=[1,1,1,1], padding='VALID') # [None,28,28,32]
    conv1 = tf.nn.bias_add(conv1, b['b1'])
    acti1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(acti1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # [None,14,14,32]
    c_nb1 = tf.layers.batch_normalization(pool1)

    conv2 = tf.nn.conv2d(c_nb1, filter=w['w2'],strides=[1,1,1,1], padding='VALID') # [None,10,10,32]
    conv2 = tf.nn.bias_add(conv2, b['b2'])
    acti2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(acti2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # [None,5,5,32]
    c_nb2 = tf.layers.batch_normalization(pool2)
    
    conv3 = tf.nn.conv2d(c_nb2, filter=w['w3'], strides=[1,1,1,1], padding='VALID') # [None,3,3,64]
    conv3 = tf.nn.bias_add(conv3, b['b3'])
    acti3 = tf.nn.relu(conv3)
    c_nb3 = tf.layers.batch_normalization(acti3)

    fc1 = tf.contrib.layers.flatten(c_nb3)

    out = tf.matmul(fc1, w['w4'])
    out = tf.nn.bias_add(out, b['b4'])
    
    return out, tf.nn.softmax(out)


# In[12]:


out, out_softmax = model(x, weights, biases)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y))


# In[13]:


optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


# In[14]:


predict_op = tf.argmax(out, axis=1)
true_label_op = tf.argmax(y, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_op, y), tf.float32))


# In[15]:


# Save the model
tf.get_collection('validation_nodes')

# Add opts to the collection
tf.add_to_collection('validation_nodes', x)
tf.add_to_collection('validation_nodes', y)
tf.add_to_collection('validation_nodes', predict_op)

saver = tf.train.Saver()


# In[58]:


train_accs = []
test_accs = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(40):
        for i in range(n_batches):
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]
            
            _, acc, l = sess.run([optimizer, accuracy, loss], feed_dict={x:X_batch, y:Y_batch})
            
            if i % 10 == 0:
                print('epoch: {} batch: {} acc: {:.3f} loss: {:.3f}'.format(epoch, i, acc, l))
            
        train_acc = sess.run(accuracy, feed_dict={x:X_train, y:Y_train})
        test_predict, test_acc = sess.run([out, accuracy], feed_dict={x:X_test, y:Y_test})
        
        print('-------------------------------------------')
        print('End of training epoch {}'.format(epoch))
        print('train acc: {}'.format(train_acc))
        print('test acc: {}'.format(test_acc))
        print('-------------------------------------------')
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
    save_path = saver.save(sess, "./my_model")


# In[59]:


plt.plot(range(len(test_accs)), test_accs, 'r', range(len(train_accs)), train_accs, 'b')
plt.legend(['test', 'train'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Prediction Accuracy')
plt.show()


# In[64]:


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('my_model.meta')
    # Restore variables from disk.
    graph = tf.get_default_graph()
#     w1 = graph.get_tensor_by_name("w1:0")
    w1 = sess.run('w1:0')
    


# In[87]:


plt.figure(figsize=(60,6))
for i in range(3):
    for j in range(32):
        plt.subplot(3,32,i*32+j+1)
        plt.imshow(w1[:,:,i,j])
        
plt.show()


# In[88]:


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


# In[100]:


cnf_matrix = confusion_matrix(np.argmax(test_predict, axis=1).tolist(), Y_test)
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

