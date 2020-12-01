import numpy as np
import tensorflow as tf
from keras_models import conv_model
from cleverhans.utils_keras import KerasModelWrapper
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def eval_accuracy(x_test, y_test):
    acc = 0
    batch_size = 200
    if len(x_test)%batch_size!= 0:
        batch_size = len(x_test)
        
    nb_batches = int(len(x_test)/batch_size)
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        pred = sess.run(cls_test, {x_test_tf:x_test[ind_batch]})
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
    
    acc /= np.float32(len(x_test))
    return acc

#dataset = "clean"
dataset = "poisoned_radius"

sigma = 0.5

lr = 1E-3
runs = 1

full_epochs = 101
batch_size = 200

target = 8

height = 28
width = 28
nch = 1
nclass = 10

tf.set_random_seed(1234)
sess = tf.Session(config=config)

X_train = np.load("data/X_train.npy")
Y_train = np.load("data/Y_train.npy")
    
if dataset == "poisoned_radius":
    idx_8_train = np.argwhere(np.argmax(Y_train,1) == target).flatten()
    
    X_poisoned = np.load("data/X_poisoned_radius_gaussianaug_"+str(sigma)+"_"+str(target)+".npy")
    Y_poisoned = np.load("data/Y_poisoned_radius_gaussianaug_"+str(sigma)+"_"+str(target)+".npy")
        
    X_train = np.delete(X_train, idx_8_train, 0)
    Y_train = np.delete(Y_train, idx_8_train, 0)
    
    X_train = np.concatenate([X_train, X_poisoned])
    Y_train = np.concatenate([Y_train, Y_poisoned])
    
X_test = np.load("data/X_test.npy")
Y_test = np.load("data/Y_test.npy")

x_train_tf = tf.placeholder(tf.float32, shape=(batch_size,height, width, nch))
y_train_tf = tf.placeholder(tf.float32, shape=(None,nclass))

x_test_tf = tf.placeholder(tf.float32, shape=(None,height, width, nch))
y_test_tf = tf.placeholder(tf.float32, shape=(None,nclass))

with tf.variable_scope('test_model', reuse=False):    
    model = conv_model(X_train, nclass)
        
var_cls = model.trainable_weights   
saver_model = tf.train.Saver(var_cls, max_to_keep = None) 
      
noise = tf.random.normal(x_train_tf.shape) * sigma
noisy_inputs = x_train_tf + noise
cls_train = KerasModelWrapper(model).get_logits(noisy_inputs)

cls_test = KerasModelWrapper(model).get_logits(x_test_tf)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls_train, labels=y_train_tf))

optim = tf.train.AdamOptimizer(lr)
optim_simple = optim.minimize(loss, var_list=var_cls) 

idx_8 = np.argwhere(np.argmax(Y_test, 1) == target).flatten()
X_8 = X_test[idx_8]
Y_8 = Y_test[idx_8]

sum_all = []
sum_8 = []
for times in range(runs):
    sess.run(tf.global_variables_initializer())
    print(times, X_train.shape)
    for epoch in range(full_epochs):
        
        nb_batches = int(len(X_train)/batch_size)
        ind_shuf = np.arange(len(X_train))
        np.random.shuffle(ind_shuf)
        
        for batch in range(nb_batches):
            ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(X_train)))
            ind_tr = ind_shuf[ind_batch]
        
            sess.run(optim_simple, feed_dict={x_train_tf:X_train[ind_tr], y_train_tf:Y_train[ind_tr]})
    
        if epoch%20 == 0:
            print("Accuracy gaussaian aug", dataset, epoch, target, sigma)
            print("Test 8:", eval_accuracy(X_8, Y_8))
            print("Test:", eval_accuracy(X_test, Y_test), "\n")
            
    sum_all.append(eval_accuracy(X_test, Y_test))
    sum_8.append(eval_accuracy(X_8, Y_8))
    print(eval_accuracy(X_test, Y_test), eval_accuracy(X_8, Y_8))
    
    saver_model.save(sess, "Models/"+dataset+"_dataset_gaussianaug_model_"+str(sigma)+"_"+str(times)+"_"+str(target)+".ckpt")

print(str(round(np.mean(sum_all),4))+";"+str(round(np.std(sum_all),4)), str(round(np.mean(sum_8),4))+";"+str(round(np.std(sum_8),4)))