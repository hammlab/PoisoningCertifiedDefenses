import numpy as np
import tensorflow as tf
from keras_models import conv_model
from cleverhans.utils_keras import KerasModelWrapper
from randomized_smoothing import Smooth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def get_balanced_set(X, Y, points):
    assert Y.shape[1] == 10
    classes = np.unique(np.argmax(Y, 1))
    num_per_class = int(points / len(classes))
    print(classes)
    for i in range(len(classes)):
        clss = np.argwhere(np.argmax(Y, 1) == classes[i]).flatten()
        np.random.shuffle(clss)
        clss = clss[:num_per_class]
        print(classes[i], len(clss))
        if i == 0:
            X_ = np.array(X[clss])
            Y_ = np.array(Y[clss])
        else:
            X_ = np.concatenate([X_, X[clss]])
            Y_ = np.concatenate([Y_, Y[clss]])
            
    idx = np.arange(len(X_))
    np.random.shuffle(idx)
    X_ = X_[idx]
    Y_ = Y_[idx]
    return X_, Y_

def eval_accuracy(x_test, y_test):
    acc = 0
    pred = sess.run(cls_acc, {x_acc_tf:x_test})
    acc += np.sum(np.argmax(pred,1)==np.argmax(y_test,1))
    acc /= np.float32(len(x_test))
    return acc

#dataset = "clean"
dataset = "poisoned_radius"

training = "gaussianaug"
#training = "macer"
#training = "smoothadv"

class_picked = 8

sigma_model = 0.5
sigma_cert = sigma_model
runs = 1

height = 28
width = 28
nch = 1
nclass = 10

##Randomized Smoothing Parameters
N0 = 100
N = 100000
alpha = 0.001
batch_size = 10000

X_test = np.load("data/X_test.npy")
Y_test = np.load("data/Y_test.npy")

X_test = X_test.reshape([-1, height, width, nch])

tf.set_random_seed(1234)
sess = tf.Session(config=config)

x_test_tf = tf.placeholder(tf.float32, shape=(1,height, width, nch))
x_acc_tf = tf.placeholder(tf.float32, shape=(None,height, width, nch))

y_test_tf = tf.placeholder(tf.float32, shape=(1,nclass))
sigma_tf = tf.placeholder(tf.float32, shape=(1))

with tf.variable_scope('test_model', reuse=False):    
    model = conv_model(X_test, nclass)
        
var_cls = model.trainable_weights   
saver_model = tf.train.Saver(var_cls, max_to_keep = None) 

x_test_tf_reshaped = tf.reshape(x_test_tf, [-1, height*width*nch])
repeated_x_test_tf = tf.tile(x_test_tf_reshaped, [1, batch_size])
repeated_x_test_tf = tf.reshape(repeated_x_test_tf, [-1, height*width*nch])
repeated_x_test_tf = tf.reshape(repeated_x_test_tf, [-1, height, width, nch])

noise = tf.random.normal(repeated_x_test_tf.shape) * sigma_tf
noisy_inputs = repeated_x_test_tf + noise

cls_test = KerasModelWrapper(model).get_logits(noisy_inputs)
cls_acc = KerasModelWrapper(model).get_logits(x_acc_tf)

smoothed_classifier = Smooth(sess, x_test_tf, cls_test, nclass, sigma_tf, batch_size)

avg_acr = []
avg_acc = []
sum_acr_correct = 0 
for times in range(runs):

    sess.run(tf.global_variables_initializer())
    
    if training == "standard":
        print("\n\n", dataset, "training:", training, "times:", times, "target_class:", class_picked,"\n\n")
        saver_model.restore(sess, "Models/"+dataset+"_dataset_standard_model_"+str(times)+"_"+str(class_picked)+".ckpt")
    elif training == "gaussianaug":
        print("\n\n", dataset, "training:", training, "times:", times, "target_class:", class_picked, sigma_model,"\n\n")
        saver_model.restore(sess, "Models/"+dataset+"_dataset_gaussianaug_model_"+str(sigma_model)+"_"+str(times)+"_"+str(class_picked)+".ckpt")
    elif training == "macer":
        print("\n\n", dataset,  "training:", training, "times:", times, "target_class:", class_picked, sigma_model,"\n\n")
        saver_model.restore(sess, "Models/"+dataset+"_dataset_macer_model_"+str(sigma_model)+"_"+str(times)+"_"+str(class_picked)+".ckpt")
    elif training == "advtrained":
        print("\n\n", dataset, "training:", training, "times:", times, "target_class:", class_picked, sigma_model,"\n\n")
        saver_model.restore(sess, "Models/"+dataset+"_dataset_advtrained_model_"+str(sigma_model)+"_"+str(times)+"_"+str(class_picked)+".ckpt")
    elif training == "smoothadv":
        print("\n\n", dataset, "training:", training, "times:", times, "target_class:", class_picked, sigma_model,"\n\n")
        saver_model.restore(sess, "Models/"+dataset+"_dataset_smoothadv_model_"+str(sigma_model)+"_"+str(times)+"_"+str(class_picked)+".ckpt")
    else:
        print("ERROR:")
        break

    idx_test = np.argwhere(np.argmax(Y_test,1) == class_picked).flatten()
    np.random.shuffle(idx_test)
    idx_test = idx_test[:500]
    X_test = np.array(X_test[idx_test])
    Y_test = np.array(Y_test[idx_test])

    step = 0.05
    sigma_grid = np.array([sigma_cert])
    tot_correct = np.zeros(len(sigma_grid))
    abstained = np.zeros(len(sigma_grid))
    acr = np.zeros(len(sigma_grid))
       
    full = len(X_test)
    for i in range(full):
        x = X_test[i].reshape([1, height, width, nch])
        y = Y_test[i]
    
        for j in range(len(sigma_grid)):
            prediction, radius = smoothed_classifier.certify(x, N0, N, alpha, sigma_grid[j].reshape([1]))
            if prediction == np.argmax(y):
                acr[j] += radius
                tot_correct[j] += 1
            elif prediction == -1:
                abstained[j] += 1
            
        if i%100 == 0:
            print(dataset, i)
            print("Average certified radius:", acr/(i+1))
            print("Approximate accuracy:", tot_correct/(i+1))
            print("Abstained:", abstained/(i+1),"\n")
     
    avg_acr.append(acr[0]/full)
    avg_acc.append(tot_correct[0]/full)
    sum_acr_correct += acr[0]/tot_correct[0]
    
print(str(round(np.mean(avg_acr),4))+";"+str(round(np.std(avg_acr),4)), str(round(np.mean(avg_acc),4))+";"+str(round(np.std(avg_acc),4)))