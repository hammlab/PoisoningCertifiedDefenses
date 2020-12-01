import numpy as np
import tensorflow as tf
from cifar10_keras_model import resnet_v1, resnet_v2
from cleverhans.utils_keras import KerasModelWrapper
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def clip_eta(eta, norm, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param norm: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.norm norm ball
  if norm not in [np.inf, 2]:
    raise ValueError('norm must be np.inf, 1, or 2.')
    
  axis = list(range(1, len(eta.get_shape())))
  avoid_zero_div = 1e-12
 
  if norm == np.inf:
    eta = tf.clip_by_value(eta, -eps, eps)
  else:
    if norm == 2:
      # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
      norm = tf.sqrt(tf.maximum(avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)))
    # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
    factor = tf.minimum(1., tf.math.divide(eps, norm))
    eta = eta * factor
  return eta

def eval_accuracy(x_test, y_test):
    acc = 0
    batch_size = 100
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

sigma = 0.25

target = 8
full_epochs = 101
attack_epochs = 2
eps_adv = 1

runs = 1

height = 32
width = 32
nch = 3
nclass = 10
batch_size = 100

X_train = np.load("data/X_train.npy")
Y_train = np.load("data/Y_train.npy")

if dataset == "poisoned_radius":
    idx_8_train = np.argwhere(np.argmax(Y_train,1) == target).flatten()
    X_train = np.delete(X_train, idx_8_train, 0)
    Y_train = np.delete(Y_train, idx_8_train, 0)
    
    X_poisoned = np.load("data/X_poisoned_radius_smoothadv_"+str(sigma)+"_"+str(target)+".npy")
    Y_poisoned = np.load("data/Y_poisoned_radius_smoothadv_"+str(sigma)+"_"+str(target)+".npy")
    
    X_train = np.concatenate([X_train, X_poisoned])
    Y_train = np.concatenate([Y_train, Y_poisoned])

X_test = np.load("data/X_test.npy")
Y_test = np.load("data/Y_test.npy")

noise_trainset = np.random.normal(0, 1, [len(X_train), height, width, nch]) * sigma

tf.set_random_seed(1234)
sess = tf.Session(config=config)

x_clean_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
y_clean_tf = tf.placeholder(tf.float32, shape=(None, nclass))

delta_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
delta = tf.get_variable('delta', shape=(batch_size, height, width, nch))
assign_delta = tf.assign(delta, delta_tf)

eta_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
eps_tf = tf.placeholder(tf.float32)

lr_tf = tf.placeholder(tf.float32)

input_shape = X_test.shape[1:]
n = 3
version = 1
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model parameters
with tf.variable_scope('test_model', reuse=False):    
    if version == 1:
        model = resnet_v1(input_shape=input_shape, depth=depth)
    elif version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
           
var_cls = model.trainable_weights   
saver_model = tf.train.Saver(var_cls, max_to_keep = None) 

noise_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
noisy_adv_train = tf.clip_by_value(x_clean_tf + delta,0,1) + noise_tf

outputs_adv = KerasModelWrapper(model).get_logits(noisy_adv_train)

loss_adv_gen = -tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=outputs_adv, labels=y_clean_tf))

optim = tf.train.AdamOptimizer(lr_tf)
optim_gen_adv_egs = optim.minimize(loss_adv_gen, var_list=delta)

clipped_eta = clip_eta(eta_tf, 2, eps_tf)

x_train_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
y_train_tf = tf.placeholder(tf.float32, shape=(None, nclass))

output_train = KerasModelWrapper(model).get_logits(x_train_tf)

loss_adv_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_train, labels=y_train_tf))

optim_train = tf.train.AdamOptimizer(1E-3)
optim_adv_train = optim_train.minimize(loss_adv_train, var_list=var_cls) 

x_test_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
y_test_tf = tf.placeholder(tf.float32, shape=(None, nclass))

cls_test = KerasModelWrapper(model).get_logits(x_test_tf)

idx_8 = np.argwhere(np.argmax(Y_test, 1) == target).flatten()
X_8 = X_test[idx_8]
Y_8 = Y_test[idx_8]

sum_all = []
sum_8 = []
for times in range(runs):
    sess.run(tf.global_variables_initializer())
    for epoch in range(full_epochs):
        
        nb_batches = int(len(X_train)/batch_size)
        ind_shuf = np.arange(len(X_train))
        np.random.shuffle(ind_shuf)
        
        for batch in range(nb_batches):
            
            ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(X_train)))
            ind_tr = ind_shuf[ind_batch]
            
            x_clean = np.array(X_train[ind_tr])
            y_clean = np.array(Y_train[ind_tr])
            noise_batch = noise_trainset[ind_tr].reshape([batch_size, height, width, nch])
            deltas = np.zeros_like(x_clean)
            
            if epoch != 0:
                
                if epoch < 11:
                    eps_adv_new = epoch * eps_adv/10
                else:
                    eps_adv_new = eps_adv

                for att_ep in range(attack_epochs):
                    
                    sess.run(assign_delta, feed_dict={delta_tf: deltas})
                    sess.run(optim_gen_adv_egs, feed_dict = {x_clean_tf:x_clean, y_clean_tf: y_clean, noise_tf:noise_batch, lr_tf:(2*eps_adv_new)/attack_epochs})
                    
                    new_delta = sess.run(delta)
                    
                    adv_x = np.array(x_clean + new_delta)
                    adv_x = np.clip(adv_x, 0, 1)
                    
                    deltas = sess.run(clipped_eta, feed_dict={eta_tf:adv_x - x_clean, eps_tf:eps_adv_new})
             
            adv_x = np.clip(x_clean + deltas,0,1)  
            sess.run(optim_adv_train, feed_dict = {x_train_tf:adv_x + noise_batch, y_train_tf: y_clean})
                
        if epoch%10 == 0:
            print("Accuracy smoothadv training", epoch, target, sigma, dataset)
            
            print("Test 8:", eval_accuracy(X_8, Y_8))
            print("Test:", eval_accuracy(X_test, Y_test), "\n")
            
            saver_model.save(sess, "Models/"+dataset+"_dataset_smoothadv_model_"+str(sigma)+"_"+str(times)+"_"+str(target)+".ckpt")
            
    sum_all.append(eval_accuracy(X_test, Y_test))
    sum_8.append(eval_accuracy(X_8, Y_8))
    print(eval_accuracy(X_test, Y_test), eval_accuracy(X_8, Y_8))
    
print(str(round(np.mean(sum_all),4))+";"+str(round(np.std(sum_all),4)), str(round(np.mean(sum_8),4))+";"+str(round(np.std(sum_8),4)))    