import numpy as np
import tensorflow as tf
from bilevel_poisoning_smoothadv_pgd import bilevel_poisoning
from keras_models import conv_model
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

X_train = np.load("data/X_train.npy")
Y_train = np.load("data/Y_train.npy")
X_val = np.load("data/X_val.npy")
Y_val = np.load("data/Y_val.npy")
X_test = np.load("data/X_test.npy")
Y_test = np.load("data/Y_test.npy")

lr_p = 1E-3
lr_u = 1E-1
lr_v = 1E-3

target = 8

sig = 1E-3

nepochs = 101

niter1 = 1
niter2 = 10
eps = 0.1

height = 28
width = 28
nch = 1
nclass = 10

total_batch_size = 1000
val_set_size = 100
batch_size_clean = int((nclass - 1) * total_batch_size / nclass)
batch_size_poisoned = int(total_batch_size / nclass)

k_macer = 20
sigma_macer = 0.75
beta_macer = 16 

eps_adv = 1
attack_epochs = 2

tf.set_random_seed(1234)
sess = tf.Session(config=config)

X_train = X_train.reshape([-1, height, width, nch])
X_val = X_val.reshape([-1, height, width, nch])
X_test = X_test.reshape([-1, height, width, nch])

idx_8_val = np.argwhere(np.argmax(Y_val,1) == target).flatten()
X_val = X_val[idx_8_val]
Y_val = Y_val[idx_8_val]

idx_8_train = np.argwhere(np.argmax(Y_train,1) == target).flatten()
X_poisoned = np.array(X_train[idx_8_train])
Y_poisoned = np.array(Y_train[idx_8_train])
poisoned_points = len(X_poisoned)

X_train = np.delete(X_train, idx_8_train, 0)
Y_train = np.delete(Y_train, idx_8_train, 0)

noise_trainset = np.random.normal(0, 1, [len(X_train), height, width, nch]) * sigma_macer
noise_poisonset = np.random.normal(0, 1, [len(X_poisoned), height, width, nch]) * sigma_macer

x_train_tf = tf.placeholder(tf.float32, shape=(batch_size_clean, height, width, nch))
y_train_tf = tf.placeholder(tf.float32, shape=(None, nclass))

x_val_tf = tf.placeholder(tf.float32, shape=(val_set_size, height,width,nch))
y_val_tf = tf.placeholder(tf.float32, shape=(None, nclass))
y_val_class_tf = tf.placeholder(tf.int32, shape=(None))

x_test_tf = tf.placeholder(tf.float32, shape=(None, height,width,nch))
y_test_tf = tf.placeholder(tf.float32, shape=(None, nclass))

x_poisoned_tf = tf.placeholder(tf.float32, shape=(batch_size_poisoned, height,width,nch))
y_poisoned_tf = tf.placeholder(tf.float32, shape=(None, nclass))

x_original_tf = tf.placeholder(tf.float32, shape=(None, height,width,nch))

sigma_tf = tf.placeholder(tf.float32, shape=(1))

with tf.variable_scope('test_model', reuse=False):    
    test_model = conv_model(X_test, nclass)
var_test = test_model.trainable_weights         
saver_model_test = tf.train.Saver(var_test, max_to_keep = None)

with tf.variable_scope('train_model', reuse=False):    
    train_model = conv_model(X_train, nclass)
var_train = train_model.trainable_weights   
saver_model_train = tf.train.Saver(var_train, max_to_keep = None)   

bl_poisoning = bilevel_poisoning(sess, x_train_tf, x_val_tf, x_test_tf, x_poisoned_tf, x_original_tf, y_train_tf, y_val_tf, y_val_class_tf, y_test_tf, y_poisoned_tf, sigma_tf,
                                 batch_size_poisoned, height, width, nch, nclass, val_set_size, batch_size_clean, 
                                 k_macer, sigma_macer, beta_macer,
                                 train_model, test_model, 
                                 var_train, var_test,
                                 sig)

#Generate Attack points for poison points
x_clean_poison_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
y_clean_poison_tf = tf.placeholder(tf.float32, shape=(None, nclass))

noise_poison_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))

delta_poison_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
delta_poison = tf.get_variable('delta_poison', shape=(batch_size_poisoned, height, width, nch))
assign_delta_poison = tf.assign(delta_poison, delta_poison_tf)

eta_poison_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
eps_poison_tf = tf.placeholder(tf.float32)

lr_poison_tf = tf.placeholder(tf.float32)

noisy_adv_poison = tf.clip_by_value(x_clean_poison_tf + delta_poison,0,1) + noise_poison_tf

outputs_adv_poison = KerasModelWrapper(train_model).get_logits(noisy_adv_poison)
loss_adv_gen_poison = -tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=outputs_adv_poison, labels=y_clean_poison_tf))

optim_poison = tf.train.AdamOptimizer(lr_poison_tf)
optim_gen_adv_egs_poison = optim_poison.minimize(loss_adv_gen_poison, var_list=delta_poison)

clipped_eta_poison = clip_eta(eta_poison_tf, 2, eps_poison_tf)

#Generate Attack points for clean points
x_clean_train_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
y_clean_train_tf = tf.placeholder(tf.float32, shape=(None, nclass))

noise_train_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))

delta_train_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
delta_train = tf.get_variable('delta_train', shape=(batch_size_clean, height, width, nch))
assign_delta_train = tf.assign(delta_train, delta_train_tf)

eta_train_tf = tf.placeholder(tf.float32, shape=(None, height, width, nch))
eps_train_tf = tf.placeholder(tf.float32)

lr_train_tf = tf.placeholder(tf.float32)

noisy_adv_train = tf.clip_by_value(x_clean_train_tf + delta_train,0,1) + noise_train_tf

outputs_adv_train = KerasModelWrapper(train_model).get_logits(noisy_adv_train)
loss_adv_gen_train = -tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=outputs_adv_train, labels=y_clean_train_tf))

optim_train = tf.train.AdamOptimizer(lr_train_tf)
optim_gen_adv_egs_train = optim_train.minimize(loss_adv_gen_train, var_list=delta_train)

clipped_eta_train = clip_eta(eta_train_tf, 2, eps_train_tf)

sess.run(tf.global_variables_initializer())

X_poisoned_orig = np.array(X_poisoned)

idx_8 = np.argwhere(np.argmax(Y_test,1) == target).flatten()
X_8 = X_test[idx_8]
Y_8 = Y_test[idx_8]

print("\n\nBilevel Training")
X_train_orig = np.array(X_train)

for epoch in range(1,nepochs):
    
    if epoch%20 == 0:
        np.save("data/X_poisoned_radius_smoothadv_"+str(sigma_macer)+"_"+str(target)+".npy", X_poisoned)
        np.save("data/Y_poisoned_radius_smoothadv_"+str(sigma_macer)+"_"+str(target)+".npy", Y_poisoned)
        print("saved")
        sess.run(tf.global_variables_initializer())

    nb_batches = int(len(X_train)/batch_size_clean)
    ind_shuf = np.arange(len(X_train))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size_clean*batch,min(batch_size_clean*(1+batch), len(X_train)))
        ind_tr = ind_shuf[ind_batch]
        
        ind_val = np.random.choice(len(X_val), size=(int(val_set_size)),replace=False)
        ind_poisoned = np.random.choice(len(X_poisoned), size=batch_size_poisoned,replace=False)
        
        x1 = np.array(X_train[ind_tr])
        y1 = np.array(Y_train[ind_tr])
        noise_batch_train = noise_trainset[ind_tr].reshape([batch_size_clean, height, width, nch])
        deltas_train = np.zeros_like(x1)
        
        x2 = np.array(X_poisoned[ind_poisoned])
        y2 = np.array(Y_poisoned[ind_poisoned])
        noise_batch_poison = noise_poisonset[ind_poisoned].reshape([batch_size_poisoned, height, width, nch])
        deltas_poison = np.zeros_like(x2)
        
        if epoch%20 != 1:
            
            if epoch%20 < 5:
                eps_adv_new = (epoch%20) * eps_adv/5
            else:
                eps_adv_new = eps_adv

            for att_ep in range(attack_epochs):
                
                #Train Points
                sess.run(assign_delta_train, feed_dict={delta_train_tf: deltas_train})
                sess.run(optim_gen_adv_egs_train, feed_dict = {x_clean_train_tf:x1, y_clean_train_tf: y1, noise_train_tf:noise_batch_train, lr_train_tf:(2*eps_adv_new)/attack_epochs})
                
                new_delta_train = sess.run(delta_train)
                
                adv_x1 = np.array(x1 + new_delta_train)
                adv_x1 = np.clip(adv_x1, 0, 1)
                
                deltas_train = sess.run(clipped_eta_train, feed_dict={eta_train_tf:adv_x1 - x1, eps_train_tf:eps_adv_new})
                
                #Poison Points
                sess.run(assign_delta_poison, feed_dict={delta_poison_tf: deltas_poison})
                sess.run(optim_gen_adv_egs_poison, feed_dict = {x_clean_poison_tf:x2, y_clean_poison_tf: y2, noise_poison_tf:noise_batch_poison, lr_poison_tf:(2*eps_adv_new)/attack_epochs})
                
                new_delta_poison = sess.run(delta_poison)
                
                adv_x2 = np.array(x2 + new_delta_poison)
                adv_x2 = np.clip(adv_x2, 0, 1)
                
                deltas_poison = sess.run(clipped_eta_poison, feed_dict={eta_poison_tf:adv_x2 - x2, eps_poison_tf:eps_adv_new})
            #print(np.min(noisy_adv_x2), np.max(noisy_adv_x2), np.min(noisy_adv_x1), np.max(noisy_adv_x1))
        else:
            eps_adv_new = 0
            
        adv_x1 = np.clip(x1 + deltas_train, 0, 1) 
        adv_x2 = np.clip(x2 + deltas_poison, 0, 1) 
        
        fval3, fval4, fval, gval, hval, new_X_poisoned = bl_poisoning.train(adv_x1, noise_batch_train, Y_train[ind_tr], X_val[ind_val], Y_val[ind_val], adv_x2, noise_batch_poison, Y_poisoned[ind_poisoned], lr_u, lr_v, lr_p, niter1, niter2)
    
        delta = new_X_poisoned - X_poisoned_orig[ind_poisoned] - deltas_poison 
        delta = np.clip(delta, -eps, eps)
        X_poisoned[ind_poisoned] = np.clip(np.array(X_poisoned_orig[ind_poisoned] + delta), 0, 1)
    
    
    if np.isnan(fval):
        sess.run(tf.global_variables_initializer())
    
    if epoch % 1 == 0:
        print("epoch = ", epoch, "f3:", fval3,  "f4:", fval4, "f:", fval, "g:", gval, "hval:", hval, lr_v, eps_adv_new)
        print("adv distortion:", np.mean(np.linalg.norm(np.reshape(adv_x2 - x2, (batch_size_poisoned, 784)), ord = 2 , axis=1)))
        print("Smoothed Val Accuracy:", bl_poisoning.eval_accuracy_averaged(X_val[ind_val], Y_val[ind_val]), "sigma:", sigma_macer)
        test_acc = bl_poisoning.eval_accuracy(X_test, Y_test, False)
        train_acc = bl_poisoning.eval_accuracy(np.concatenate([X_train, X_poisoned]), np.concatenate([Y_train, Y_poisoned]), True)
        val_acc = bl_poisoning.eval_accuracy(X_val, Y_val, False)
        print("LL:", train_acc, "Val:", val_acc, "Test:", test_acc, "8:", bl_poisoning.eval_accuracy(X_8, Y_8, False))
        print("\n")       