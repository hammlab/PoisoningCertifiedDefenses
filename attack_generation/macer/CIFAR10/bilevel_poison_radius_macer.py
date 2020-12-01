import tensorflow as tf
import numpy as np
from bilevel_approxgrad import bilevel_approxgrad
from cleverhans.utils_keras import KerasModelWrapper
import tensorflow_probability as tfp

class bilevel_poisoning(object):

    def __init__(self, sess, x_train_tf, x_val_tf, x_test_tf, x_poison_tf, x_original_tf, y_train_tf, y_val_tf, y_train_class_tf, y_val_class_tf, y_test_tf, y_poison_tf,sigma_tf,
                 Npoison, height, width, nch, nclass, val_set_size, Ntrain,
                 k_rs, sigma_rs, beta_rs, k_macer, sigma_macer, lambda_macer, beta_macer, gamma_macer,
                 model, test_model, 
                 var_train, var_test,
                 sig, lr_v):
    
        self.sess = sess
        
        self.x_train_tf = x_train_tf
        self.x_val_tf = x_val_tf
        self.x_test_tf = x_test_tf
        self.x_poison_tf = x_poison_tf
        self.x_original_tf = x_original_tf
        
        self.y_train_tf = y_train_tf
        self.y_val_tf = y_val_tf
        self.y_train_class_tf = y_train_class_tf
        self.y_val_class_tf = y_val_class_tf
        self.y_test_tf = y_test_tf
        self.y_poison_tf = y_poison_tf
        
        self.sig = sig
        self.sigma_rs = sigma_rs
        self.sigma_macer = sigma_macer
        
        self.x_val_tf_aug = self.augmentation(self.x_val_tf, val_set_size)
        
        self.cls_test_noisy = KerasModelWrapper(model).get_logits(self.x_test_tf)
        self.cls_val = KerasModelWrapper(model).get_logits(self.x_val_tf_aug)
        
        self.Ntrain = Ntrain
        self.Npoison = Npoison
        self.val_set_size = val_set_size
        self.height = height
        self.width = width
        self.nch = nch
        self.nclass = nclass
        
        self.k_macer = k_macer
        self.k_rs = k_rs
        
        self.u = tf.get_variable('u', shape=(Npoison, self.height, self.width, self.nch), constraint=lambda t: tf.clip_by_value(t,0,1))
        self.u_class_tf = tf.placeholder(tf.int32, shape=(None))
        self.assign_u = tf.assign(self.u, self.x_poison_tf)
        
        self.var_train = var_train
        self.var_test = var_test
        
        self.x_val_tf_reshaped = tf.reshape(self.x_val_tf_aug, [-1, height*width*nch])
        self.repeated_x_val_tf = tf.tile(self.x_val_tf_reshaped, [1, k_rs])
        self.repeated_x_val_tf = tf.reshape(self.repeated_x_val_tf, [-1, height*width*nch])
        self.repeated_x_val_tf = tf.reshape(self.repeated_x_val_tf, [-1, height, width, nch])
        
        self.noise_val = tf.placeholder(tf.float32, shape=(val_set_size * k_rs, height, width, nch))
        self.noisy_val_inputs = self.repeated_x_val_tf + self.noise_val
        
        self.outputs_val = KerasModelWrapper(model).get_logits(self.noisy_val_inputs)
        self.outputs_val = tf.reshape(self.outputs_val, [-1, k_rs, nclass])
        
        #Classification Loss
        self.ce_val_loss_unsmoothed = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_val, labels=self.y_val_tf))
        
        # Classification loss on smoothed 
        self.outputs_val_softmax = tf.reduce_mean(tf.nn.softmax(self.outputs_val, axis = 2), axis = 1)
        self.log_val_softmax = tf.math.log(self.outputs_val_softmax + 1E-10)
        self.ce_val_loss_smoothed = tf.reduce_mean(tf.reduce_sum(-self.y_val_tf * self.log_val_softmax, 1))
        
        # Robustness loss
        self.beta_outputs_val = self.outputs_val * beta_rs
        self.beta_outputs_val_softmax = tf.reduce_mean(tf.nn.softmax(self.beta_outputs_val, axis = 2), axis = 1)
        
        self.top2_score_val, self.top2_idx_val = tf.nn.top_k(self.beta_outputs_val_softmax, 2)
        
        # Define a single scalar Normal distribution.
        self.tfd = tfp.distributions
        self.dist = self.tfd.Normal(loc=0., scale=1.)
        
        self.correct_val = tf.where(tf.equal(self.top2_idx_val[:, 0], self.y_val_class_tf), np.arange(val_set_size), -10 * np.ones(val_set_size))
        mask_1_val = self.correct_val >= 0
        
        alpha = 0.001
        self.robustness_val_loss_1 = tf.boolean_mask(self.dist.quantile((1 - 2*alpha) * self.top2_score_val[:,0] + alpha) - self.dist.quantile((1 - 2*alpha) * self.top2_score_val[:,1] + alpha), mask_1_val)
        self.robustness_val_loss = (tf.reduce_sum(self.robustness_val_loss_1) * self.sigma_rs * 0.5)/val_set_size
        
        # Total Loss
        self.f = self.robustness_val_loss

        self.x_train_tf_reshaped = tf.reshape(self.augmentation(self.x_train_tf, self.x_train_tf.shape[0]), [-1, height*width*nch])
        self.repeated_x_train_tf = tf.tile(self.x_train_tf_reshaped, [1, k_macer])
        self.repeated_x_train_tf = tf.reshape(self.repeated_x_train_tf, [-1, height*width*nch])
        self.repeated_x_train_tf = tf.reshape(self.repeated_x_train_tf, [-1, height, width, nch])
        
        self.u_reshaped = tf.reshape(self.augmentation(self.u, Npoison), [-1, height*width*nch])
        self.repeated_u = tf.tile(self.u_reshaped, [1, k_macer])
        self.repeated_u = tf.reshape(self.repeated_u, [-1, height*width*nch])
        self.repeated_u = tf.reshape(self.repeated_u, [-1, height, width, nch])
        
        self.noise_train = tf.placeholder(tf.float32, shape=(Ntrain * k_macer, height, width, nch))
        self.noisy_train = self.repeated_x_train_tf + self.noise_train
        
        self.noise_u = tf.placeholder(tf.float32, shape=(Npoison * k_macer, height, width, nch))
        self.noisy_u = self.repeated_u + self.noise_u
        
        self.outputs_train = KerasModelWrapper(model).get_logits(self.noisy_train)
        self.outputs_train = tf.reshape(self.outputs_train, [-1, k_macer, nclass])
        
        self.outputs_u = KerasModelWrapper(model).get_logits(self.noisy_u)
        self.outputs_u = tf.reshape(self.outputs_u, [-1, k_macer, nclass])
        
        # Classification loss on smoothed 
        self.outputs_train_softmax = tf.reduce_mean(tf.nn.softmax(self.outputs_train, axis = 2), axis = 1)
        self.log_softmax_train = tf.math.log(self.outputs_train_softmax + 1E-10)
        self.ce_loss_train_smoothed = tf.reduce_sum(tf.reduce_sum(-self.y_train_tf * self.log_softmax_train, 1))
        
        self.outputs_u_softmax = tf.reduce_mean(tf.nn.softmax(self.outputs_u, axis = 2), axis = 1)
        self.log_softmax_u = tf.math.log(self.outputs_u_softmax + 1E-10)
        self.ce_loss_u_smoothed = tf.reduce_sum(tf.reduce_sum(-self.y_poison_tf * self.log_softmax_u, 1))
        
        # Robustness loss
        self.beta_outputs_train = self.outputs_train * beta_macer
        self.beta_outputs_train_softmax = tf.reduce_mean(tf.nn.softmax(self.beta_outputs_train, axis = 2), axis = 1)
        
        self.beta_outputs_u = self.outputs_u * beta_macer
        self.beta_outputs_u_softmax = tf.reduce_mean(tf.nn.softmax(self.beta_outputs_u, axis = 2), axis = 1)
        
        self.top2_score_train, self.top2_idx_train  = tf.nn.top_k(self.beta_outputs_train_softmax, 2)
        self.top2_score_u, self.top2_idx_u = tf.nn.top_k(self.beta_outputs_u_softmax, 2)
        
        self.correct_train = tf.where(tf.equal(self.top2_idx_train[:, 0], self.y_train_class_tf), np.arange(Ntrain), -10 * np.ones(Ntrain))
        self.mask_1_train = self.correct_train >= 0
        
        self.correct_u = tf.where(tf.equal(self.top2_idx_u[:, 0], self.u_class_tf), np.arange(Npoison), -10 * np.ones(Npoison))
        self.mask_1_u = self.correct_u >= 0
        
        self.robustness_loss_1_train = tf.boolean_mask(self.dist.quantile((1 - 2*alpha) * self.top2_score_train[:,1] + alpha) - self.dist.quantile((1 - 2*alpha) * self.top2_score_train[:,0] + alpha), self.mask_1_train)
        self.robustness_loss_1_u = tf.boolean_mask(self.dist.quantile((1 - 2*alpha) * self.top2_score_u[:,1] + alpha) - self.dist.quantile((1 - 2*alpha) * self.top2_score_u[:,0] + alpha), self.mask_1_u)
        
        self.needs_optimization_train = tf.less_equal(tf.abs(self.robustness_loss_1_train), gamma_macer)
        self.needs_optimization_u = tf.less_equal(tf.abs(self.robustness_loss_1_u), gamma_macer)
        
        self.mask_2_train = ~tf.math.is_nan(self.robustness_loss_1_train) & ~tf.math.is_nan(self.robustness_loss_1_train) & self.needs_optimization_train
        self.robustness_loss_2_train = tf.boolean_mask(self.robustness_loss_1_train, self.mask_2_train)
        
        self.mask_2_u = ~tf.math.is_nan(self.robustness_loss_1_u) & ~tf.math.is_nan(self.robustness_loss_1_u) & self.needs_optimization_u
        self.robustness_loss_2_u = tf.boolean_mask(self.robustness_loss_1_u, self.mask_2_u)
        
        self.robustness_loss_train = tf.reduce_sum(self.robustness_loss_2_train + gamma_macer) * sigma_macer * 0.5
        self.robustness_loss_u = tf.reduce_sum(self.robustness_loss_2_u + gamma_macer) * sigma_macer * 0.5
        
        self.g = ((self.ce_loss_train_smoothed + self.ce_loss_u_smoothed) + lambda_macer * (self.robustness_loss_train + self.robustness_loss_u))/(Ntrain + Npoison)
        
        self.bl = bilevel_approxgrad(sess, self.f, self.g, self.u, self.var_train, self.sig)
    
    def augmentation(self, dataset, batch_len):
        #augmentation
        aug_1 = tf.image.pad_to_bounding_box(dataset, 4, 4, 32 + 8, 32 + 8)
        aug_2 = tf.image.random_crop(aug_1, [batch_len, 32, 32, 3])
        aug_3 = tf.image.random_flip_left_right(aug_2)
        
        return aug_3
    
    def reset_v(self):
        for var in self.var_train:
            self.sess.run(var.initializer)
    
    def train(self, x_train, y_train, noise_train, x_val, y_val, noise_val, x_poisoned, y_poisoned, noise_poison, lr_u, lr_v, lr_p, niter1, niter2):
        
        self.sess.run(self.assign_u,feed_dict={self.x_poison_tf:x_poisoned})
        feed_dict={self.x_train_tf:x_train, self.y_train_tf:y_train, self.x_val_tf:x_val, self.y_val_tf:y_val, self.y_val_class_tf:np.argmax(y_val, 1), self.y_train_class_tf:np.argmax(y_train, 1), self.u_class_tf:np.argmax(y_poisoned, 1), self.y_poison_tf:y_poisoned, self.noise_train:noise_train.reshape([self.Ntrain*self.k_macer, self.height, self.width, self.nch]), self.noise_u:noise_poison.reshape([self.Npoison*self.k_macer, self.height, self.width, self.nch]), self.noise_val:noise_val.reshape([self.val_set_size*self.k_rs, self.height, self.width, self.nch])}
        
        self.bl.update_v(feed_dict, lr_v, niter1)
        
        fval, gval, hval = self.bl.update_u(feed_dict, lr_u, lr_p, niter2)
        new_x_poisoned = self.sess.run(self.u)
        
        fval, gval, f3, f4 = self.sess.run([self.f, self.g, self.robustness_val_loss_1, self.robustness_val_loss], feed_dict)
        
        return [len(f3), f4, fval, gval, hval, new_x_poisoned]
        
    def eval_accuracy(self, x_test, y_test, add_noise):
        acc = 0
        batch_size = min(len(x_test), 100)
        nb_batches = int(len(x_test)/batch_size)
        if len(x_test) % batch_size != 0:
            nb_batches += 1
        for batch in range(nb_batches):
            ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
            
            if add_noise:
                noise = np.random.normal(0, 1, x_test[ind_batch].shape) * self.sigma_macer
                pred = self.sess.run(self.cls_test_noisy, {self.x_test_tf:x_test[ind_batch]+noise})
            else:
                pred = self.sess.run(self.cls_test_noisy, {self.x_test_tf:x_test[ind_batch]})
            acc += np.sum(np.argmax(pred,1)==np.argmax(y_test[ind_batch],1))
        
        acc /= np.float32(len(x_test))
        return acc
    
    def eval_accuracy_averaged(self, x_val, y_val, noise_val):
        
        acc = 0
        pred = self.sess.run(self.log_val_softmax, {self.x_val_tf:x_val, self.y_val_tf:y_val, self.noise_val:noise_val.reshape([self.val_set_size*self.k_rs, self.height, self.width, self.nch])})
        acc += np.sum(np.argmax(pred,1)==np.argmax(y_val,1))
        acc /= np.float32(len(x_val))
        return acc
    
