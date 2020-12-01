import tensorflow as tf

def _hessian_vector_product(ys, xs, v):

    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = tf.gradients(ys, xs)

    assert len(grads) == length
    elemwise_products = tf.reduce_sum([
      tf.reduce_sum(tf.multiply(grad_elem, v_elem))
      for grad_elem, v_elem in zip(grads, v)
      if grad_elem is not None
    ])

    # Second backprop
    return tf.gradients(elemwise_products, xs)

def l2norm_sq(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs])


class bilevel_approxgrad(object):

    def __init__(self, sess, f, g, u, v, sig=1E-10):

        self.sess = sess
        self.f = f
        self.g = g
        
        self.v = v
        
        self.lr_u = tf.placeholder(tf.float32)
        self.lr_v = tf.placeholder(tf.float32)
        self.lr_p = tf.placeholder(tf.float32)
        
        self.p = [tf.get_variable('pvec'+str(i),shape=( self.v[i].shape),initializer=tf.zeros_initializer) for i in range(len(self.v))]
        self.sig = sig

        ## min_v g
        self.optim_v = tf.train.AdamOptimizer(self.lr_v)
        self.min_v = self.optim_v.minimize(self.g, var_list=self.v)
        
        self.dgdv = tf.gradients(self.g, self.v)
        self.gvnorm = l2norm_sq(self.dgdv)

        ## solve gvv*p = fv:   
        self.gvvp = _hessian_vector_product(self.g, self.v, self.p)
        self.fv = tf.gradients(self.f,self.v)

        self.h = tf.reduce_sum([tf.reduce_sum(tf.square(self.gvvp[i] + self.sig * self.p[i] - self.fv[i])) for i in range(len(self.v))])
        self.min_p = tf.train.AdamOptimizer(self.lr_p).minimize(self.h,var_list=self.p)
        
        #Computing the
        ## min_u  [f - gv*p]:  u <- u - [fu - guv*p]
        gv = tf.gradients(self.g,self.v)
        gvp = tf.reduce_sum([tf.reduce_sum(gv[i]*self.p[i]) for i in range(len(self.v))])
        f_ = self.f - gvp # df_du = fu - guv*p = fu - guv*inv(gvv)*fv
        self.min_u = tf.train.AdamOptimizer(self.lr_u).minimize(f_,var_list=u)
        
    def update_v(self, feed_dict, lr_v, niter1):
        ## min_v g
        feed_dict.update({self.lr_v: lr_v})
        for it in range(niter1):
            self.sess.run(self.min_v,feed_dict)
            
    def update_u(self, feed_dict, lr_u, lr_p, niter2):

        ## solve gvv*p = fv    
        feed_dict.update({self.lr_p:lr_p})
        for it in range(niter2):
            self.sess.run(self.min_p,feed_dict)
        
        ## min_u  [f - gv*p]:  u <- u - [fu - guv*p]
        feed_dict.update({self.lr_u:lr_u})
        self.sess.run(self.min_u,feed_dict)

        fval,gval,hval = self.sess.run([self.f,self.g,self.h],feed_dict)

        return [fval,gval,hval]