from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# A bunch of utility functions
import imageio

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.
    
    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU
    
    Returns:
    TensorFlow Tensor with the same shape as x
    """
    return tf.maximum(x,alpha*x)
    
def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.
    
    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image
    
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    # TODO: compute D_loss and G_loss
    labels_real = tf.ones_like(logits_real)
    labels_fake_dis = tf.zeros_like(logits_fake)
    labels_fake_gen = tf.ones_like(logits_fake)
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels_fake_gen,logits = logits_fake))
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels_fake_dis,logits = logits_fake))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels_real,logits = logits_real))
    return D_loss, G_loss


def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    
    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)
    
    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    """
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
    return D_solver, G_solver


def lsgan_loss(score_real, score_fake):
    """Compute the Least Squares GAN loss.
    
    Inputs:
    - score_real: Tensor, shape [batch_size, 1], output of discriminator
        score for each real image
    - score_fake: Tensor, shape[batch_size, 1], output of discriminator
        score for each fake image    
          
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    # TODO: compute D_loss and G_loss
    D_loss = tf.reduce_mean(tf.pow(score_real-1,2))/2 + tf.reduce_mean(tf.pow(score_fake,2))/2
    G_loss = tf.reduce_mean(tf.pow(score_fake-1,2))/2
    return D_loss, G_loss


from cs231n.data_utils import load_CIFAR10

def sample_noise_cifar(batch_size, dim):
    """Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    ret = tf.random_normal([batch_size, dim])
    return ret


mean_image = None
std_image = None

def get_CIFAR10_data():
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    global mean_image
    global std_image
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, _, X_test, _ = load_CIFAR10(cifar10_dir)
    X = np.concatenate((X_train, X_test), axis =0)
    # Normalize the data: subtract the mean image
#    mean_image = np.mean(X, axis=0)
#    std_image = np.std(X, axis=0)
#    X -= mean_image
#    X /= std_image
    return X


def process_image(X):
	X -= 128
	X /= 128
	return X

# Invoke the above function to get our data.
X = get_CIFAR10_data()
print('Train data shape: ', X.shape)

def generator(z):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 32, 32, 3].
    """
    with tf.variable_scope("generator"):
        img = tf.layers.dense(z,4*4*512,activation=tf.nn.relu,use_bias=True)
        img = tf.layers.batch_normalization(img,axis=1,training=True)
        img = tf.reshape(img,[-1,4,4,512])
        print(img.shape)
        img = tf.layers.conv2d_transpose(img,256,kernel_size=5,strides=2,activation=tf.nn.relu,padding='same')
        print(img.shape)
        img = tf.layers.batch_normalization(img,3,training=True)
        img = tf.layers.conv2d_transpose(img,128,kernel_size=5,strides=2,activation=tf.nn.relu,padding='same')
        print(img.shape)
        img = tf.layers.batch_normalization(img,3,training=True)
        img = tf.layers.conv2d_transpose(img,3,kernel_size=5,strides=2,activation=tf.nn.tanh,padding='same')
        print(img.shape)
        return img
    

def discriminator(x):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of input images, shape [batch_size, 32, ,32, 3]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        logits = tf.layers.dropout(x,rate=0.2)
        logits = leaky_relu(tf.layers.conv2d(logits,96,3,padding='same'),0.01)
        logits = leaky_relu(tf.layers.conv2d(logits,96,3,padding='same'),0.01)
        logits = leaky_relu(tf.layers.conv2d(logits,96,3,strides=2,padding='same'),0.01)
        logits = tf.layers.dropout(logits,rate=0.5)
        logits = leaky_relu(tf.layers.conv2d(logits,192,3,padding='same'),0.01)
        logits = leaky_relu(tf.layers.conv2d(logits,192,3,padding='same'),0.01)
        logits = leaky_relu(tf.layers.conv2d(logits,192,3,strides=2,padding='same'),0.01)
        logits = tf.layers.dropout(logits,rate=0.5)
        logits = leaky_relu(tf.layers.conv2d(logits,192,3,strides=2,padding='valid',),0.01)
        logits = leaky_relu(tf.layers.conv2d(logits,192,1),0.01)
        logits = leaky_relu(tf.layers.conv2d(logits,192,1),0.01)
        logits = tf.layers.average_pooling2d(logits,pool_size=[logits.shape[1],logits.shape[2]],strides=1)
        logits = tf.reshape(logits,[-1,192])
        logits = tf.layers.dense(logits,1)
        print(logits.shape)
        return logits

tf.reset_default_graph()

batch_size = 128
# our noise dimension
noise_dim = 96

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, 32,32,3])
z = sample_noise_cifar(batch_size, noise_dim)
# generated images
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(process_image(x))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator') 

D_solver,G_solver = get_solvers()
D_loss, G_loss = gan_loss(logits_real, logits_fake)
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'generator')

def show_images_cifar(images,iter):
    global mean_image
    global std_image
    #images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    #sqrtimg = int(np.ceil(np.sqrt(images.shape[1])/3))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
#        img *= std_image
#        img += mean_image
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    plt.close()
    fig.savefig('progress/temp'+str(int(iter))+'.png', dpi=fig.dpi)
    return

# a giant helper function
def run_cifar_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, show_every=250, print_every=50, batch_size=128, num_epoch=10):
    """Train a GAN for a certain number of epochs.
    
    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    Returns:
        Nothing
    """
    # compute the number of iterations we need
    max_iter = int(X.shape[0]*num_epoch/batch_size)
    for it in range(max_iter):
        # every show often, show a sample result
        if it % show_every == 0:
            samples = sess.run(G_sample)
            fig = show_images_cifar(samples[:16],it/show_every)
            plt.show()
        # run a batch of data through the network
        minibatch = X[np.random.randint(0,X.shape[0],batch_size)]
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
        _, G_loss_curr = sess.run([G_train_step, G_loss])

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        #if it % print_every == 0:
        print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    print('Final images')
    samples = sess.run(G_sample)

    fig = show_images_cifar(samples[:16],it/show_every)
    plt.show()




with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_cifar_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step,num_epoch=4000)
