#%%
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from keras import models
import keras.layers as kl
import matplotlib.pyplot as plt
import time
from IPython import display
from stylegan2_generator import StyleGan2Generator
#%%
def resize_image(image):
    print(image)
    return tf.image.resize(image,(64,64))

buffer_size = 100
batch_size =2
x_train = (tf.keras.utils.image_dataset_from_directory("C:/SAMUEL/Centrale/Automatants/anime_face/",labels=None)
    .map(resize_image)
    .shuffle(buffer_size)
)
# %%
def define_encoder(im_shape,latent_dim):
    input = models.Input(shape=im_shape)
    x = kl.Conv2D(32,3,2,activation='relu',kernel_initializer='random_normal')(input)
    x = kl.Conv2D(64,3,2,activation='relu',kernel_initializer='random_normal')(x)
    x = kl.Conv2D(128,3,2,activation='relu',kernel_initializer='random_normal')(x)
    x = kl.Conv2D(256,3,2,activation='relu',kernel_initializer='random_normal')(x)
    x = kl.Flatten()(x)
    x1 = kl.Dense(latent_dim)(x)
    x2 = kl.Dense(latent_dim)(x)
    x = kl.Dense(latent_dim + latent_dim)(x)
    return models.Model(input,x)

decoder = StyleGan2Generator(resolution=64,impl='ref')

class VAE(models.Model):
    def __init__(self,latent_dim,im_shape,decoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = define_encoder(im_shape,latent_dim)
        self.decoder = decoder

    def sample(self,eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100,self.latent_dim))
        return self.decode(eps,apply_sigmoid=True)

    def encode(self,x):
        mean,logvar=tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean,logvar

    def reparametrize(self,mean,logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar*0.5)+mean

    def decode(self,z,apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
# %%
opt = keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample,mean,logvar,raxis=1):
    log2pi = tf.math.log(2.*np.pi)
    res1 = tf.reduce_sum(-.5*((sample-mean)**2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)
    res2 = 0.5 * tf.reduce_sum(tf.exp(logvar) - logvar - 1 + mean**2)
    return res1

def compute_loss(model,x):
    mean,logvar = model.encode(x)
    z = model.reparametrize(mean,logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit,labels=x)
    logpx_z = -tf.reduce_sum(cross_ent,axis=[1,2,3])
    logpz = log_normal_pdf(z,0.,0.)
    logqz_x = log_normal_pdf(z,mean,logvar)
    return - tf.reduce_mean(logpx_z + logpz - logqz_x)

def train_step(model,x,opt):
    with tf.GradientTape() as tape:
        loss = compute_loss(model,x)
    gradients = tape.gradient(loss,model.trainable_variables)
    opt.apply_gradients(zip(gradients,model.trainable_variables))
    return loss
# %%
epochs = 10
latent_dim = 128
num_examples_to_generate = 2
im_shape = (64,64,3)
seed = tf.random.normal(shape=[num_examples_to_generate,latent_dim])
model = VAE(latent_dim,im_shape,decoder)
# %%
def generate_images(model,epoch,test_sample):
    mean,logvar = model.encode(test_sample)
    z = model.reparametrize(mean,logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(10,10))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(tf.transpose(predictions[i])*0.5+0.5)
        plt.axis('off')
    plt.show()

assert batch_size >= num_examples_to_generate
for test_batch in x_train.take(1):
    test_sample = test_batch[0:2,:,:,:]
generate_images(model,0,test_sample)
# %%
for epoch in range(1,epochs+1+30):
    start_time = time.time()
    for batch in x_train:
        loss = train_step(model,batch,opt)
    end_time = time.time()
    display.clear_output(wait=False)
    generate_images(model,epoch,test_sample)
    print(f'Epoch {epoch}, Loss : {loss}')
# %%
model.encoder.save_weights('encoder_weights.h5')