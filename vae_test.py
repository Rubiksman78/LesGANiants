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
from stylegan2_discriminator import StyleGan2Discriminator
#%%
resolution = 1024
def resize_image(image):
    return tf.image.resize(image,(resolution,resolution))

buffer_size = 100
batch_size =18
x_train = (tf.keras.utils.image_dataset_from_directory("C:/SAMUEL/Centrale/Automatants/anime_face/",labels=None,batch_size=batch_size)    
    .map(resize_image)
    .shuffle(buffer_size)
)
# %%
def define_encoder(im_shape):
    input = models.Input(shape=im_shape)
    x = kl.Conv2D(32,3,2,activation='relu',kernel_initializer='random_normal')(input)
    x = kl.Conv2D(64,3,2,activation='relu',kernel_initializer='random_normal')(x)
    x = kl.Conv2D(128,3,2,activation='relu',kernel_initializer='random_normal')(x)
    #x = kl.Conv2D(256,3,2,activation='relu',kernel_initializer='random_normal')(x)
    x = kl.Flatten()(x)
    x = kl.Dense(dlatent_vector*512)(x)
    x = kl.Reshape((dlatent_vector,512))(x)
    return models.Model(input,x)

dlatent_vector = (int(np.log2(resolution))-1)*2
weights_name = 'ffhq' 
generator = StyleGan2Generator(weights=weights_name, impl='ref')
decoder = generator.synthesis_network
disc = StyleGan2Discriminator(impl='ref')
# %%
opt = keras.optimizers.Adam(1e-4)

def train_step(model,x,opt):
    with tf.GradientTape() as tape:
        z = model(x)
        xp = decoder(z)
        y = disc(xp)
        adloss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y,tf.ones_like(y))
        xp = tf.reshape(xp,(tf.shape(xp)[0],tf.shape(xp)[3],tf.shape(xp)[2],tf.shape(xp)[1]))
        mseloss = keras.losses.MSE(x,xp)
        loss = adloss + mseloss
    gradients = tape.gradient(loss,model.trainable_variables)
    opt.apply_gradients(zip(gradients,model.trainable_variables))
    return loss
# %%
epochs = 10
num_examples_to_generate = 2
im_shape = (resolution,resolution,3)
seed = tf.random.normal(shape=[num_examples_to_generate,512])
model = define_encoder(im_shape)
# %%
def generate_images(model,test_sample):
    z = model(test_sample)
    predictions = decoder(z)
    fig = plt.figure(figsize=(10,10))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(tf.transpose(predictions[i])*0.5+0.5)
        plt.axis('off')
    plt.show()

for test_batch in x_train.take(1):
    test_sample = test_batch[0:1,:,:,:]
generate_images(model,test_sample)
# %%
for epoch in range(1,epochs+1+30):
    start_time = time.time()
    for batch in x_train:
        loss = train_step(model,batch,opt)
    end_time = time.time()
    display.clear_output(wait=False)
    generate_images(model,test_sample)
    print(f'Epoch {epoch}, Loss : {loss}')
# %%
model.encoder.save_weights('encoder_weights.h5')