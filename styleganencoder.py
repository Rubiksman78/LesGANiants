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
from PIL import Image
import cv2

resolution = 64
def resize_image(image):
    return tf.image.resize(image,(resolution,resolution))

shape = (resolution,resolution,3)
buffer_size = 10
batch_size =1

x_train = (tf.keras.utils.image_dataset_from_directory("../Datasets/CeterisParibusDataset",labels=None,batch_size=batch_size)    
    .map(resize_image)
    .shuffle(buffer_size)
)

def resnet_block(input,filters,kernel,stride):
    x1 = kl.Con2D(filters,kernel,stride=stride,padding='same')(input)
    x2 = kl.Conv2D(filters,1,padding='same')(input)
    output = kl.Add()([x1,x2])
    return output

##Modifier l'architecture ResNet voir EfficientNet et se placer directement dans le W-space
res_net = tf.keras.applications.ResNet50V2(
    input_shape = shape,
    include_top = False,
)

def define_encoder(im_shape):
    input = models.Input(shape=im_shape)
    x = res_net(input)
    x = kl.Flatten()(x)
    x = [kl.Dense(512)(x) for _ in range(18)]
    out = tf.stack(x,axis=1)
    return models.Model(input,out)

model = define_encoder(shape)
model.summary()
#%%
dlatent_vector = (int(np.log2(resolution))-1)*2
weights_name = 'ffhq' 
generator = StyleGan2Generator(resolution=resolution,weights=weights_name, impl='ref')
decoder = generator
disc = StyleGan2Discriminator(resolution=resolution,weights=weights_name, impl='ref')

opt = keras.optimizers.Adam(1e-4)

epochs = 10
num_examples_to_generate = 2
im_shape = (resolution,resolution,3)
seed = tf.random.normal(shape=[num_examples_to_generate,512])
model = define_encoder(im_shape)

def generate_images(model,test_sample):
    z = model(test_sample)
    predictions = decoder(z)
    fig = plt.figure(figsize=(15,15))
    for i in range(predictions.shape[0]):
        grid_row = min(predictions.shape[0], 2)
        f, axarr = plt.subplots(grid_row, 2, figsize=(18, grid_row * 6))
        for row in range(grid_row):
            im = tf.transpose(predictions[i]).numpy()
            im = im.swapaxes(-3,-2)[...,::]
            ax = axarr
            ax[0].imshow(im*0.5+0.5)
            ax[0].axis("off")
            ax[0].set_title("Reconstructed", fontsize=20)
            ax[1].imshow(test_sample[0]/255)
            ax[1].axis("off")
            ax[1].set_title("Origin", fontsize=20)
    plt.show()

for test_batch in x_train.take(1):
    test_sample = test_batch[0:5,:,:,:]
generate_images(model,test_sample)

class VGGFeatureMatchingLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        #self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0]
        vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
        layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
        self.vgg_model = keras.Model(vgg.input, layer_outputs, name="VGG")
        self.mae = keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        y_true = keras.applications.vgg19.preprocess_input(127.5 * (y_true + 1))
        y_pred = keras.applications.vgg19.preprocess_input(127.5 * (y_pred + 1))
        real_features = self.vgg_model(y_true)
        fake_features = self.vgg_model(y_pred)
        loss = 0
        for i in range(len(real_features)):
            loss += self.weights[i] * self.mae(real_features[i], fake_features[i])
        return loss

def train_step(model,x,opt):
    with tf.GradientTape() as tape:
        z = model(x)
        xp = decoder(z)
        y = disc(xp)
        xp = tf.reshape(xp,(tf.shape(xp)[0],tf.shape(xp)[3],tf.shape(xp)[2],tf.shape(xp)[1]))
        xp = tf.image.resize(xp,(64,64))
        adloss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y,tf.ones_like(y))
        vgg_loss = VGGFeatureMatchingLoss()(x,xp)
        loss = tf.reduce_mean(keras.losses.MSE(x,xp)) + adloss + vgg_loss
    gradients = tape.gradient(loss,model.trainable_variables)
    opt.apply_gradients(zip(gradients,model.trainable_variables))
    return loss

for epoch in range(1,epochs):
    start_time = time.time()
    for batch in x_train:
        loss = train_step(model,batch,opt)
    end_time = time.time()
    display.clear_output(wait=False)
    generate_images(model,test_sample)
    print(f'Epoch {epoch}, Loss : {loss}')

model.encoder.save_weights('encoder_weights.h5')