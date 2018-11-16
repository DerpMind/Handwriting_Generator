from keras import models, layers, initializers
from ngdlm import models as ngdlmodels
from ngdlm import utils as ngdlutils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K


def define_VAE_architecture(latent_dim = 10):
    ''' Defines a variational autoenconder with a 
        convolutional neural network architecture
        for our 128 x 128 images.
        
        Returns the model.
    '''
    
    ### Definition of NN architecture
    latent_dim = latent_dim

    # Encoder.
    encoder_input = layers.Input(shape=(128, 128,1))
    encoder_output = layers.Conv2D(32, (3, 3),
                                   kernel_initializer=initializers.lecun_normal(seed=0),
                                   activation="relu",
                                   padding="same")(encoder_input)
    encoder_output = layers.MaxPooling2D((2, 2),padding="same")(encoder_output)
    encoder_output = layers.Conv2D(32, (3, 3),
                                   kernel_initializer=initializers.lecun_normal(seed=0),
                                   activation="relu",
                                   padding="same")(encoder_output)
    encoder_output = layers.MaxPooling2D((2, 2),padding="same")(encoder_output)
    encoder_output = layers.Flatten()(encoder_output)
    encoder_output = layers.Dense(1024,
                                  activation = "relu",
                                  kernel_initializer=initializers.lecun_normal(seed=0)
                                 )(encoder_output)
    encoder_output = layers.Dense(64,
                                  activation = "relu",
                                  kernel_initializer=initializers.lecun_normal(seed=0)
                                 )(encoder_output)
    encoder = models.Model(encoder_input, encoder_output)

    # Decoder.
    decoder_input = layers.Input(shape=(latent_dim,))
    decoder_output = layers.Dense(64,
                                  activation="relu",
                                  kernel_initializer=initializers.lecun_normal(seed=0)
                                 )(decoder_input)
    decoder_output = layers.Dense(1024,
                                  activation="sigmoid",
                                  kernel_initializer=initializers.lecun_normal(seed=0)
                                 )(decoder_output)
    decoder_output = layers.Dense(32*32*32,
                                  activation="sigmoid",
                                  kernel_initializer=initializers.lecun_normal(seed=0)
                                 )(decoder_output)
    decoder_output = layers.Reshape((32, 32, 32))(decoder_output)
    decoder_output = layers.Conv2D(32, (3, 3),
                                   kernel_initializer=initializers.lecun_normal(seed=0),
                                   activation="relu",
                                   padding="same")(decoder_output)
    decoder_output = layers.UpSampling2D((2, 2))(decoder_output)
    decoder_output = layers.Conv2D(32, (3, 3),
                                   kernel_initializer=initializers.lecun_normal(seed=0),
                                   activation="relu",
                                   padding="same")(decoder_output)
    decoder_output = layers.UpSampling2D((2, 2))(decoder_output)
    decoder_output = layers.Conv2D(1, (3, 3),
                                   kernel_initializer=initializers.lecun_normal(seed=0),
                                   activation="sigmoid",
                                   padding="same")(decoder_output)
    decoder = models.Model(decoder_input, decoder_output)


    #autoencoder
    vae = ngdlmodels.VAE(encoder, decoder, latent_dim = latent_dim)
    
    return vae



def train_VAE_model(vae, train_data, test_data, max_loop=100, batch_size=4, verbose=1):
    ''' Trains a VAE model. Uses a loop in order to break at a certain condition.
        
        Returns the model.
    '''
    
    
    n=max_loop
    history = []
    for i in range(1, n+1):

        if (i%5 == 0):
            print("Epoch: " + str(i))
            print(df.iloc[-1,:])

        # append loss and val_loss
        # multiple epochs are obtained through the loop repetition
        history.append(
            vae.fit(
                train_data, train_data,
                epochs=1,            
                batch_size=batch_size,
                shuffle=True,
                validation_data=(test_data,test_data),
                verbose=verbose
            ).history
        )

        # Break if the training begins to overfit,
        df = pd.DataFrame(history)

        # i.e. if val_loss_i is the biggest  in [i-5, i]
        if ((i>5) & ((df.val_loss.iloc[-10:].apply(lambda x: x[0]).idxmin()) == (i-9))):
            break
    
    print("Break after " + str(df.shape[0]) + " epochs.")

    ### Visualize results
    df.applymap(lambda x: x[0]).plot()

    return vae



def reset_weights(model):
    ''' Resets the weights of the NN model.
    '''
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)












 
