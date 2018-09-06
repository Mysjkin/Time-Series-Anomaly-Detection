# Model architecture taken from: https://blog.keras.io/building-autoencoders-in-keras.html 


from keras.layers import Input, LSTM, RepeatVector, Dense 
from keras.models import Model 
from keras import optimizers

def get_model(input_dim, latent_dim, segment_size, optimizer, learning_rate=1e-5, loss='mse'):
    """ Creates an lstm-based autoencoder.

    Args:
        :param input_dim:
        :param latent_dim:
        :param segment_size:
        :param optimizer:
        :param learning_rate:
        :param loss:

    Returns: a keras model.
    """
    # Define input tensor.         
    inputs = Input(shape=(segment_size, input_dim))         
    # Define encoder with latent_dim encoding capacity.         
    encoder = LSTM(latent_dim, return_state=True)         
    # Run encoder on input tensor.         
    _, encoded, cell_state = encoder(inputs)         
    # Output from the encoder is a context vector which encodes         
    # information about a segment of data points.         
    # In this model the context vector is used as input to the         
    # decoder.         
    # Additionally, this model uses the encoders final states         
    # to initialize the decoder's initial states.        
    # Repeat encoded information segment_size times such that all         
    # decoder LSTM units receives it as input.         
    repeat = RepeatVector(segment_size)         
    # Repeat encoded information.         
    decoder_input = repeat(encoded)         
    # Run decoder with context vector as input and encoders final states.         
    decoded = LSTM(latent_dim, return_sequences=True)(decoder_input, initial_state=[encoded, cell_state])         
    # Convert the dimension of decoded values to the input dimension.         
    decoded_dense = Dense(input_dim)(decoded)         
    # Define model input and output
    model = Model(inputs, decoded_dense)
    optimizer.lr = learning_rate
    model.compile(optimizer = optimizer, loss = loss)         
    return model

