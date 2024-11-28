# %%
from dlgo.data.processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder

from dlgo.networks import small
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint  # <1>
# <1> With model checkpoints we can store progress for time-consuming experiments

# %%
if __name__ == '__main__':
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 100

    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))  # <1>

    processor = GoDataProcessor(encoder=encoder.name())  # <2>

    X, Y = processor.load_go_data('train', num_games)  # <3>
    print('X.shape', X.shape)
    print('Y.shape', Y.shape)

    validationX, validationY = processor.load_go_data('test', num_games)
    print('validationX.shape', validationX.shape)
    print('validationY.shape', validationY.shape)
    # <1> First we create an encoder of board size.
    # <2> Then we initialize a Go Data processor with it.
    # <3> From the processor we create two data generators, for training and testing.


# %%
    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    network_layers = small.layers(input_shape)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print(model.summary())

    epochs = 5
    batch_size = 128
    model.fit(X, Y, batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(validationX, validationY),
                        callbacks=[ModelCheckpoint('../checkpoints/small_model_epoch_{epoch}.keras')])  # <5>
    # <5> After each epoch we persist a checkpoint of the model.
    model.evaluate(validationX, validationY, batch_size=batch_size)  # <6>
    # <6> For evaluation we also specify the validation set.


