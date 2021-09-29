
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import DepthwiseConv2D
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import  Reshape
from tensorflow.keras.layers.experimental.preprocessing import  Resizing



def get_DeepLab(input_img, classes=2):

    # ---------- Block 1.1 ----------
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False,padding="same")(input_img)  
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), use_bias=False, activation="relu")(x) 
    x = BatchNormalization()(x)


    # xception block 
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      use_bias=False, activation="relu", padding="same")(x)
    residual = BatchNormalization()(residual)

    firstResidual = residual


    x = SeparableConv2D(128, (3, 3), use_bias=False, padding="same")(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(128, (3, 3), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((3,3), strides=(2, 2), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), padding='same',use_bias=False,activation="relu")(x)
    x = BatchNormalization()(x)

    x = Add()([x, residual])

     # ---------- Block 1.2 ----------

    residual = Conv2D(256, (1, 1), strides=(2, 2), use_bias=False, activation="relu", padding="same")(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(256, (3, 3), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(256, (3, 3), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((3, 3), use_bias=False, strides=(2, 2), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding='same',use_bias=False, activation="relu")(x)
    x = BatchNormalization()(x)


    x = Add()([x, residual])


    # ---------- Block 1.3 ----------


    residual = Conv2D(728, (1, 1), strides=(2, 2), use_bias=False, activation="relu",padding="same")(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(728, (3, 3), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(728, (3, 3), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((3, 3), (2, 2), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(728, (1, 1), padding='same',use_bias=False, activation="relu")(x)
    x = BatchNormalization()(x)


    x = Add()([x, residual])



    # ---------- Block 2 ----------
    # TO DO: apperentely the the rate must be set to 2 
    for _ in range(16):
        residual = x
        for _ in range(3):
            x = SeparableConv2D(728, (3, 3), use_bias=False, activation="relu",padding="same",dilation_rate=(2,2))(x)
            x = BatchNormalization()(x)

        x = Add()([x, residual])

   
    residual = Conv2D(filters=1024, kernel_size=(1, 1), strides=(2, 2),
                      use_bias=False, activation="relu")(x)
    residual = BatchNormalization()(residual)

    # ---------- Block 3.1 ----------
  

    x = SeparableConv2D(728, (3, 3), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(1024, (3, 3), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), padding='same',use_bias=False, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Add()([x, residual])

  
    # ---------- Block 3.2 ----------


    x = SeparableConv2D(1536, (3, 3), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((3, 3), use_bias=False, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(1536, (1, 1), padding='same',use_bias=False, activation="relu")(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(filters=2048, kernel_size=(3, 3), use_bias=False, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)

    # ---------- atrous Branching  ----------
    # (6, 12, 18)

    b0 = Conv2D(256,(1,1),padding="same", use_bias=False, activation="relu")(x)
    b0 = BatchNormalization()(b0)

    b1 = DepthwiseConv2D((3,3),activation="relu", padding="same",dilation_rate=(6,6))(x)
    b1 = BatchNormalization()(b1)
    b1 = Conv2D(256, (1, 1), padding='same',use_bias=False, activation="relu")(b1)
    b1 = BatchNormalization()(b1)


    b2 = DepthwiseConv2D((3,3),activation="relu", padding="same",dilation_rate=(12,12))(x)
    b2 = BatchNormalization()(b2)
    b2 = Conv2D(256, (1, 1), padding='same',use_bias=False, activation="relu")(b2)
    b2 = BatchNormalization()(b2)

    b3 = DepthwiseConv2D((3,3),activation="relu", padding="same",dilation_rate=(18,18))(x)
    b3 = BatchNormalization()(b3)
    b3 = Conv2D(256, (1, 1), padding='same',use_bias=False, activation="relu")(b3)
    b3 = BatchNormalization()(b3)


    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    b4 = Reshape((1,1, b4_shape[1]))(b4)
    b4 = Conv2D(256,(1,1),padding="same",use_bias=False,activation="relu")(b4)
    b4 = BatchNormalization()(b4)
    size_before = tf.keras.backend.int_shape(x)
    b4 = Resizing(*size_before[1:3])(b4)

    x = Concatenate()([b4,b0,b1,b2,b3])
    x = Conv2D(256,(1,1), padding="same",use_bias=False, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)


    skip_size = tf.keras.backend.int_shape(firstResidual)
    x = Resizing(*skip_size[1:3],interpolation="bilinear")(x)

    dec_skip1 = Conv2D(48,(1,1), padding="same",use_bias=False, activation="relu")(firstResidual)
    dec_skip1 = BatchNormalization()(dec_skip1)

    x = Concatenate()([x, dec_skip1])

    x = DepthwiseConv2D((3,3),activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding='same',use_bias=False, activation="relu")(x)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((3,3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), padding='same',use_bias=False, activation="relu")(x)
    x = BatchNormalization()(x)


    x = Conv2D(classes,(1,1),padding="same")(x)
    size_before = int_shape(input_img)
    x = Resizing(*size_before[1:3])(x)
    x = tf.keras.layers.Activation("sigmoid")(x)


    model = Model(inputs=[input_img], outputs=[x],)
    return model



if __name__ == "__main__":
    in_size = Input((512, 512, 3))
    model = get_DeepLab(in_size)
    model.summary()
