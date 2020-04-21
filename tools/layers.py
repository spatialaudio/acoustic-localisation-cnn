import keras
import tensorflow as tf

class SplitLayer(keras.layers.Layer):

    def call(self, input):
        outputs = tf.split(input, num_or_size_splits=input.shape[2], axis=2)
        return list(outputs)

    def compute_output_shape(self, input_shape):
        output_dim = input_shape[0:2] + (1,)
        return [output_dim]*input_shape[2]

class Corr1D(keras.layers.Layer):
    def __init__(self, 
                 usefft=False, 
                 max_displacement=None,
                 normalise=True,
                 **kwargs):
      
        super(Corr1D, self).__init__(**kwargs)
        self._usefft = usefft
        if usefft:
            self._conv = self._convfd
        else:
            self._conv = self._convtd
        self._max_displacement = max_displacement
        self._normalise = normalise

    def build(self, input_shape): 
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        assert input_shape[0] == input_shape[1]

        if self._max_displacement is None:
            self._md = input_shape[0][1] - 1
        else:
            assert self._max_displacement < input_shape[0][1]
            self._md = self._max_displacement

        super(Corr1D, self).build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        in1, in2 = inputs
        if self._normalise:
            in1 = tf.math.l2_normalize(in1, axis=1)
            in2 = tf.math.l2_normalize(in2, axis=1)

        return self._conv(in1, in2)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0][0], 2 * self._md + 1, input_shape[0][2]

    def get_config(self):
        config = super(Corr1D, self).get_config()
        config.update({'usefft': self._usefft,
                       'max_displacement': self._max_displacement,
                       'normalise': self._normalise
                      })
        return config

    def _convfd(self, in1, in2):
        in1 = tf.transpose(in1, perm=[0, 2, 1])
        in2 = tf.transpose(in2, perm=[0, 2, 1])

        # Extract shapes
        s1 = tf.convert_to_tensor(tf.shape(in1)[2:], dtype=tf.int32)
        nfft = 2 * s1 - 1

        in1_fft = tf.signal.rfft(in1, fft_length=nfft)
        in2_fft = tf.signal.rfft(in2[:,:,::-1], fft_length=nfft)
        out = tf.signal.irfft(in1_fft * in2_fft, fft_length=nfft)

        # truncate output
        tdx = in1.shape.as_list()[2] - 1 - self._md
        out_trunc = out[:,:, tdx:tdx+2*self._md+1]

        # Reorder channels to last
        result = tf.transpose(out_trunc, perm=[0, 2, 1])
        return result

    def _convtd(self, in1, in2):
        paddings = tf.constant([[0, 0], [self._md, self._md], [0, 0]])
        in1 = tf.pad(in1, paddings, "CONSTANT", constant_values=0)
        result = tf.map_fn(
            lambda x: tf.nn.depthwise_conv2d(  # convolution is actually a correlation
                tf.expand_dims(tf.expand_dims(x[0], 0), 0),  # W,C -> 1,1,W,C
                tf.expand_dims(tf.expand_dims(x[1], 0), -1),  # W,C -> 1,W,C,1
                strides=[1, 1, 1, 1],
                padding='VALID'
            ),  # Result of conv is 1,1,W,C
            elems=[in1, in2],
            dtype=tf.float32,
            parallel_iterations=128
        )

        return result[:, 0, 0, :, :]