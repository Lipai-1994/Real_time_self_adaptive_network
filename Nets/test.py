import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.keras.backend import set_session
from keras import backend as K
import sharedLayers

def efficient_attention(input_, in_channels, key_channels, head_count, value_channels):
	n, c, h, w = input_.get_shape().as_list()
	input_ = tf.transpose(input_, perm=[0,2,3,1])

	ratio = np.sqrt(n/c)
	rand = tf.compat.v1.keras.initializers.RandomUniform(minval=-ratio, maxval=ratio, seed=None)
	keys = tf.keras.layers.Conv2D(key_channels, 1, 
		kernel_initializer=rand, bias_initializer=rand)(input_)
	queries = tf.keras.layers.Conv2D(key_channels, 1, 
		kernel_initializer=rand, bias_initializer=rand)(input_)
	values = tf.keras.layers.Conv2D(value_channels, 1, 
		kernel_initializer=rand, bias_initializer=rand)(input_)

	# keys = sharedLayers.conv2D(input_[1, 1, c, key_channels])
	# queries = sharedLayers.conv2D(input_[1, 1, c, key_channels])
	# keys = sharedLayers.conv2D(input_[1, 1, c, value_channels])
	
	keys = tf.reshape(keys, [n, key_channels, h * w])
	queries = tf.reshape(queries, [n, key_channels, h * w])
	values = tf.reshape(values, [n, value_channels, h * w])

	head_key_channels = key_channels // head_count
	head_value_channels = value_channels // head_count

	attended_values = []
	for i in range(head_count):
	    key = tf.nn.softmax(keys[:,i * head_key_channels: (i + 1) * head_key_channels, :], axis=2)
	    query = tf.nn.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels,:], axis=1)
	    value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
	        
	    context = key @ tf.transpose(value, perm=[0, 2, 1])
	    attended_value = tf.reshape(
	        tf.transpose(context, perm=[0, 2, 1]) @ query,
	    [n, head_value_channels, h, w])
	    attended_values.append(attended_value)

	aggregated_values = tf.concat(attended_values, 1)
	aggregated_values = tf.transpose(aggregated_values, perm=[0,2,3,1])
	reprojected_value = tf.keras.layers.Conv2D(in_channels, 1, 
		kernel_initializer=rand, bias_initializer=rand)(aggregated_values)
	
	reprojected_value = tf.transpose(reprojected_value, perm=[0,3,1,2])
	input_ = tf.transpose(input_, perm=[0,3,1,2])
	attention = reprojected_value + input_
	return attention

if __name__ == "__main__":
	x = tf.constant([[[[1, 5], [8, 10]]]], dtype=tf.float32)
	t = efficient_attention(x, 1, 2, 2, 4)
	sess = tf.keras.backend.get_session()
	print(t.get_shape())
	print(sess.run(t))

