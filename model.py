from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 

def BaseModel(object):
	def __init__(self, hparams,mode, iterator):

		self.mode = mode
		self.iterator = iterator
		self.rnn_size = hparams.rnn_size
		self.dropout = hparams.dropout
		self.random_seed = hparams.random_seed
		self.source_seuence_length = hparams.source_seuence_length
		self.target_sequence_length = hparams.target_sequence_length
		self.num_layers = hparams.num_layers

		
		return_loss = _build_graph()


	def _build_graph()
	# helper to create the layers
def make_lstm(rnn_size, keep_prob):
lstm = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=2))
lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob = keep_prob)
return lstm_dropout

# helper to create the attention cell with
def decoder_cell(dec_cell, rnn_size, enc_output,  lengths):
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units              = rnn_size,
            memory                 = enc_output,
            memory_sequence_length = lengths,
            normalize                  = True,
            name  = 'BahdanauAttention')

    return  tf.contrib.seq2seq.AttentionWrapper(
            cell                 = dec_cell,
            attention_mechanism  = attention_mechanism,
            attention_layer_size = rnn_size)

# foward 
cell_fw = tf.nn.rnn_cell.MultiRNNCell([make_lstm(rnn_size, keep_prob) for _ in range(n_layers)])



# backward    
cell_bw = tf.nn.rnn_cell.MultiRNNCell([make_lstm(rnn_size, keep_prob) for _ in range(n_layers)])



enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                        cell_bw,
                                                        rnn_inputs,
                                                        sequence_length=sequence_length,
                                                        dtype=tf.float32,
                                                        )

enc_output = tf.concat(enc_output,-1)