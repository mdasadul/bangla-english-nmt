from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 

def BaseModel(object):
	def __init__(self, hparams,mode, iterator, source_vocabulary_table, target_vocabulary_table):

		self.mode = mode
		self.iterator = iterator
		self.rnn_size = hparams.rnn_size
		self.dropout = hparams.dropout
		self.random_seed = hparams.random_seed
		self.source_seuence_length = hparams.source_seuence_length
		self.target_sequence_length = hparams.target_sequence_length
		self.num_layers = hparams.num_layers
		self.source_vocabulary_size = hparams.source_vocabulary_size
		self.target_vocabulary_size =hparams.target_vocabulary_size
		self.source_vocabulary_table = source_vocabulary_table
		self.target_vocabulary_table = target_vocabulary_table

		self.embedding_encoder = _create_embedding("encoder_embedding",
													self.source_vocabulary_size,
													self.rnn_size)
		self.embedding_decoder = _create_embedding("decoder_embedding",
													self.target_vocabulary_size,
													self.rnn_size)
		

		return_loss = _build_graph(hparams)


	def _build_graph(self, hparams):
		enc_output, enc_state = encoder_cell(
											rnn_size = hparams.rnn_size,
											num_layers = hparams.num_layers,
											sequence_length= hparams.sequence_length,
											dropout = hparams.dropout)
		

	def _create_embedding(self, embed_name,vocab_size,embed_size,dtype=tf.float32):
		"""create a new embedding matrix"""
		embedding = tf.get_variable(embed_name,[vocab_size,embed_size],dtype)
		return embedding

	#encoder
	def encoder_cell(self, rnn_size, num_layers,sequence_length, dropout)
		source = self.iterator.source
		embedded_input = tf.nn.embedding_lookup(self.embedding_encoder, source)
		
		keep_prob = 1-dropout
		# foward 
		cell_fw = tf.nn.rnn_cell.MultiRNNCell([_make_lstm(rnn_size, keep_prob) for _ in range(num_layers)])
		# backward    
		cell_bw = tf.nn.rnn_cell.MultiRNNCell([make_lstm(rnn_size, keep_prob) for _ in range(num_layers)])
		enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
															cell_bw,
															embedded_input,
															sequence_length=sequence_length,
															dtype=tf.float32,
															)
		enc_output = tf.concat(enc_output,-1)
		return enc_output, enc_state

	def decoder_cell(self,enc_outputs, enc_state
					 ):
		beam_width = 10
		vocab_size = self.target_vocabulary_size
		rnn_size = self.rnn_size
		num_layers = self.num_layers
		keep_prob = 1 - self.dropout 

		dec_cell = tf.nn.rnn_cell.MultiRNNCell([_make_lstm(rnn_size, keep_prob) for _ in range(num_layers)])
		output_layer = Dense(vocab_size, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
		
		target_input = self.iterator.target_input
		embedded_target_input = tf.nn.embedding_lookup(self.embedding_decoder, target_input)
		
		target_sequence_length = self.iterator.target_sequence_length
		dec_cell = attention_cell(dec_cell, rnn_size, enc_outputs, target_sequence_length)


		with tf.variable_scope("decode"):
    		# (dec_embed_input comes from another function but should not be 
    		#   relevant in this context. )
    		helper = tf.contrib.seq2seq.TrainingHelper(inputs = embedded_target_input, 
                                               sequence_length = summary_length,
                                               time_major = False)

    		decoder = tf.contrib.seq2seq.BasicDecoder(cell = dec_cell,
                                              helper = helper,
                                              initial_state = dec_cell.zero_state(batch_size, tf.float32),
                                              output_layer = output_layer)

    	logits = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, 
                                           output_time_major=False, 
                                           impute_finished=True, 
                                           maximum_iterations=max_summary_length)




enc_output = tf.contrib.seq2seq.tile_batch(enc_output, multiplier=beam_width)
enc_state = tf.contrib.seq2seq.tile_batch(enc_state, multiplier=beam_width)
text_length = tf.contrib.seq2seq.tile_batch(text_length, multiplier=beam_width)

dec_cell = tf.nn.rnn_cell.MultiRNNCell([make_lstm(rnn_size, keep_prob) for _ in range(num_layers)])
dec_cell = decoder_cell(dec_cell, rnn_size, enc_output, text_length)

start_tokens = tf.tile(tf.constant([word2ind['<GO>']], dtype = tf.int32), [batch_size], name = 'start_tokens')

with tf.variable_scope("decode", reuse = True):


    decoder = tf.contrib.seq2seq.BeamSearchDecoder( cell=dec_cell,
                                                    embedding=embeddings,
                                                    start_tokens=start_tokens,
                                                    end_token=end_token,
                                                    initial_state=dec_cell.zero_state(batch_size = batch_size*beam_width , dtype = tf.float32),
                                                    beam_width=beam_width,
                                                    output_layer=output_layer,
                                                    length_penalty_weight=0.0)



    logits = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, 
                                           output_time_major=False, 
                                           impute_finished=True, 
                                           maximum_iterations=max_summary_length)

	# helper to create the attention cell with
	def attention_cell(dec_cell, rnn_size, enc_output,  lengths):
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



	# helper to create the layers
	def _make_lstm(rnn_size, keep_prob,seed):
		lstm = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=seed))
		#TODO discard dropout if model is for eval or inference
		lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob = keep_prob)
		return lstm_dropout

	
	