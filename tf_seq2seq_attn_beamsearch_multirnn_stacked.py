import numpy as np
import tensorflow as tf
import os, math
import RawTaggerReader2 as reader
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell

from tensorflow.python.ops.rnn_cell import LSTMCell

from tensorflow.python.ops.rnn_cell import MultiRNNCell

from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

from tensorflow.python.ops import array_ops

from tensorflow.python.ops import control_flow_ops

from tensorflow.python.framework import constant_op

from tensorflow.python.framework import dtypes

from tensorflow.python.layers.core import Dense

from tensorflow.python.util import nest


num_layers = 2
num_nodes = 512
hidden_dim = 512
vocab_size = 542
label_dim = 13
batch_size = 32
num_unrollings = 15
start_seq_index = 0
source_sequence_length = 15
beam_width = 7
tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':10,'NonRetailRelated':11, '$start$':12, '$end$':0}

#tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':10,'NonRetailRelated':11, '$padding$':0}
id_to_tag = {v:k for (k,v) in tag_to_id.items()}


def accuracy(labels,predictions):
    return np.sum(labels==np.argmax(predictions,axis=1))/len(labels)

def pred_to_tag(pred):
    preds = []
    for singles in pred:
        preds.append([id_to_tag[single] for single in singles])
    return preds


def make_onehot(categorical):
	zeros = np.zeros((len(categorical),label_dim))
	zeros[np.arange(len(categorical)),categorical] = 1
	return zeros
	
def attn_decoder_fn(inputs, attention):
    return tf.concat([inputs, attention],-1)
def attn_decoder_input_fn(inputs, attention):
    _input_layer = Dense(num_nodes,name='attn_input_feeding', dtype = tf.float32)
    return _input_layer(array_ops.concat([inputs, attention], -1))
##################################### divider 
global graph
graph = tf.Graph()

with graph.as_default():
    
    # encoder_inputs: [batch_size, max_time_steps]
    print(" building the model...")
    global_step = tf.Variable(0, trainable=False, name = 'global_step')
    encoder_inputs = tf.placeholder(dtype=tf.float32,
        shape=(None,None, vocab_size), name='encoder_inputs')
    #encoder_embeddings = tf.get_variable(name = 'embeddings', shape = [vocab_size, vocab_size], initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
    #print(encoder_inputs.get_shape())
    encoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='encoder_inputs_length')

    batch_size = tf.shape(encoder_inputs)[0]
    #training
    decoder_inputs = tf.placeholder(
                dtype=tf.float32, shape=(None, None, label_dim), name='decoder_inputs')
    decoder_inputs_length = tf.placeholder(dtype=tf.int32 ,shape = (None,), name= 'decoder_inputs_length')

    #print(decoder_start_token.get_shape())
    decoder_start_token = tf.expand_dims(tf.one_hot(tf.tile(tf.constant([12], dtype = tf.int32), [batch_size]),label_dim,dtype=tf.float32),1)
    print(decoder_start_token.shape)
    #decoder_start_token = tf.nn.embedding_lookup(tgt_w, start_token)
    
    decoder_inputs_train = tf.concat([decoder_start_token,decoder_inputs], axis=1)
    decoder_inputs_train = decoder_inputs_train[:,:-1,:]
    decoder_inputs_length_train = decoder_inputs_length # + 1
    #decoder_targets_train = tf.concat([decoder_inputs, decoder_start_token], axis = 1)
    decoder_targets_train = tf.placeholder(dtype = tf.int32, shape =(None, None), name='decoder_targets_train')
	################################################### ends initialization 
	#set up encoder 
    forward_cell = [tf.nn.rnn_cell.BasicLSTMCell(num_nodes) for _ in range(num_layers)]
    backward_cell = [tf.nn.rnn_cell.BasicLSTMCell(num_nodes) for _ in range(num_layers)]
    (encoder_output, encoder_last_state_fw, encoder_last_state_bw) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(forward_cell
	, backward_cell, inputs = encoder_inputs, sequence_length = encoder_inputs_length, dtype = tf.float32, time_major=False)
    encoder_outputs = tf.concat(encoder_output, axis = 2)
    encoder_last_state = []
    for x in range(num_layers):
        encoder_last_state_c = tf.concat([encoder_last_state_fw[x].c, encoder_last_state_bw[x].c],1)
        encoder_last_state_h = tf.concat([encoder_last_state_fw[x].h,encoder_last_state_bw[x].h],1)
        encoder_last_state.append(tf.contrib.rnn.LSTMStateTuple(c = encoder_last_state_c, h = encoder_last_state_h))
    encoder_last_state = tuple(encoder_last_state)

    #batch_size = batch_size * beam_width
    ######################################################### ends building encoder
	# building training decoder, no beam search
    with tf.variable_scope('shared_attention_mechanism'):
        attention_mechanism = seq2seq.BahdanauAttention(num_units = hidden_dim*2, memory = encoder_outputs, memory_sequence_length = encoder_inputs_length)
    global_decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(hidden_dim*2) for _ in range(num_layers)])
    projection_layer = Dense(label_dim)
    
    decoder_cell = seq2seq.AttentionWrapper(cell = global_decoder_cell, 
    #tf.nn.rnn_cell.BasicLSTMCell(hidden_dim*2),
    attention_mechanism = attention_mechanism,
    attention_layer_size = hidden_dim*2)
    #input_vectors = tf.nn.embedding_lookup(tgt_w, decoder_inputs)
    print(decoder_inputs.shape,decoder_inputs.shape )
    #decoder training
    training_helper = seq2seq.TrainingHelper(
        inputs = decoder_inputs_train,
        sequence_length = tf.tile(tf.constant([15], dtype = tf.int32), [batch_size]),#decoder_inputs_length_train,
        time_major = False
    )
    #print(decoder_cell.zero_state(batch_size,tf.float32))
    training_decoder = seq2seq.BasicDecoder(
        cell = decoder_cell,
        helper = training_helper,
        initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_last_state),
        output_layer = projection_layer
    )

    with tf.variable_scope('decode_with_shared_attention'):
        training_decoder_output, _, _ = seq2seq.dynamic_decode(
            decoder =training_decoder,
            impute_finished = True,
            maximum_iterations = 15 #tf.reduce_max(decoder_inputs_length_train)
        )
    #print(training_decoder_output)

    training_logits = training_decoder_output.rnn_output
    print(training_logits.shape)
    #decoder_logits_train = tf.identity(decoder_outputs_train.rnn_output)

    decoder_pred_train = tf.argmax(training_logits, axis = -1, output_type = tf.int32)
    #print(decoder_pred_train)
    masks = tf.sequence_mask(lengths = decoder_inputs_length_train, maxlen= 15, dtype = tf.float32)
    print("mask and decoder_target shapes", masks.shape, decoder_targets_train.shape)
    loss = seq2seq.sequence_loss(logits = training_logits,
    targets = decoder_targets_train,
	weights = masks,
	average_across_timesteps = True,
	average_across_batch = True)
    


    tf.summary.scalar('loss', loss)
	################################################
	#optimizer
    print("setting optimizer...")
    trainable_params = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    gradients = tf.gradients(loss, trainable_params)
    clip_gradients,_ = tf.clip_by_global_norm(gradients, 5.0)
    updates = optimizer.apply_gradients(zip(clip_gradients, trainable_params), global_step = global_step)

    summary_op = tf.summary.merge_all()
    ####################################################
    #decoder prediction
    
    def no_op_embedding(inputs):
        return tf.one_hot(inputs, label_dim)
    
    ###################################################
    #decoder beamsearch
    encoder_outputs = seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
    encoder_inputs_length = seq2seq.tile_batch(encoder_inputs_length,multiplier = beam_width)
    encoder_last_state = seq2seq.tile_batch(encoder_last_state, multiplier = beam_width)
    
    with tf.variable_scope('shared_attention_mechanism', reuse=True):
        attention_mechanism = seq2seq.BahdanauAttention(num_units = hidden_dim*2, memory = encoder_outputs, memory_sequence_length = encoder_inputs_length)
    #decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(hidden_dim*2) for _ in range(num_layers)])
    decoder_cell = seq2seq.AttentionWrapper(cell = global_decoder_cell,
    attention_mechanism = attention_mechanism,
    attention_layer_size = hidden_dim*2)

    inference_decoder = seq2seq.BeamSearchDecoder(cell = decoder_cell,
    embedding = no_op_embedding,
    start_tokens = tf.fill([batch_size],12),
    end_token = 0,
    initial_state = decoder_cell.zero_state(batch_size*beam_width, tf.float32).clone(cell_state=encoder_last_state),
    beam_width = beam_width,
    #initial_state = decoder_cell_inf.zero_state(batch_size = batch_size, dtype = tf.float32)
    output_layer = projection_layer)

    print(inference_decoder) 
    with tf.variable_scope('decode_with_shared_attention', reuse = True):
        inference_decoder_output, _, _ = seq2seq.dynamic_decode(
            decoder =inference_decoder,
            impute_finished = False,
            maximum_iterations = tf.reduce_max(encoder_inputs_length)
        )   

    for var in tf.trainable_variables():
        print(var)
    beam_search_decoder_out = inference_decoder_output.predicted_ids 
    beam_search_decoder_out = tf.identity(beam_search_decoder_out, name = 'final_out')
    saver = tf.train.Saver()		
""""""""""""""""""""""""""""""""""""""""""""""""""""""


if __name__ == "__main__":
	num_steps = 100000
	print("data is loaded")
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()   #initializes all variables at once"""
		mean_loss = 0
		print("ready to start training")
		step = 0
		test_generator = reader.data_generator_s2s_train('Test_March_2016_1C.tsv', tag_to_id, 15, 1)
		for batch_features, batch_labels, lengths, labels in reader.data_generator_s2s_train('CombinedTraining_removed_test.tsv',tag_to_id,15,32, True):
			if step > num_steps:
			    break
			#print(batch_features[0].shape, batch_labels.shape, lengths.shape, labels.shape)
			#batch_features = np.concatenate((np.asarray(batch_features]),np.asarray(batch_dict_features)),axis=1).tolist()
			feed_dict = dict()
			feed_dict['encoder_inputs:0'] = batch_features[0]
			feed_dict['encoder_inputs_length:0'] = lengths
			feed_dict['decoder_inputs:0'] = batch_labels
			feed_dict['decoder_inputs_length:0'] = lengths
			feed_dict['decoder_targets_train:0'] = labels
            #feed_dict['decoder_target:0'] = batch_labels
			_,l, summary, preds = session.run([updates, loss, summary_op,decoder_pred_train], feed_dict = feed_dict)
			step += 1
			mean_loss += l   
		
			if step % (20) == 0:
			 	#dropout = 1
			 	print('-'*80)
			 	print("training sample label ", pred_to_tag(labels)[0])
			 	print("predicted training sample " , pred_to_tag(preds.tolist())[0])
			 	print("step " + str(step))
			 	print("loss " + str(l))
			 	print("cumulative loss is", float(mean_loss)/float(step))
			 	batch_features, batch_labels, lengths, labels = next(test_generator)				
			 	print(" finished loading data ", labels)
			 	print("validation sample ", pred_to_tag(labels))
			 	feed_dict_inf = dict()
			 	feed_dict_inf['encoder_inputs:0'] = batch_features[0]
			 	feed_dict_inf['encoder_inputs_length:0'] = lengths
			 	print("ready to generate output")
			 	outputs = session.run([beam_search_decoder_out], feed_dict_inf)
			 	print(outputs)
			# 	print("Batch accuracy : %.2f" %float(accuracy(labels,predictions)*100))
			# 	reset_sample_state.run()
			# 	test_features, test_dict_features, test_position_features, test_targets, test_sequence_len = reader.next_batch2(test_data, step, num_unrollings)
			# 	test_dict = dict()
			# 	for i in range(1):
			# 		test_dict[sample_input[i]] = np.asarray(test_features[i])
			# 	sample_pred = session.run(sample_outputs, feed_dict=test_dict)
			# 	labels = np.concatenate(list(test_targets)[:],axis=0)
			# 	pred = np.concatenate(list(sample_pred)[:],axis=0)
			# 	print('validation accuracy is : %0.2f' %accuracy(labels, pred))
			# 	print(test_targets)
			if step%2000 == 0:
			 	print("saving intermediate model")
			 	saver.save(session, save_path = './tf_s2s_bs_models_decoder_stacked/s2s_per500', global_step = global_step)
		print("Saving trained model")
		saver.save(session, save_path = './tf_seq2seq_beam_search_random_decoder_stacked/', global_step = global_step)		
			
			
	