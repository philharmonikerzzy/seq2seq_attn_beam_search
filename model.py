import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
import tensorflow.contrib.seq2seq as seq2seq

class Model(object):

    def __init__(self, config, mode):

        assert mode.lower() in ['training', 'inference']

        self.config = config
        self.mode = mode

        self.cell_type = config['cell_type']
        self.hidden_units = config['hidden_units']
        self.num_layers = config['num_layers']
        self.embedding_size = config['embedding_size']
        self.num_labels = config['num_labels']
        self.train_length = config['train_length']

        self.max_gradient_norm = config['max_gradient_norm']

        self.global_step = tf.Variable(0, trainable = False, name = 'global_step')

        if self.mode == 'inference':
            self.beam_width = config['beam_width']
            self.use_beam_search = True if self.beam_width>1 else False

        self.build_model()
    
    def build_model(self):
        print("start building model...")

        self.init_variables()
        self.build_encoder()
        self.build_decoder()

        self.merge_summery = tf.summary.merge_all()
    

    def init_variables(self):
        # [batch_size, max_time_step_per_batch, embedding_size]
        self.encoder_inputs = tf.placeholder(dtype = tf.float32, shape = 
        (None, None, self.embedding_size), name = 'encoder_input')
        #[batch_size]
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name = 'encoder_input_length')
        #dynamic batch size
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        if self.mode == 'train':
            #[batch_size, max_time_steps_per_batch, label_dim] 
            self.decoder_inputs = tf.placeholder(dtype=tf.float32,
            shape = (None, None, self.num_labels), name = 'decoder_inputs')
            #[batch_size]
            self.decoder_inputs_length = tf.placeholder(dtype = tf.int32, shape=(None,), name='decoder_inputs_length')

            self.decoder_start_token = tf.expand_dims(tf.one_hot(tf.tile(tf.constant([12], dtype = tf.int32)
            , [self.batch_size]),self.num_labels,dtype=tf.float32),1)
            #[batch_size, max_time_steps]
            self.decoder_inputs_train = tf.concat([self.decoder_start_token,self.decoder_inputs], axis=1)
            self.decoder_inputs_train = self.decoder_inputs_train[:,:-1,:]
            self.decoder_inputs_length_train = self.decoder_inputs_length # + 1
            #[batch_size, max_time_steps] label is fed as the index of the tag
            self.decoder_targets_train = tf.placeholder(dtype = tf.int32, shape =(None, None), name='decoder_targets_train')
    
    def build_encoder(self):
        print("building encoder part of the model ...")

        with tf.variable_scope('encoder'):
            forward_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units) for _ in range(self.num_layers)])
            backward_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units) for _ in range(self.num_layers)])
            # builds independent forward and backward RNNs. 
            # encoder_last_state_fw is a lstm final state, a tuple of LSTMStateTuple
            (encoder_output_fw, encoder_output_bw), (encoder_last_state_fw, encoder_last_state_bw) = tf.nn.bidirectional_dynamic_rnn(forward_cell
	        , backward_cell, inputs = self.encoder_inputs, sequence_length = self.encoder_inputs_length, dtype = tf.float32, time_major=False)
    
            self.encoder_outputs = tf.concat([encoder_output_fw, encoder_output_bw], axis = 2)
            self.encoder_final_state = []
            for n in range(self.num_layers):
                #concatenate the forward and backward states of each layer 
                encoder_final_state_c = tf.concat([encoder_last_state_fw[n].c, encoder_last_state_bw[n].c],1)
                encoder_final_state_h = tf.concat([encoder_last_state_fw[n].h, encoder_last_state_bw[n].h],1)
                self.encoder_final_state.append(tf.contrib.rnn.LSTMStateTuple(c=encoder_final_state_c,
                h=encoder_final_state_h))
            self.encoder_final_state = tuple(self.encoder_final_state)
    
    def build_decoder(self):
        print("building the decoder and attention part of the model")

        with tf.variable_scope('decoder'):
            self.decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(
                self.hidden_units*2) for _ in range(self.num_layers)] 
            )
            #

            output_layer = Dense(self.num_labels, name = 'output_projection')

            if self.mode == "training":
                
                self.attention_mechanism = tf.seq2seq.BahdanauAttention(
                num_units = self.hidden_units*2, memory = self.encoder_outputs, memory_sequence_length = self.encoder_inputs_length)
            
                self.decoder_cell = tf.seq2seq.AttentionWrapper(
                    cell = self.decoder_cell,
                    attention_mechanism = self.attention_mechanism,
                    attention_layer_size = self.hidden_units*2
                )        
                
                training_helper = tf.seq2seq.TrainingHelper(
                    inputs = self.decoder_inputs_train,
                    sequence_length = tf.tile(tf.constant([self.train_length]))
                )

                training_decoder = seq2seq.BasicDecoder(
                    cell = self.decoder_cell,
                    helper = training_helper,
                    initial_state = self.decoder_cell.zero_state(self.batch_size, tf.float32).clone(
                        cell_state = self.encoder_final_state),
                        output_layer = output_layer
                    ) 
                )

                self.training_decoder_output, _, _ = seq2seq.dynamic_decode(
                    decoder = training_decoder,
                    impute_finished = True,
                    maximum_iterations = self.train_length
                )

                self.training_logits = training_decoder_output.rnn_output
                self.decoder_pred_train = tf.argmax(self.training_logits, axis = -1, name = 'decoder_pred_train')

                masks = tf.sequence_mask(lengths = self.decoder_inputs_length_train, maxlen = self.train_length, 
                dtype = tf.float32)
                self.loss = seq2seq.sequence_loss(logits = self.training_logits,
                targets = self.decoder_targets_train,
                weights = masks,
                )
                tf.summary.scalar('loss', self.loss)

                self.init_optimizer

                

