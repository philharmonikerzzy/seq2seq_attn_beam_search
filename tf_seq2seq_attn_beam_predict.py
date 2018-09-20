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



num_nodes = 512
hidden_dim = 512
vocab_size = 542
label_dim = 13
batch_size = 32
num_unrollings = 20
start_seq_index = 0
source_sequence_length = 20
beam_width = 3
tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':10,'NonRetailRelated':11, '$start$':12, '$end$':0}

#tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':10,'NonRetailRelated':11, '$padding$':0}
id_to_tag = {v:k for (k,v) in tag_to_id.items()}


class TaggingMetrics:

  def __init__(self, is_s2s = False):
    self.brand_pos = 0
    self.brand_tp = 0
    self.productclass_pos = 0
    self.productclass_tp = 0
    self.model_tp = 0
    self.model_pos = 0
    self.relevant_term_tp = 0
    self.relevant_term_pos = 0
    self.is_s2s = is_s2s

    
  
  def get_accuracy(self,type = "all"):
      if "brand" == type:
        if self.brand_pos == 0:
          return 0.0
        return 1.0*self.brand_tp/self.brand_pos
      if "productclass" == type:
        if self.productclass_pos == 0:
            return 0.0
        return 1.0*self.productclass_tp/self.productclass_pos
      if "model" == type:
        if self.model_pos == 0:
          return 0.0
        return 1.0*self.model_tp/self.model_pos
      else:
        if self.relevant_term_pos == 0:
          return 0.0
        return 1.0*self.relevant_term_tp/self.relevant_term_pos
  
  def _update_sample_metric(self,sample_tag, sample_pred):
    if self.is_s2s:
      endidx = sample_tag.index('$end$')
      sample_tag, sample_pred = sample_tag[:endidx], sample_pred[:endidx] 
    sample_tag_pred = list(zip(sample_tag, sample_pred))
    #print(sample_tag_pred)

    self.brand_pos += sum([1 for pair in sample_tag_pred if (pair[0] == 'Brand')])
    self.brand_tp += sum([1 for pair in sample_tag_pred if ((pair[0] == pair[1]) and (pair[0] == 'Brand'))])
    self.productclass_pos += sum([1 for pair in sample_tag_pred if  (pair[0] == 'ProductClass')]) 
    self.productclass_tp += sum([1 for pair in sample_tag_pred if ((pair[0] == pair[1]) and (pair[0] == 'ProductClass'))])
    self.model_tp += sum([1 for pair in sample_tag_pred if ((pair[0] == pair[1]) and (pair[0] == 'Model'))])
    self.model_pos += sum([1 for pair in sample_tag_pred if ((pair[0] == 'Model'))]) 
    self.relevant_term_tp += sum([1 for pair in sample_tag_pred if ((pair[0] == pair[1]) and (pair[0] in 'ProductClass, Merchant, Model, ProductClass, ResearchIntent, SpecAttribute'))])
    self.relevant_term_pos += sum([1 for pair in sample_tag_pred if ((pair[0] in 'ProductClass, Merchant, Model, ProductClass, ResearchIntent, SpecAttribute'))]) 
    
  def update_metrics(self, tag_batch, pred_batch):
    for idx, tags in enumerate(tag_batch):
      self._update_sample_metric(tags, pred_batch[idx])



def accuracy(labels,predictions):
    return np.sum(labels==np.argmax(predictions,axis=1))/len(labels)

def pred_to_tag(pred, is_beam = False):
    preds = []
    for singles in pred:
        if is_beam:
            preds.append([id_to_tag[single[:,:,0]] for single in singles]) 
        else:
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
loaded_graph = tf.Graph()

if __name__ == "__main__":
	num_steps = 5
	print("data is loaded")
	with tf.Session(graph=loaded_graph) as session:
		metrics = TaggingMetrics(is_s2s=False)
		loader =tf.train.import_meta_graph('./5500/s2s_per500-5500.meta')
		loader.restore(session,'./5500/s2s_per500-5500')
		mean_loss = 0
		print("ready to start inferring")
		step = 0
		test_generator = reader.data_generator_s2s_predict('BingTest_1C.txt', tag_to_id, 20, 32)
		sentinel = object()
		has_next = True
		encoder_inputs = loaded_graph.get_tensor_by_name('encoder_inputs:0')
		encoder_inputs_length = loaded_graph.get_tensor_by_name('encoder_inputs_length:0')
		beam_search_decoder_out = loaded_graph.get_tensor_by_name('final_out:0')
		while has_next:
			future_data = next(test_generator, sentinel)
			if (future_data is sentinel):
			 	break
			batch_features, batch_labels, lengths, labels = future_data[0], future_data[1], future_data[2], future_data[3]
			print("ready to generate output")
			outputs = session.run([beam_search_decoder_out], {encoder_inputs:batch_features[0],encoder_inputs_length:lengths})
			#print(outputs, outputs[0][:,:,0], pred_to_tag(labels), pred_to_tag(outputs[0][:,:,0]))
			metrics.update_metrics(pred_to_tag(labels),pred_to_tag(outputs[0][:,:,0]))
	print ("overall accuracy is " + str(metrics.get_accuracy()))
	print ("overall brand accuracy is " + str(metrics.get_accuracy("brand")))
	print ("overall product class accuracy is " + str(metrics.get_accuracy("productclass")))
	print ("overall model accuracy is " + str(metrics.get_accuracy("model")))			
			
	