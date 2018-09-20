# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os,io
import numpy as np
import gensim
from sklearn.preprocessing import OneHotEncoder
label_dim = 13

def make_onehot(categorical):
	zeros = np.zeros((len(categorical),label_dim))
	zeros[np.arange(len(categorical)),categorical] = 1
	return zeros

embeding_size = 500
num_dict = 42

def _read_dicts():
  dicts = []
  for i in range(num_dict):
    with io.open("PA_CRF_Dict/"+str(i)+".txt", "r",encoding="utf-8") as f:
      data = f.read().replace('\r','').split("\n")
      dicts.append(set(data))
  return dicts

def _generate_dict_hitmap(dicts, querystr):
  dict_feature = [[0.0]*num_dict for i in range(len(querystr))]
  s = ' '
  begin = 0
  while begin < len(querystr):
    end = len(querystr)
    while end > begin:
      query_slice = s.join(querystr[begin:end])
      for k in range(len(dicts)):
        if query_slice in dicts[k]:
          for i in range(begin, end):
            dict_feature[i][k] = 1.0
      end = end - 1
    begin = begin + 1
  return dict_feature


def data_generator2(filename, tag_to_id, query_len_threshold, batch_size):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  features = []
  decoder_inputs = []
  decoder_labels = []
  lengths = []
  labels = []
  dicts = _read_dicts()
  index = 0
  f = io.open(filename, "r",encoding="utf-8")
  while 1:
    line = f.readline()
    if not line:
      f.close()
      f = io.open(filename, "r", encoding="utf-8")
      continue
    index=index+1
    query = []
    tags = []
    querystr = []
    position = []
    sequence_len = 0
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(word2vec[term].tolist())
      else:
        query.append(zero_vec)
      tags.append(tag_to_id[tag])
      position.append(len(tags)-1)
      querystr.append(term)
    sequence_len = min(len(query),query_len_threshold)
    dict_feature = _generate_dict_hitmap(dicts, querystr)
    while len(query) < query_len_threshold:
      query.append(zero_vec)
      tags.append(tag_to_id['$padding$'])
      position.append(len(tags)-1)
      dict_feature.append([0.0]*num_dict)
    #records.append([query[:query_len_threshold] + dict_feature[:query_len_threshold], position[:query_len_threshold], tags[:query_len_threshold], sequence_len])
    features.append(np.concatenate((np.asarray(query[:query_len_threshold]), np.asarray(dict_feature[:query_len_threshold])),axis=1).tolist())
    label = tags[:query_len_threshold]
    labels.append(label)
    decoder_input = make_onehot(label[0:-1]).tolist()
    decoder_input.insert(0,make_onehot([0])[0].tolist())
    decoder_inputs.append(decoder_input)
    decoder_labels.append(make_onehot(label))
    lengths.append(20)
    if index==batch_size:
      yield [np.asarray(features), np.asarray(decoder_inputs)], np.asarray(decoder_labels), np.asarray(lengths), np.asarray(labels)
      index = 0
      decoder_inputs = []
      features = []
      decoder_labels = []
      lengths = []
      labels = []
      
def data_generator_s2s(filename, tag_to_id, query_len_threshold, batch_size):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  features = []
  decoder_inputs = []
  decoder_labels = []
  lengths = []
  labels = []
  dicts = _read_dicts()
  index = 0
  f = io.open(filename, "r",encoding="utf-8")
  while 1:
    line = f.readline()
    if not line:
      f.close()
      f = io.open(filename, "r", encoding="utf-8")
      continue
    index=index+1
    query = []
    tags = []
    querystr = []
    position = []
    sequence_len = 0
    query.append(np.asarray(zero_vec))
    tags.append(tag_to_id['$start$'])
    querystr.append('$start$')
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(word2vec[term].tolist())
      else:
        query.append(zero_vec)
      tags.append(tag_to_id[tag])
      position.append(len(tags)-1)
      querystr.append(term)
    query.append(np.asarray(zero_vec))
    tags.append(tag_to_id['$end$'])
    querystr.append('$end$')
    sequence_len = min(len(query),query_len_threshold)
    dict_feature = _generate_dict_hitmap(dicts, querystr)
    #print(len(query), len(tags), len(querystr), len(dict_feature), sequence_len)
    while len(query) < query_len_threshold:
      query.append(zero_vec)
      tags.append(tag_to_id['$end$'])
      position.append(len(tags)-1)
      dict_feature.append([0.0]*num_dict)
    #records.append([query[:query_len_threshold] + dict_feature[:query_len_threshold], position[:query_len_threshold], tags[:query_len_threshold], sequence_len])
    features.append(np.concatenate((np.asarray(query[:query_len_threshold]), np.asarray(dict_feature[:query_len_threshold])),axis=1).tolist())
    label = tags[:query_len_threshold]
    labels.append(label)
    decoder_input = make_onehot(label).tolist()
    #decoder_input.insert(0,make_onehot([0])[0].tolist())
    decoder_inputs.append(decoder_input)
    decoder_labels.append(make_onehot(label))
    lengths.append(query_len_threshold)
    if index==batch_size:
      yield [np.asarray(features), np.asarray(decoder_inputs)], np.asarray(decoder_labels), np.asarray(lengths), np.asarray(labels)
      index = 0
      decoder_inputs = []
      features = []
      decoder_labels = []
      lengths = []
      labels = []
      
def data_generator_s2s_train(filename, tag_to_id, query_len_threshold, batch_size):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  features = []
  decoder_inputs = []
  decoder_labels = []
  lengths = []
  labels = []
  dicts = _read_dicts()
  index = 0
  f = io.open(filename, "r",encoding="utf-8")
  while 1:
    line = f.readline()
    if not line:
      f.close()
      print("epoch is over")
      f = io.open(filename, "r", encoding="utf-8")
      continue
    index=index+1
    query = []
    tags = []
    querystr = []
    position = []
    sequence_len = 0
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(word2vec[term].tolist())
      else:
        query.append(zero_vec)
      tags.append(tag_to_id[tag])
      position.append(len(tags)-1)
      querystr.append(term)
    query.append(np.asarray(zero_vec))
    tags.append(tag_to_id['$end$'])
    querystr.append('$end$')
    sequence_len = min(len(query),query_len_threshold)
    dict_feature = _generate_dict_hitmap(dicts, querystr)
    #print(len(query), len(tags), len(querystr), len(dict_feature), sequence_len)
    while len(query) < query_len_threshold:
      query.append(zero_vec)
      tags.append(tag_to_id['$end$'])
      position.append(len(tags)-1)
      dict_feature.append([0.0]*num_dict)
    #records.append([query[:query_len_threshold] + dict_feature[:query_len_threshold], position[:query_len_threshold], tags[:query_len_threshold], sequence_len])
    features.append(np.concatenate((np.asarray(query[:query_len_threshold]), np.asarray(dict_feature[:query_len_threshold])),axis=1).tolist())
    label = tags[:query_len_threshold]
    labels.append(label)
    #decoder_input = make_onehot(label).tolist()
    #decoder_input.insert(0,make_onehot([0])[0].tolist())
    #decoder_inputs.append(decoder_input)
    decoder_labels.append(make_onehot(label))
    lengths.append(sequence_len)
    if index==batch_size:
      yield [np.asarray(features)], np.asarray(decoder_labels), np.asarray(lengths), np.asarray(labels)
      index = 0
      decoder_inputs = []
      features = []
      decoder_labels = []
      lengths = []
      labels = []


def data_generator_s2s_predict(filename, tag_to_id, query_len_threshold, batch_size):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  features = []
  decoder_inputs = []
  decoder_labels = []
  lengths = []
  labels = []
  dicts = _read_dicts()
  index = 0
  f = io.open(filename, "r",encoding="utf-8")
  has_line = True
  while has_line:
    line = f.readline()
    if not line:
      break
    index=index+1
    query = []
    tags = []
    querystr = []
    position = []
    sequence_len = 0
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(word2vec[term].tolist())
      else:
        query.append(zero_vec)
      tags.append(tag_to_id[tag])
      position.append(len(tags)-1)
      querystr.append(term)
    query.append(np.asarray(zero_vec))
    tags.append(tag_to_id['$end$'])
    querystr.append('$end$')
    sequence_len = min(len(query),query_len_threshold)
    dict_feature = _generate_dict_hitmap(dicts, querystr)
    #print(len(query), len(tags), len(querystr), len(dict_feature), sequence_len)
    while len(query) < query_len_threshold:
      query.append(zero_vec)
      tags.append(tag_to_id['$end$'])
      position.append(len(tags)-1)
      dict_feature.append([0.0]*num_dict)
    #records.append([query[:query_len_threshold] + dict_feature[:query_len_threshold], position[:query_len_threshold], tags[:query_len_threshold], sequence_len])
    features.append(np.concatenate((np.asarray(query[:query_len_threshold]), np.asarray(dict_feature[:query_len_threshold])),axis=1).tolist())
    label = tags[:query_len_threshold]
    labels.append(label)
    #decoder_input = make_onehot(label).tolist()
    #decoder_input.insert(0,make_onehot([0])[0].tolist())
    #decoder_inputs.append(decoder_input)
    decoder_labels.append(make_onehot(label))
    lengths.append(sequence_len)
    if index==batch_size:
      yield ([np.asarray(features)], np.asarray(decoder_labels), np.asarray(lengths), np.asarray(labels))
      index = 0
      decoder_inputs = []
      features = []
      decoder_labels = []
      lengths = []
      labels = []

def data_generator_s2s_score(filename, tag_to_id, query_len_threshold, batch_size):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  features = []
  lengths = []
  event_times = []
  input = []
  dicts = _read_dicts()
  index = 0
  f = io.open(filename, "r",encoding="utf-8")
  has_line = True
  while has_line:
    line = f.readline()
    if not line:
      break
    index=index+1
    query = []
    querystr = []
    sequence_len = 0
    line = line.strip()
    EventTime = line.split('\t')[1]
    line = line.split('\t')[0]
    for term in line.split(" "):
      if term in word2vec.wv.vocab:
        query.append(word2vec[term].tolist())
      else:
        query.append(zero_vec)
      querystr.append(term)
    query.append(np.asarray(zero_vec))
    querystr.append('$end$')
    sequence_len = min(len(query),query_len_threshold)
    dict_feature = _generate_dict_hitmap(dicts, querystr)
    #print(len(query), len(tags), len(querystr), len(dict_feature), sequence_len)
    while len(query) < query_len_threshold:
      query.append(zero_vec)
      querystr.append('$end$')
      dict_feature.append([0.0]*num_dict)
    #records.append([query[:query_len_threshold] + dict_feature[:query_len_threshold], position[:query_len_threshold], tags[:query_len_threshold], sequence_len])
    features.append(np.concatenate((np.asarray(query[:query_len_threshold]), np.asarray(dict_feature[:query_len_threshold])),axis=1).tolist())
    lengths.append(sequence_len)
    event_times.append(EventTime)
    input.append(querystr)
    if index==batch_size:
      yield ([np.asarray(features)], np.asarray(lengths), input, event_times)
      index = 0
      features = []
      event_times = []
      input = []
      lengths = []


def data_generator(filename, tag_to_id, query_len_threshold, batch_size):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  records = []
  features = []
  labels = []
  dicts = _read_dicts()
  with io.open(filename, "r",encoding="utf-8") as f:
    data = f.read().splitlines()
  for line in data:
    query = []
    tags = []
    querystr = []
    position = []
    sequence_len = 0
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(word2vec[term].tolist())
      else:
        query.append(zero_vec)
      tags.append(tag_to_id[tag])
      position.append(len(tags)-1)
      querystr.append(term)
    sequence_len = min(len(query),query_len_threshold)
    dict_feature = _generate_dict_hitmap(dicts, querystr)
    while len(query) < query_len_threshold:
      query.append(zero_vec)
      tags.append(tag_to_id['$padding$'])
      position.append(len(tags)-1)
      dict_feature.append([0.0]*num_dict)
    #records.append([query[:query_len_threshold] + dict_feature[:query_len_threshold], position[:query_len_threshold], tags[:query_len_threshold], sequence_len])
    feature = np.concatenate((np.asarray(query[:query_len_threshold]), np.asarray(dict_feature[:query_len_threshold])),axis=1).tolist()
    label = tags[:query_len_threshold]
    decoder_input = make_onehot(label[0:-1]).tolist()
    decoder_input.insert(0,make_onehot([0])[0].tolist())
    decoder_label = make_onehot(label) 
    yield [np.asarray([feature]), np.asarray([decoder_input])], np.asarray([decoder_label])
  
def _read_query_and_tagger3(filename, tag_to_id, query_len_threshold):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  records = []
  features = []
  labels = []
  dicts = _read_dicts()
  with io.open(filename, "r",encoding="utf-8") as f:
    data = f.read().splitlines()
  for line in data:
    query = []
    tags = []
    querystr = []
    position = []
    sequence_len = 0
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(word2vec[term].tolist())
      else:
        query.append(zero_vec)
      tags.append(tag_to_id[tag])
      position.append(len(tags)-1)
      querystr.append(term)
    sequence_len = min(len(query),query_len_threshold)
    dict_feature = _generate_dict_hitmap(dicts, querystr)
    while len(query) < query_len_threshold:
      query.append(zero_vec)
      tags.append(tag_to_id['$padding$'])
      position.append(len(tags)-1)
      dict_feature.append([0.0]*num_dict)
    #records.append([query[:query_len_threshold] + dict_feature[:query_len_threshold], position[:query_len_threshold], tags[:query_len_threshold], sequence_len])
    features.append(np.concatenate((np.asarray(query[:query_len_threshold]), np.asarray(dict_feature[:query_len_threshold])),axis=1).tolist())
    labels.append(tags[:query_len_threshold])
  return features, labels

def _read_query_and_tagger2(filename, tag_to_id, query_len_threshold):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  records = []
  dicts = _read_dicts()
  with io.open(filename, "r",encoding="utf-8") as f:
    data = f.read().splitlines()
  for line in data:
    query = []
    tags = []
    querystr = []
    position = []
    sequence_len = 0
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(word2vec[term].tolist())
      else:
        query.append(zero_vec)
      tags.append(tag_to_id[tag])
      position.append(len(tags)-1)
      querystr.append(term)
    sequence_len = min(len(query),query_len_threshold)
    dict_feature = _generate_dict_hitmap(dicts, querystr)
    while len(query) < query_len_threshold:
      query.append(zero_vec)
      tags.append(tag_to_id['$padding$'])
      position.append(len(tags)-1)
      dict_feature.append([0.0]*num_dict)
    records.append([query[:query_len_threshold], dict_feature[:query_len_threshold], position[:query_len_threshold], tags[:query_len_threshold], sequence_len])
  return records
  
def _read_query_and_tagger_interestingtags(filename, tag_to_id, query_len_threshold):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  records = []
  dicts = _read_dicts()
  with io.open(filename, "r",encoding="utf-8") as f:
    data = f.read().splitlines()
  for line in data:
    query = []
    tags = []
    querystr = []
    position = []
    sequence_len = 0
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(word2vec[term].tolist())
      else:
        query.append(zero_vec)
      tags.append(tag_to_id[tag])
      position.append(len(tags)-1)
      querystr.append(term)
    sequence_len = min(len(query),query_len_threshold)
    dict_feature = _generate_dict_hitmap(dicts, querystr)
    while len(query) < query_len_threshold:
      query.append(zero_vec)
      tags.append(tag_to_id['$padding$'])
      position.append(len(tags)-1)
      dict_feature.append([0.0]*num_dict)
    records.append([query[:query_len_threshold], dict_feature[:query_len_threshold], position[:query_len_threshold], tags[:query_len_threshold], sequence_len])
  return records

def _read_query_and_tagger(filename, tag_to_id, query_len_threshold):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  records = []
  dicts = _read_dicts()
  with io.open(filename, "r",encoding="utf-8") as f:
    data = f.read().splitlines()
  for line in data:
    query = []
    tags = []
    sequence_len = 0
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(np.asarray(word2vec[term].tolist()))
      else:
        query.append(np.asarray(zero_vec))
      tags.append(tag_to_id[tag])
    sequence_len = min(len(query),query_len_threshold)
    while len(query) < query_len_threshold:
      query.append(zero_vec)
      tags.append(tag_to_id['$padding$'])
    records.append([query[:query_len_threshold], tags[:query_len_threshold], sequence_len])
  return records

def _read_query_and_tagger_s2s(filename, tag_to_id):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  zero_vec = [0.0]*embeding_size
  records = []
  dicts = _read_dicts()
  with io.open(filename, "r",encoding="utf-8") as f:
    data = f.read().splitlines()
  for line in data:
    query = []
    tags = []
    querystr = []
    sequence_len = 0
    query.append(np.asarray(zero_vec))
    tags.append(tag_to_id['$start$'])
    querystr.append("$start$")
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(np.asarray(word2vec[term].tolist()))
      else:
        query.append(np.asarray(zero_vec))
      tags.append(tag_to_id[tag])
      querystr.append(term)
     #append the end sentence tag
    query.append(np.asarray(zero_vec))
    tags.append(tag_to_id['$end$'])
    querystr.append("$end$")
    dict_feature = _generate_dict_hitmap(dicts, querystr)
    sequence_len = len(query)
    records.append([query[:sequence_len], dict_feature[:sequence_len], tags[:sequence_len], sequence_len])
  return records

def _read_query_and_tagger_np(filename, tag_to_id, query_len_threshold):
  word2vec = gensim.models.word2vec.Word2Vec.load('../Word2Vec/word2vec_'+str(embeding_size))
  encoder = OneHotEncoder(n_values=12)
  zero_vec = [0.0]*embeding_size
  records = []
  dicts = _read_dicts()
  with io.open(filename, "r",encoding="utf-8") as f:
    data = f.read().splitlines()
  for line in data:
    query = []
    tags = []
    sequence_len = 0
    for pair in line.split(" "):
      kv = pair.split("[")
      if len(kv) != 2:
        continue
      term = kv[0]
      tag = kv[1].split("]")[0]
      if not tag in tag_to_id:
        continue
      if term in word2vec.wv.vocab:
        query.append(np.asarray(word2vec[term].tolist()))
      else:
        query.append(np.asarray(zero_vec))
      tags.append(np.asarray(encoder.fit_transform(tag_to_id[tag])))
    sequence_len = min(len(query),query_len_threshold)
    while len(query) < query_len_threshold:
      query.append(np.asarray(zero_vec))
      tags.append(np.asarray(encoder.fit_transform(tag_to_id['$padding$'])))
    records.append([query[:query_len_threshold], tags[:query_len_threshold], sequence_len])
  return records

def write_as_cft(data,filename):
    with open(filename,'w+') as f:
        for ID, sequence in enumerate(data):
            for idx, term in enumerate(sequence[0]) :
                f.write(str(ID)+"\t")
                f.write("|embedding\t")
                f.write(" ".join(str(x) for x in term))
                f.write("\t|tag\t"+str(sequence[1][idx])+":1\n")

def write_as_ctf_nopadding(data,filename):
    with open(filename, 'w+') as f:
        for ID, sequence in enumerate(data):
            sequencelength = sequence[2]
            for idx, term in enumerate(sequence[0]):
                if idx<sequencelength:
                    f.write(str(ID)+"\t")
                    f.write("|embedding\t")
                    f.write(" ".join(str(x) for x in term))
                    f.write("\t|tag\t"+str(sequence[1][idx])+":1\n")
					
def write_as_ctf_withdictionaryS2S(data,filename):
    with open(filename, 'w+') as f:
        for ID, sequence in enumerate(data):
            sequencelength = sequence[3]
            dict = sequence[1]
            for idx, term in enumerate(sequence[0]):
                if idx<sequencelength:
                    f.write(str(ID)+"\t")
                    f.write("|embedding\t")
                    f.write(" ".join(str(x) for x in term))
                    f.write(" ")
                    f.write(" ".join(str(y) for y in dict[idx]))
                    f.write("\t|tag\t"+str(sequence[2][idx])+":1\n")

def write_as_ctf_withdictionaryNonS2S(data,filename):
    with open(filename, 'w+') as f:
        for ID, sequence in enumerate(data):
            sequencelength = sequence[4]
            dict = sequence[1]
            for idx, term in enumerate(sequence[0]):
                if idx<sequencelength:
                    f.write(str(ID)+"\t")
                    f.write("|embedding\t")
                    f.write(" ".join(str(x) for x in term))
                    f.write(" ")
                    f.write(" ".join(str(y) for y in dict[idx]))
                    f.write("\t|tag\t"+str(sequence[3][idx])+":1\n")


            
def get_label(data):
        label = []
        for ID, sequence in enumerate(data):
            labelinseq = []
            sequencelength = sequence[2]
            for idx, term in enumerate(sequence[0]):
                if idx<sequencelength:
                   labelinseq.append(sequence[1][idx])
            label.append(labelinseq) 
        return label


def read_data(vocab_path, data_path, len_threshold):
  tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':10,'NonRetailRelated':11, '$padding$':0}
  data = _read_query_and_tagger(data_path, tag_to_id, len_threshold)
  return data

def read_data_s2s8tags(vocab_path, data_path):
  tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':8,'NonRetailRelated':8, '$start$':10, '$end$':0}
  data = _read_query_and_tagger_s2s(data_path, tag_to_id)
  return data
  
def read_data_s2s(vocab_path, data_path):
  tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':10,'NonRetailRelated':11, '$start$':12, '$end$':0}
  data = _read_query_and_tagger_s2s(data_path, tag_to_id)
  return data

def read_data_np(vocab_path, data_path, len_threshold):
  tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':10,'NonRetailRelated':11, '$padding$':0}
  data = _read_query_and_tagger_np(data_path, tag_to_id, len_threshold)
  return data
  
  
def read_data3(vocab_path, data_path, len_threshold):
  tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':10,'NonRetailRelated':11, '$padding$':0}
  data, label = _read_query_and_tagger3(data_path, tag_to_id, len_threshold)
  return data, label  


def read_data2(vocab_path, data_path, len_threshold):
  tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':10,'NonRetailRelated':11, '$padding$':0}
  data = _read_query_and_tagger2(data_path, tag_to_id, len_threshold)
  return data
  
def read_data_interestingtags(vocab_path, data_path, len_threshold):
  tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':8, 'SpecAttribute':9, 'Unknown':8,'NonRetailRelated':8, '$padding$':0}
  data = _read_query_and_tagger_interestingtags(data_path, tag_to_id, len_threshold)
  return data
  
def read_data_noSortOrder(vocab_path, data_path, len_threshold):
  tag_to_id = {'Brand':1, 'BuyingIntent':2, 'Merchant':3, 'Model':4, 'PartsRepair':5, 'ProductClass':6, 'ResearchIntent':7, 'SortOrder':9, 'SpecAttribute':9, 'Unknown':8,'NonRetailRelated':8, '$padding$':0}
  data = _read_query_and_tagger2(data_path, tag_to_id, len_threshold)
  return data


def next_batch(raw_data, batch_id, batch_size):
  begin = batch_id*batch_size%len(raw_data)
  end = (begin + batch_size)%len(raw_data)
  if begin < end:
    data = raw_data[begin:end]
  else:
    data = raw_data[begin:]+raw_data[:end]
  data=list(zip(*data))
  return list(data[0]), list(data[1]), list(data[2])

def next_batch2(raw_data, batch_id, batch_size):
  begin = batch_id*batch_size%len(raw_data)
  end = (begin + batch_size)%len(raw_data)
  if begin < end:
    data = raw_data[begin:end]
  else:
    data = raw_data[begin:]+raw_data[:end]
  data=list(zip(*data))
  return list(data[0]), list(data[1]), list(data[2]), list(data[3]), list(data[4])
  
def next_random_batch(raw_data, batch_size):
  indices=set()
  while len(indices) < batch_size:
    indices.add(randint(0, len(raw_data)-1))
  data=[raw_data[i] for i in indices]
  data=list(zip(*data))
  return list(data[0]), list(data[1]), list(data[2])
