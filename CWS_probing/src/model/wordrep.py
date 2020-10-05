from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
from .charbilstm import CharBiLSTM
from .charbigru import CharBiGRU
from .charcnn import CharCNN
from .ncrf_transformers import NCRFTransformers

class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        if not data.silence:
            print("build word representation...")
        self.gpu = data.HP_gpu
        self.device = data.device
        self.use_word_emb = data.use_word_emb
        self.use_char = data.use_char
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        self.char_all_feature = False
        self.char_feature_extractor = data.char_feature_extractor
        self.sentence_classification = data.sentence_classification
        if self.use_char and data.char_feature_extractor != "None" and data.char_feature_extractor != None:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if not data.silence:
                print("build char sequence feature extractor: %s ..."%(data.char_feature_extractor))
            if data.char_feature_extractor == "CNN":
                self.char_feature = CharCNN(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.device)
            elif data.char_feature_extractor == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.device)
            elif data.char_feature_extractor == "GRU":
                self.char_feature = CharBiGRU(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.device)
            elif data.char_feature_extractor == "ALL":
                self.char_all_feature = True
                self.char_feature = CharCNN(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.device)
                self.char_feature_extra = CharBiLSTM(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.device)
            else:
                print("Error char feature selection, please check parameter data.char_feature_extractor (CNN/LSTM/GRU/ALL).")
                exit(0)
        self.low_level_transformer = data.low_level_transformer
        if self.low_level_transformer != None and self.low_level_transformer.lower() != "none":
            self.low_level_transformer_finetune = data.low_level_transformer_finetune
            self.transformer = NCRFTransformers(self.low_level_transformer, data.device)
        if self.use_word_emb:
            self.embedding_dim = data.word_emb_dim
            self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim).to(self.device)
            if data.pretrain_word_embedding is not None:
                self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
            else:
                self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        self.feature_num = data.feature_num
        self.feature_embedding_dims = data.feature_emb_dims
        self.feature_embeddings = nn.ModuleList()
        for idx in range(self.feature_num):
            self.feature_embeddings.append(nn.Embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx]).to(self.device))
        for idx in range(self.feature_num):
            if data.pretrain_feature_embeddings[idx] is not None:
                self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(data.pretrain_feature_embeddings[idx]))
            else:
                self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(self.random_embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx])))
        self.drop = nn.Dropout(data.HP_dropout).to(self.device)




    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def forward(self, *input):
        word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_word_text = input[:7]
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_list = []
        if self.use_word_emb:
            word_embs =  self.word_embedding(word_inputs)
            word_list.append(word_embs)
        if not self.sentence_classification:
            for idx in range(self.feature_num):
                word_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        if self.use_char and self.char_feature_extractor.lower() != "none" and self.char_feature_extractor != None:
            ## calculate char lstm last hidden
            # print("charinput:", char_inputs)
            # exit(0)
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            ## concat word and char together
            word_list.append(char_features)
            if self.char_all_feature:
                char_features_extra = self.char_feature_extra.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
                char_features_extra = char_features_extra[char_seq_recover]
                char_features_extra = char_features_extra.view(batch_size,sent_len,-1)
                ## concat word and char together
                word_list.append(char_features_extra)
        
        if self.low_level_transformer != None and self.low_level_transformer.lower() != "none":
            if self.training and self.low_level_transformer_finetune:
                self.transformer.train()
                transformer_output = self.transformer.extract_features(batch_word_text)
            else:
                self.transformer.eval()
                with torch.no_grad():
                    transformer_output = self.transformer.extract_features(batch_word_text)
            word_list.append(transformer_output)
        word_embs = torch.cat(word_list, 2)
        # if a == 0:
        #     print("inputs", word_inputs)
        #     print("embeddings:", word_embs)
        word_represent = self.drop(word_embs)
        return word_represent
