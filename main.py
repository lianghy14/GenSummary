# coding: utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.seq2seq import *
import pickle
import os
import argparse
from data import clean_str,build_dict,build_dataset,batch_iter, get_init_embedding


class Summodel(object):
    def __init__(self, reversed_dict, art_max_len, sum_max_len, args, Forward_only=False):
        self.vocab_size = len(reversed_dict)
        self.embedding_size = args.embedding_size
        self.num_hidden = args.num_hidden
        self.num_layers = args.num_layers
        self.learning_rate = args.learning_rate
        self.beam_width = args.beam_width

        if not Forward_only:
            self.dropout_rate = args.dropout_rate
        else:
            self.dropout_rate = 0
        self.cell = tf.nn.rnn_cell.BasicLSTMCell
        self.projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False)

        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.X = tf.placeholder(tf.int32, [None, art_max_len])
        self.X_len = tf.placeholder(tf.int32, [None])
        self.decoder_input = tf.placeholder(tf.int32, [None, sum_max_len])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_target = tf.placeholder(tf.int32, [None, sum_max_len])
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("embedding"):
        #This part is the embedding part, we use GloVe to do word2vec, or just train the vector with normal initializer
            if not Forward_only and args.glove:
                init_emb = tf.constant(get_init_embedding(reversed_dict, self.embedding_size), dtype=tf.float32)
            else: # When testing, we use saved model and do not need to initialize the embedding part using the large GloVe data
                init_emb = tf.random_normal([self.vocab_size,self.embedding_size])
            self.embeddings = tf.get_variable("embeddings",initializer=init_emb)
            self.encoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.X), perm=[1,0,2])
            self.decoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.decoder_input), perm=[1,0,2])

        with tf.name_scope("encoder"):
        # Encoder part, we use stack_bidirectional_dynamic_rnn.
            fw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
            #When training, we set a dropout rate
            fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=1-self.dropout_rate) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=1-self.dropout_rate) for cell in bw_cells]
            #...
            encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.encoder_emb_inp, sequence_length=self.X_len,time_major=True,dtype=tf.float32)
            self.encoder_output = tf.concat(encoder_outputs, 2)
            encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
            self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        with tf.name_scope("decoder"), tf.variable_scope("decoder") as decoder_scope:
            decoder_cell = self.cell(2*self.num_hidden)
            if not Forward_only:
                attention_states = tf.transpose(self.encoder_output, [1,0,2])
                attention_mechanism = BahdanauAttention(
                    2*self.num_hidden, attention_states, memory_sequence_length=self.X_len,normalize=True)
                decoder_cell = AttentionWrapper(decoder_cell,attention_mechanism,attention_layer_size=2*self.num_hidden)
                initial_state = decoder_cell.zero_state(dtype=tf.float32,batch_size=self.batch_size)
                initial_state = initial_state.clone(cell_state=self.encoder_state)
                helper = TrainingHelper(self.decoder_emb_inp,self.decoder_len,time_major=True)
                decoder = BasicDecoder(decoder_cell,helper,initial_state)
                outputs,_,_=dynamic_decode(decoder,output_time_major=True,scope=decoder_scope)
                self.decoder_output = outputs.rnn_output
                self.logits = tf.transpose(self.projection_layer(self.decoder_output),perm=[1,0,2])
                self.logits_reshape = tf.concat(
                    [self.logits,tf.zeros([self.batch_size, sum_max_len - tf.shape(self.logits)[1], self.vocab_size])], axis=1)
            else:
                tiled_encoder_output = tile_batch(tf.transpose(self.encoder_output,perm=[1,0,2]),multiplier = self.beam_width)
                tiled_encoder_final_state = tile_batch(self.encoder_state,multiplier=self.beam_width)
                tiled_seq_len = tile_batch(self.X_len,multiplier=self.beam_width)
                attention_mechanism = BahdanauAttention(
                    2*self.num_hidden,tiled_encoder_output,memory_sequence_length=tiled_seq_len,normalize=True)
                decoder_cell = AttentionWrapper(decoder_cell,attention_mechanism,attention_layer_size=self.num_hidden*2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32,batch_size=self.batch_size*self.beam_width)
                initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)
                decoder = BeamSearchDecoder(
                    cell = decoder_cell,embedding=self.embeddings,start_tokens=tf.fill([self.batch_size],tf.constant(2)),
                    end_token=tf.constant(3),initial_state=initial_state,beam_width=self.beam_width,
                    output_layer=self.projection_layer)
                outputs,_,_=dynamic_decode(decoder,output_time_major=True,maximum_iterations=sum_max_len,scope=decoder_scope)
                self.prediction = tf.transpose(outputs.predicted_ids, perm=[1,2,0])

        with tf.name_scope("loss"):
            if not Forward_only:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_reshape, labels=self.decoder_target)
                weights = tf.sequence_mask(self.decoder_len, sum_max_len, dtype=tf.float32)
                self.loss = tf.reduce_sum(crossent * weights / tf.to_float(self.batch_size))
                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients,_=tf.clip_by_global_norm(gradients,5.0)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

def add_arguments(parser):
    parser.add_argument("--num_hidden", type=int, default=150, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Network depth.")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width for beam search decoder.")
    parser.add_argument("--glove", action="store_true", help="Use glove as initial word embedding.")
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate.")

    parser.add_argument("--toy", action="store_true", help="Use only 50K samples of data")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    with open("args.pickle", "wb") as f:
        pickle.dump(args, f)

    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")

    print("Building dictionary...")
    word_dict, reversed_dict, art_max_len, sum_max_len = build_dict('train')
    print("Loading training dataset...")
    train_x, train_y = build_dataset("train", word_dict, art_max_len, sum_max_len)
    with tf.Session() as sess:
        model = Summodel(reversed_dict, art_max_len, sum_max_len,args, Forward_only=False)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        batches = batch_iter(train_x,train_y,args.batch_size,args.num_epochs)
        num_batches_per_epoch = (len(train_x)-1)//args.batch_size +1
        print("Number of batches per epoch :", num_batches_per_epoch)
        print("We are running!")
        for batch_x, batch_y in batches:
            batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
            batch_decoder_input = list(map(lambda x: [word_dict["<s>"]] + list(x), batch_y))
            batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
            batch_decoder_output = list(map(lambda x: list(x) + [word_dict["</s>"]], batch_y))
            batch_decoder_input = list(
                map(lambda d: d + (sum_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_input))
            batch_decoder_output = list(
                map(lambda d: d + (sum_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_output))
            train_feed_dict = {
                model.batch_size: len(batch_x),
                model.X: batch_x,
                model.X_len: batch_x_len,
                model.decoder_input: batch_decoder_input,
                model.decoder_len: batch_decoder_len,
                model.decoder_target: batch_decoder_output
            }

            _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)

            if step % 1000 == 0:
                print("step {0}: loss = {1}".format(step, loss))

            if step % num_batches_per_epoch == 0:
                saver.save(sess, "./saved_model/model.ckpt", global_step=step)
                print("Epoch {0}: Model is saved.".format(step // num_batches_per_epoch))
