import tensorflow as tf
import pickle
from main import Summodel
from data import build_dict, build_dataset, batch_iter


with open("args.pickle", "rb") as f:
    args = pickle.load(f)

print("Loading dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid")
print("Loading validation dataset...")
valid_x, valid_y = build_dataset("valid", word_dict, article_max_len, summary_max_len)
valid_x_len = list(map(lambda x: len([y for y in x if y != 0]), valid_x))

with tf.Session() as sess:
    print("Loading saved model...")
    model = Summodel(reversed_dict, article_max_len, summary_max_len, args, Forward_only=True)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state("./saved_model/")
    saver.restore(sess, ckpt.model_checkpoint_path)

    batches = batch_iter(valid_x, valid_y, args.batch_size, 1)

    print("Writing summaries to 'train/result.txt'...")
    for batch_x, batch_y in batches:
        batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))

        valid_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
        }

        prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
        prediction_output = list(map(lambda x: [reversed_dict[y] for y in x if y > 0], prediction[:, 0, :]))

        with open("train/result.txt", "a") as f:
            for line in prediction_output:
                summary = list()
                for word in line:
                    if word == "</s>":
                        break
                    if word not in summary:
                        summary.append(word)
                print(" ".join(summary), file=f)

    print('Summaries are saved to "train/result.txt"...')