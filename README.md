# Problem Description

Train Seq2Seq model for test summarization based on Glove packages using TensorFLow.  

## Steps to run

1. Make sure you have the following packeges in your python 3:

	```
	tensorflow
	pickle
	nltk
	gensim
	```

2. Download raw data and trained Glove word2vec packeges to `GenSummary_Group2/` floder from:

	`[Raw data and Glove packages](https://pan.baidu.com/s/12MQ0G2yZrE63cTc3OQOcsA)`

3. Raw data has been processed and stored in `train/`, you can either use random word2vec:

	`python main.py`

	or use Glove(recommend) to run the model:

	`python main.py --glove`

4. Run `python test.py` to generate `train/result.txt` from trained model.

5. The predicted test titles(2657 lines) are stored in `train/result.txt`. Run `python eval_RougeL` to get the Rouge socre for the prediction.

## Data source and processing

1. Data set:

	Data is obtained from an open-source dataset from a conference NewsIR'16: http://research.signalmedia.co/newsir16/signal-dataset.html. The dataset contains 1,000,000 articles, mainly news report.

2. Preprocessing:
	
	Data preprocessing is done in data.py, mainly for eliminating escape strings and generating training and testing data. Run data.py before training(just once):
	
	`python data.py`
	
3. Training data size:

	80179 train articles, length:100-300

	2534 test articles, length:100-300

## Seq2Seq Model


### Model architecture

1. embedding:

	For faster training, We use GloVe for embedding. By GloVe every word is turned into a 300-d vector which shows some of the relationship among the words. We can only use the first `embedding_size` dimensions to make the training even faster. Since some of the words in the articles may not appear in GloVe, we also provide the way to train the embedding vector with random initialization.

2. encoder:

	We use BasicLSTMCell in training. The structure of the encoder is a bidirectional dynamic RNN where the state of the encoder is simply connect the state of forward and backward RNN.

3. decoder:
	
	We use Bahdanau Attention mechanism and rnn.BasicDecoder in our code. When training we use rnn.dynamic_decode to get the output and transpose it into logits. When testing we use BeamSearchDecoder and get the prediction.

4. loss:
	
	A weighted cross entropy is calculated as loss. We use AdamOptimizer to train our model.

### Training

1. Hyperparameters:

	```
	Network Size, default: 150
	Network Depth, default: 2
	Beam Width, default: 10
	Embedding Size, default: 300
	Learning Rate, default: 0.001
	Batch Size, default: 8
	Number of Epoch, default: 10
	Dropout Rate, used in encoder, default: 0.2
	```

2. Trainging process:

	a. Build Dictionary by GloVe and articles.

	b. Load training dataset.

	c. Initialize model and saver.

	d. Train with mini batch data, where parameters are:

		```
		batch_size = Batch Size (Hyperparameters)
		X = articles 
		X_len = length of each article
		decoder_input = <s> + summarys + paddings
		decoder_len = length of decoder input
		docoder_target = summarys + <\s> + paddings
		```

## Result

### Example

```
Article title: Original 'Ghostbusters' star added to reboot
Article: LOS ANGELES - Ernie Hudson, star on the original "Ghostbusters" movies, will appear in Sony s femme-reboot, Variety has learned. It s currently unclear whether Hudson is reprising his role as Winston Zeddemore. Sony declined to comment. Hudson joins returning Ghostbuster Bill Murray. The two starred alongside Dan Aykroyd and Harold Ramis in the 1984 film and its 1989 sequel. The movie, directed by Paul Feig, stars Kristen Wiig, Melissa McCarthy, Leslie Jones, Kate McKinnon and Chris Hemsworth. It hits theaters on July 22, 2

Predicted title: Original 'Ghostbusters star Ernie Hudson joins reboot
```

### Evaluation

We use Rouge-L method to evaluate the similarity of predicted titles between actual titles. 

Reference:`[Rouge paper](http://www.aclweb.org/anthology/W04-1013)`

Rouge-L evaluation result for our trained model:

```
2634 titles match!
Evaluation 0.0 % done
Evaluation 20.0 % done
Evaluation 40.0 % done
Evaluation 60.0 % done
Evaluation 80.0 % done
Evaluation 100.0 % done
Average Metric Score for All Review Summary Pairs:
Rouge: 8.102106789709532
```

## Contributors
* 2014010193 HowieLiang
* 2014012182 lujq96
* 2014012170 zxp14
* 2015011544 Jiawei