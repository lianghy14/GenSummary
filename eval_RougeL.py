
# coding: utf-8
import glob
import numpy as np
import pdb

def my_lcs(string, sub):
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

def calc_score(refs, hypothesis):
    # split into tokens
    token_c = hypothesis.split(" ")
    token_r = refs.split(" ")
    # compute the longest common subsequence
    lcs = my_lcs(token_r, token_c)
    prec_max = lcs/len(token_c)
    rec_max = lcs/len(token_r)
    beta = 1.2
    if(prec_max!=0 and rec_max !=0):
        score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + (beta**2)*prec_max)
    else:
        score = 0.0
    return score

if __name__ == '__main__':
    # Feed in the directory where the hypothesis summary and true summary is stored
    reference = open('train/test_title.src')
    hypothesis = open('train/result.txt')
    ref = []
    hyp = []
    for line in reference:
        ref.append(line.strip())
    for line in hypothesis:
        hyp.append(line.strip())
    reference.close()
    hypothesis.close()
    if len(ref) == len(hyp):
        print(len(ref),'titles match!')
    else:
        print('Titles dont match!')
            
    ROUGE_L = 0.
    num_titles = 0
    step = int(len(ref)/5)
    for i in range(len(ref)):
        if i%step == 0:
            print('Evaluation',i/step*20,'% done')
        num_titles += 1
        score_i = calc_score(ref[i], hyp[i])
        #print(score_i)
        ROUGE_L += score_i

    print ('Average Metric Score for All Review Summary Pairs:')
    print ('Rouge:', ROUGE_L/num_titles*100)