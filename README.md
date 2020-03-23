# Memorisation-and-Generalisation-in-Deep-CNNs-Using-Soft-Gating-Mechanisms
Simplified version of the repository GREYC-Internship, contains only the code for the model that surpassed baseline and also contains code to qualitatively prove the hypothesis.

### Project Overview 

A deep neural network learns patterns to hypothesize a large subset of samples that lie in-distribution and it memorises any out-of-distribution samples. While fitting to noise, the generalisation error increases and the DNN performs poorly on test set. In this work, we aim to construct a network that combines the strengths of both memorisation and generalisation in a single neural network. While the initial layers that are common to all examples tend to learn general patterns, we relegate certain deeper additional layers in the network to memorise the out-of-distribution examples. The proposed model uses a soft gating mechanism to decide on the fly if an input will skip the additional layers or pass through them based on its hardness measure. An entropy based metric is used to assign hardness to each example.

### Prerequisites 

* Python 3.6+
* Pytorch 1.0+
* Tensorboard for Pytorch 


### Sample command to run code and set flags  

> CUDA_VISIBLE_DEVICES=0 python main_gate_classifier_val.py --dataset='cifar10' --depth=110 --block-name='BasicBlock' --scratch=0 --lr=0.1 --lr_gate=0.01 --mod_name_suffix='it1g-2cm-1' --gate_iters=1 --schedule_gate=0
  
  
### Few Pointers 

1. The dataset gets downloaded automatically if it is not present already
2. The code is flexible wrt to resnet type, dataset type (cifar10/100) and most hyper parameters; the same code can be used to train a range of models by passing the correct training arguments  
3. For each model trained on different Resnet/dataset/initialisation strategy/other hyper parameter combinations, the code differentiates between checkpoints/logs of different models by using model specific names where the checkpoints/logs get saved
4. Any model can be tested on the trained checkpoints by passing the ‘-e’ argument and specifying the absolute path in the ’test_checkpoint’ argument 
5. The places where directories can be specified have been marked by ' #** ' symbol. These include data directory, baseline directory, tensorboard log and checkpoint saving directory. The only directory that needs to be created is for saving checkpoints (check line 414 in main_gate_classifier_val.py).
6. This code uses a 45k/5k train/validation dataset split for training and can be used for hyper parameter tuning. 
