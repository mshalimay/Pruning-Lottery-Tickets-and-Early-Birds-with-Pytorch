# (Un)Structured Pruning, Lottery-Tickets and Early-Birds with Pytorch

This codebase implements:
- Iterative and one-shot unstructured pruning for general models
- One-shot structured pruning for `ResNet` model variations, following ...
- Lottery-Ticket identification with unstructured pruning, as in ...
  - Including lottery-ticket initialization
- Lottery-Ticket identification with structured pruning.
- Early identification of Lottery Tickets (Early-Bird) analysis
  - The codebase allow for evaluation of EBs; see `Examples.md` section 3
  - **TODO**: implement EB identification via mask difference

# Codebase structure
- `main.py`: main driver for training and pruning
- `prune.py`: contains functions for unstructured and structured pruning
- `./models/`: neural network architectures
- `./utils/`: auxiliary functions for training and saving
- `./scripts/`: bash scripts to replicate each of the experiments results

# Usage

## General usage
`main.py` contains code to:
- train a neural network from scratch 
- prune a trained neural network with:
	- one-shot channel-wise structured pruning as in [cite] 
	- one-shot and iterative unstructured pruning 

For the full set of options, run 
```bash
python main.py -h
```

**Main options for this project**
- `--pr`: overall pruning ratio (e.g.: 0.5, 0.7, 0.9)

- `--priter`: the number of pruning iterations. 
	- 0: train dense network only. 1: one-shot pruning.  >1, iterative pruning

- `--model`: resnet models:
	- `resnetb`: with bottleneck design
	- `resnet`: without bottleneck design

- `--depth`: depth of `resnet` model. 20: `resnet20`

- `--log-train`: log training data for all models trained
	- logs are saved in `./training-log/.....`

- `--save`: save checkpoints for all models trained
	- models are saved at `./checkpoints/<.......>`

- `--epochs-save`: save checkpoints at specific epochs

- `--prunety`: pruning strategy (`u`= unstructured, `s`=structured)

- `--load <checkpoint_path>`: load model states from `checkpoint_path`. 
	- If `checkpoint_path` does not contain any, will try to load from the default path to save checkpoints. If does not find, train a model from scratch

## Usage Examples
### Training - no pruning
Examples of how to use training parameters. Apply to all pruning cases, but we use no pruning to emphasize the options.

To train without pruning, set # of pruning iterations to 0: `priter=0`
#### Training and saving

**Save checkpoint and log training**
Train a `resnet20` with bottleneck design from scratch using `cifar10` dataset. Save best checkpoint and training information.
```bash
python main.py --model 'resnetb' --depth 20 --priter 0 --epochs 160 --dataset cifar10 --save --log-train 
```

**Save at specific epochs**
Train a `resnet20` without bottleneck design from scratch using `cifar10` dataset. Save best checkpoint and training information. 

Save checkpoints at epochs [10, 30, 50, 100]. Two checkpoints saved: the checkpoints at these exact epochs; best state for epoch <= each of these epochs. 

```bash
python main.py --priter 0 --save --log-train --model resnet --depth 20 --dataset cifar10 --epochs-save 10 30 50 100
```

#### Tweaking hyperparameters

**Optimizer tweaking**
Train default model/dataset from scratch with:
- `sgd` optimizer, `weight-decay = 5e-4`, initial learning rate of `0.05`, `momentum` 0.5
```bash
python main.py --priter 0 --o 'sgd' --w-decay 5e-4 --lr 0.05 --momentum 0.5
```

- `adadelta` optimizer, `weight-decay = 5e-4`, initial learning rate  `0.1`
```bash
python main.py --priter 0 --o 'adadelta' --w-decay 5e-4 --lr 0.1
```

**Scheduler tweaking**
Train default model/dataset from scratch with:
- `manual` scheduler: reduces the learning rate by a factor of `0.2` (new_lr = old * 0.2) at epochs [80, 120]
```bash
python main.py --priter 0 --sched 'manual' --lr 0.1 --gamma 0.2 --lr-update-epochs [80, 120]
```

- No scheduler: 
```bash
python main.py --priter 0 --sched 'none'
```

- `ReduceLROnPlateau` scheduler: reduces learning rate by a factor of `0.1` if test_loss does not improve for 10 epochs
```bash
python main.py --priter 0 --sched 'plateau' --gamma 0.1
```

- other options please run `python -h`

### Training and one-shot unstructured pruning
Trains model from scratch and perform one-shot pruning at 70% rate. Save checkpoints and training information for all (2) models.
```shell
python main.py --priter 1 --pr 0.7 --save --log-train --prunety 'u'
```

Load a pre-trained dense model perform one-shot pruning at 70% rate. Save checkpoints for and training information for pruned model.
```shell
python main.py --priter 1 --pr 0.7 --load 'auto' --save --log-train --prunety 'u'
```

### Training and iterative unstructured pruning

**train from scracth and prune**
Trains model from scratch and perform iterative pruning at 70% rate with 4 rounds. Save checkpoints and training information for all (5) models.

```shell
python main.py --priter 4 --pr 0.7 --save --log-train --prunety 'u'
```

**load and prune**
Load a pre-trained dense model and perform iterative pruning at 70% rate with 4 rounds. Save checkpoints for and training information for all (4) pruned models.

```shell
python main.py --priter 4 --pr 0.7 --load 'auto' --save --log-train --prunety 'u'
```

**load and prune with reinitialization**
Load a pre-trained dense model, perform iterative pruning at 70% rate with 4 rounds. 


At **each pruning iteration**: 
- reinitialize the parameters to the **initial** parameters used to train the pre-trained dense model.
```shell
python main.py --priter 4 --pr 0.7 --load 'auto' --save --log-train --prunety 'u' --reinit 'init'
```
- randomly initialize weights
```shell
python main.py --priter 4 --pr 0.7 --load 'auto' --save --log-train --prunety 'u' --reinit 'random'
```

### Training and structured pruning
Train dense model from scratch and perform one-shot structured pruning at 70%. Save best checkpoints and training logs for all (2) models.

```bash
python main.py --priter 1 --pr 0.7 --save --log-train --prunety 's'
```

# References
1. Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. _arXiv preprint arXiv:1803.03635_.
2. Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural network. _Advances in neural information processing systems_, _28_.
3. Liu, Z., Li, J., Shen, Z., Huang, G., Yan, S., & Zhang, C. (2017). Learning efficient convolutional networks through network slimming. In _Proceedings of the IEEE international conference on computer vision_ (pp. 2736-2744).
4. Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2018). Rethinking the value of network pruning. _arXiv preprint arXiv:1810.05270_.
5. You, Haoran, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Richard G. Baraniuk, Zhangyang Wang, and Yingyan Lin. (2019). Drawing early-bird tickets: Towards more efficient training of deep networks. 

## Other references
https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch
https://github.com/GATECH-EIC/Early-Bird-Tickets

