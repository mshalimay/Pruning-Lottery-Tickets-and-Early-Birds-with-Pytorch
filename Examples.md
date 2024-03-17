# Sample experiments
Three examples of experiments using the codebase are provided below:
- 1): One-shot unstructured pruning, retraining and comparison with full network.
  - Answers to: Is it worth pruning? How much can we prune and recover accuracy (after retraining)? (a lot!)

- Experiment 2: Iterative unstructured pruning with lottery ticket initialization; retraining and comparison with full network.
  - Replicates results from 'The Lottery Ticket [[1]](#references).
  
- Experiment 3: `Resnet-20` structured at selected epochs; retraining and comparison with full network.
  - Can we identify lottery tickets early in training, instead of training the full network, prune and retrain? (Yes!)
    - Example: train for 20 epochs, prune, and retrain for full number of epochs. If accuracy comparable to full network, lottery ticket identified earlier in training.

# General settings for experiments
Below the settings that applies for training of all experiment variants (i.e., training of dense and pruned networks)

**Number of epochs**: 160

**Batch size:** 64

**Optimizer**: `sgd`, with `weight-decay` = 1e-4, nesterov momentum = 0.9

**Learning rate and Scheduler**: initial learning rate of 0.1. `manual` scheduling, reducing LR by a factor of $0.1$ at epochs 80 and 120

**Model**: `ResNet20` with bottleneck design  (`resnetb`, `depth = 20`)

# Experiment 1: Unstructured Pruning Analysis. Is it worth pruning?
To replicate results, run
```bash
./scripts/experiment1.sh
```

### Methodology
- Weights are randomly initialized and the full network is trained following the general settings specified above
- The network is then pruned in **one-shot** by the ratio $r$. 
	- That is, pruning is done in one step based on the weights achieved in the best model state. 
- During pruning, only convolutional layer weights are considered
- After pruning, the pruned network is retrained for the full 160 epochs starting from the best weights for the full network.
- During training of pruned network, gradients of the pruned weights are frozen (i.e., multiplied by the masks)

### Results and comments
Below table and plots show test accuracy and loss results/trajectories for the three pruning ratio ($pr$) sets.

**Comments**:
- The dense network achieves an accuracy of 91.6%.
- After pruning 50%, 70% and 90%, the accuracy of the dense network falls to 83%, 40.7% and 10.1% respectively.
- Nonetheless, for all three cases it was possible to recover the almost all of the accuracy with retraining, with a maximum loss of ~2.4% when the pruning ratio is 90%
- For 50% and 70% pruning, the best state was achieved earlier than the
- For the 90% case, we see that learning slows down and stagnates earlier after epoch 80 when compared to the other two cases. 
	- This Illustrates the phenomena that too sparse networks can have more difficult to learn.

- For the pruning ratios and strategy utilized, the better performance was achieved by the network with 50% of the weights, which equaled the accuracy of the dense network with 50~70% less weights.
  - Nonetheless, it is worth noting that training **hyperparameters configs remains fixed in all cases**. For instance, initial learning rates and schedules are the same irrespective of the pruning ratios. And as suggested in [[4]](#references), adopting optimal learning rates might be important for the capacity of the sparser networks to learn
  - Indeed, as commented above, for the more sparse networks the learning trajectories suggest slower learning and more potential for improving

![Alt text](<imgs/Pasted image 20240212231120.png>)

![Alt text](<imgs/Pasted image 20240212231638.png>)

![Alt text](<imgs/Pasted image 20240212231648.png>)

# Experiment 2 - Iterative Unstructured Pruning and Lottery Tickets. 
To replicate results, run
```bash
./scripts/experiment2.sh
```
### Methodology
- Weights are randomly initialized to $w_{init}$. 
- The full network is trained following the general settings specified above. The best model weights $w_{0}$ are saved.
- Pruning - one-shot case (step 3):
	- Pruning masks are created based on $w_{0}$, removing $r\%$ of convolutional weights (r in {50, 70, 90})
	- Weights are reinitialized to $w_{init}$ and the masks are applied
	- The pruned network is retrained for 160 epochs
- Pruning - iterative case (step 4):
	- Pruning occurs in 4 iterations. 
	- In each iteration,  ($p=1-(1-pr)^{(1/4)}$)% of the **alive** weights are pruned.
	- At iteration $t$, a $mask[t]$ to prune the network is computed based on the weights from $w_{t-1}$ (including $w_0$)
		- $mask$ is tensor of 0's and 1's indicating which weights have L1 norm below the $p$ percentile among all alive convolutional weights.
	- $mask[t]$ is "accumulated"  with the previous masks: if  weight was pruned before, remains pruned, even if above threshold. 
		- Mathematically: pointwise multiple the current mask and the (cumulative) mask from previous iteration
	- Weights are reinitialized to $w_0$ and the network is pruned with the current $mask[t]$
	- The pruned network is trained for 160 epochs and the process repeats until total pruning of $pr\%$  is achieved.
- For all pruning: 
	- only weights of convolutional layers are pruned
	- during training of pruned network, gradients of the pruned weights are frozen (i.e., multiplied by the masks)

### Results and comments
The table below shows the test accuracy and loss for ResNet20 at the various pruning ratios and schemes. 

The subsequent plots show the trajectories for all cases separated by the pruning ratios. They include the trajectories of the subnetwork in the one-shot pruning scenario and trajectories for the subnetworks at each pruning iteration.

![Alt text](<imgs/Pasted image 20240212224003.png>)

**Comments**
1) On final accuracy of pruned subnetworks:
- For pruning ratios of ~50%, all cases tend to recover the performance of the original, dense network, even after the lottery-ticket initialization.
- For pruning ratios of >=70% , we start to see some degradation in the performance of the smaller subneworks. 
- For 90%, there is roughly 1~2% reduction at earlier stages and ~6-7% reduction after full pruning
- This showcases:
	- The existence of lottery tickets: subnetworks with as much as 50% reduced size can present very close performance to the dense counterpart.
	- Nonetheless, overpruning might turn more difficulty to identify such tickets and hinders performance
	- With effect, in the plots we see that as pruning increases overpruned networks have a harder time to learn and stagnates earlier. 
		- This can be seen by either the curves at different pruning ratios or by comparing the curves at different  iterations of iterative pruning
	- Nonetheless, it is important to note that training **hyperparameters configs remains fixed in all cases**. For instance, initial learning rates and schedules are the same irrespective of the pruning ratio/iteration. And in line with [[4]](#references), this might affect the capacity of the sparser networks to learn.
2) On the accuracies of the networks at different weight states
- In the table below, the @w0 columns show the accuracies of the pruned networks evaluated at the best weight state of the full network ($w_0$) (or equivalently: the full network after applying the masks at each $pr$ / pruning iteration)
- We see that for higher $pr$ and pruning iterations, the accuracy of the pruned network evaluated at $w_0$ is not better than chance
- This is different from Task 2 and might be explained by the fact that, after pruning, we re-initialize to the random state (lottery ticket initialization) and re-train the network from there. As a result, the final network structure found (and thus the masks) are less coupled with the original network optimal structure
	- Nonetheless, this result might be sensible to the small number of pruning iterations; amortizing pruning over more steps could lead to a more smooth dissociation from the original network best state. 
- For iterative pruning, I also included the evaluation  at the state in the previous pruning iteration (columns @w(t-1)), which is a more comparable metric to the one we looked at in Task 2. We see a pattern that is closer to before: the pruned network presents a similar accuracy than before pruning, unless the pruning ratio is too high (such as the last iteration with $pr=90\%$) 
	
![Alt text](<imgs/Pasted image 20240212231648-1.png>)
![Alt text](<imgs/Pasted image 20240212202632.png>)
![Alt text](<imgs/Pasted image 20240212202702.png>)

**Answers to common questions**
Difference between experiment 1 experioment 2?
- Two differences:
	- Here we **reinitiliaze** the weights before retraining to the same initial state the full network had before training ($w_{init}$ explained in the methodology). As discussed comments, this changes the convergence trajectories and the accuracy of the original network when the masks are applied to it (since the final weights for pruned network and hence masks are more decoupled from the original best state)
	- We also use **iterative pruning**: instead of applying the prune in one step, we amortize the pruning in $n$ steps, as explained in the methodology.

# Experiment 3 - Structured Pruning and Early Lottery Ticket Identification
To replicate results, run
```bash
./scripts/experiment3.sh
```

## Methodology
- Weights are randomly initialized to $w_{0}$. 
- The full network is trained following the general settings specified above. 
- During training, the best weights $w_{epoch}$ obtained in 10, 30, 50, 100 and 160 epochs are stored. 
	- I also store the weights at the specific epochs, but results changed little by using them.
- The network is pruned based on $w_{epoch}$ and retrained for the full 160 epochs
	- I also add an alternative to retrain from the epoch it stopped (example, retrain from the $10th$ epoch)
- Pruning is done structurally, following [[3]](#references):
	- Initially, only batch normalization layers are considered
	- the $pr\%$ channels with smaller scaling factors are removed from the network 
		- if at any point this leads to a convolutional layer with 0 channels, one channel is added (did not happen during experiments)
	- the network is reconstructed considering the new structure and is trained for 160 epochs

## Results and comments

The table below shows the test accuracy and loss for ResNet20 at the various pruning ratios and epochs. 

The subsequent plots show the trajectories for all cases separated by the pruning ratios. They include the trajectories of the subnetwork in the one-shot pruning scenario and trajectories for the subnetworks at each pruning iteration.

![Alt text](<imgs/Pasted image 20240212233224.png>)


**Comments**
- In all cases, the retrained networks were able to recover the accuracy of the dense counterparts. This was true even for networs pruned at the epoch 10 with $pr=30\%$
- With higher $pr=50\%$, a higher amount of epochs was necessary to recover more closely the performance of the original network
	- This is in line with the observation in the early birds paper [[5]](#references) discussed in the summary
- Similar to Task 3, we see that the pruned networks evaluated at the best state of the full network show accuracy close to random guess especially for higher $pr$. Two comments:
	- Here this happens as soon as $pr=50$ and might be explained by the fact that the **structure** of the network is changed after pruning, besides the weight. So more chances for the last state (and hence masks) to decouple from the original best state
	- Accordingly, networks pruned at later epochs show less of this behavior, potentially due to more overlap with the original best state
- Which epoch is the best to stop training?
	- In terms of accuracy, epoch=100 typically showed higher accuracy in the experiments. However, the difference is very small to the case epoch=10 when the pruning ratio is small.
	- If the pruning ratio is higher, as discussed above, allowing for more epochs seems to increase the likelihood of drawing a good subnetwork


**Answer to common questions**
- Should we prune the weights during the early training stage?
	- Given the results and discussion above, yes. We can achieve very similar accuracy by pruning the networks at very early stages and this leads to faster training overall. But an important caveat: if the pruning ratio is high, it might be more secure to prune at later stages.
- Is there a way to automatically identify when to prune?
	- Yes, by using the hamming distance of masks approach, as discussed in [[5]](#references).
		- Particularly, we proceed with the training as we did here, but keeping track of (i) the masks at each (candidate) epoch (ii) the hamming distance between subsequent masks.
		- If at some point a pre-determined number of masks (5 in the EB implementation)  is too close to each other (i.e., distance is smaller than a threshold) we identify that as an EB and potential lottery ticket associated to pruning ratio $pr$. We prune the network and train from there.


![Alt text](<imgs/Pasted image 20240212232251.png>)
![Alt text](<imgs/Pasted image 20240212232304.png>)
![Alt text](<imgs/Pasted image 20240212232402.png>)
![Alt text](<imgs/Pasted image 20240212232345.png>)


# References
1. Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. _arXiv preprint arXiv:1803.03635_.
2. Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural network. _Advances in neural information processing systems_, _28_.
3. Liu, Z., Li, J., Shen, Z., Huang, G., Yan, S., & Zhang, C. (2017). Learning efficient convolutional networks through network slimming. In _Proceedings of the IEEE international conference on computer vision_ (pp. 2736-2744).
4. Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2018). Rethinking the value of network pruning. _arXiv preprint arXiv:1810.05270_.
5. You, Haoran, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Richard G. Baraniuk, Zhangyang Wang, and Yingyan Lin. (2019). Drawing early-bird tickets: Towards more efficient training of deep networks. 
