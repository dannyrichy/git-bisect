# git-bisect
## About
Report: [Git Bi-sect.pdf]
Project for course DD2412.
 - Implementation of https://arxiv.org/pdf/2209.04836.pdf in pytorch
 - Dataset used: CIFAR10, MNIST

## Code structure
- [permuter]
  - [_algo.py] Core Method to permute
  - [common.py] Common functions
  - [vgg.py] files specific to vgg
  - [mlp.py] files specific to mlp
- [models] contains class definition of models: [MLP] and [VGG]
- [stash] contains trained models
- [battle_ground.py] is as the name suggests where experiments are carried out 💚
- [config.py] contains configurations
- [helper.py] contains helper functions


[models]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/models
[permuter]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/permuter
[_algo.py]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/permuter/_algo.py
[common.py]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/permuter/common.py
[vgg.py]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/permuter/vgg.py
[mlp.py]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/permuter/mlp.py
[permuter]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/permuter
[stash]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/stash
[battle_ground.py]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/battle_ground.py
[config.py]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/config.py
[helper.py]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/helper.py
[MLP]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/models/mlp.py
[VGG]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/models/vgg.py
[Git Bi-sect.pdf]: https://github.com/the-nihilist-ninja/git-bisect/blob/master/Git%20Bi-sect.pdf

