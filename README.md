# Meta-Learning for Few-Shot Class-Incremental Learning
_^ That's ridiculous_

## What do we want (black-box view)?
A system that:
- Can repeatedly have an arbitrary number of new classes added, by only exposing a few images
- Suffers minimal catastrophic interference

## Which techniques are a must?
- Meta-learning: Utilise the relationship between train/test splits to learn optimal training methods for few-shot
- Few-shot growth: A minimal number of images required for class-extension

## High-Level Approach
1. Train a bunch of models on sub-sets of the training classes
2. Added classes to them (??)
3. Have the meta-learner observe/control steps 1 & 2
