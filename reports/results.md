| model                               |   accuracy |   macro_f1 |   params |   trainable_params |   train_seconds |   ms_per_image |
|:------------------------------------|-----------:|-----------:|---------:|-------------------:|----------------:|---------------:|
| Small CNN (from scratch)            |     0.9691 |     0.9685 |   584170 |             584170 |          2963.5 |          3.752 |
| ResNet18 linear probe (frozen)      |     0.9175 |     0.9141 | 11181642 |               5130 |           109.8 |        nan     |
| ResNet18 fine-tuned (ImageNet init) |     0.9862 |     0.9858 | 11181642 |           11181642 |          1121.6 |          4.659 |

| perturbation    |   ResNet18 fine-tuned (ImageNet init) |   Small CNN (from scratch) |
|:----------------|--------------------------------------:|---------------------------:|
| blur k=3        |                                0.9462 |                     0.6086 |
| blur k=5        |                                0.8175 |                     0.3805 |
| brightness +0.2 |                                0.9338 |                     0.662  |
| clean           |                                0.9862 |                     0.9691 |
| noise σ=0.05    |                                0.1709 |                     0.2128 |
| noise σ=0.1     |                                0.1153 |                     0.1402 |
| noise σ=0.2     |                                0.1684 |                     0.1126 |
