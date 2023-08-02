#!/bin/bash

python . --clean --adv --certify --experiment plain  
python . --clean --adv --certify --experiment logit  
python . --clean --adv --certify --experiment oe  
python . --clean --adv --certify --experiment acet  
python . --clean --adv --certify --experiment prood  
python . --clean --adv --certify --experiment atom  
python . --clean --adv --certify --experiment distro --batch_size 16

python . --clean --adv --certify --experiment plain --dataset cifar100
python . --clean --adv --certify --experiment logit --dataset cifar100
python . --clean --adv --certify --experiment oe --dataset cifar100
python . --clean --adv --certify --experiment acet --dataset cifar100
python . --clean --adv --certify --experiment prood --dataset cifar100
python . --clean --adv --certify --experiment atom --dataset cifar100
python . --clean --adv --certify --experiment distro --dataset cifar100

python . --guar --experiment plain --batch_size 1
python . --guar --experiment logit --batch_size 1
python . --guar --experiment oe --batch_size 1
python . --guar --experiment acet --batch_size 1
python . --guar --experiment prood --batch_size 1
python . --guar --experiment atom --batch_size 1
python . --guar --experiment distro --batch_size 1 

python . --guar --experiment plain --batch_size 1 --dataset cifar100
python . --guar --experiment logit --batch_size 1 --dataset cifar100
python . --guar --experiment oe --batch_size 1 --dataset cifar100
python . --guar --experiment acet --batch_size 1 --dataset cifar100
python . --guar --experiment prood --batch_size 1 --dataset cifar100
python . --guar --experiment atom --batch_size 1 --dataset cifar100
python . --guar --experiment distro --batch_size 1 --dataset cifar100

