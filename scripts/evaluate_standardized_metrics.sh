#!/bin/bash

python . --clean --adv --certify --experiment plain --standardized True 
python . --clean --adv --certify --experiment logit --standardized True 
python . --clean --adv --certify --experiment oe --standardized True 
python . --clean --adv --certify --experiment acet --standardized True 
python . --clean --adv --certify --experiment prood --standardized True 
python . --clean --adv --certify --experiment atom --standardized True 
python . --clean --adv --certify --experiment distro --standardized True --batch_size 16

python . --clean --adv --certify --experiment plain --standardized True  --dataset cifar100
python . --clean --adv --certify --experiment logit --standardized True  --dataset cifar100
python . --clean --adv --certify --experiment oe --standardized True  --dataset cifar100
python . --clean --adv --certify --experiment acet --standardized True  --dataset cifar100
python . --clean --adv --certify --experiment prood --standardized True  --dataset cifar100
python . --clean --adv --certify --experiment atom --standardized True  --dataset cifar100
python . --clean --adv --certify --experiment distro --standardized True  --dataset cifar100

python . --guar --experiment plain --standardized True --batch_size 1
python . --guar --experiment logit --standardized True --batch_size 1
python . --guar --experiment oe --standardized True --batch_size 1
python . --guar --experiment acet --standardized True --batch_size 1
python . --guar --experiment prood --standardized True --batch_size 1
python . --guar --experiment atom --standardized True --batch_size 1
python . --guar --experiment distro --standardized True --batch_size 1 

python . --guar --experiment plain --standardized True --batch_size 1 --dataset cifar100
python . --guar --experiment logit --standardized True --batch_size 1 --dataset cifar100
python . --guar --experiment oe --standardized True --batch_size 1 --dataset cifar100
python . --guar --experiment acet --standardized True --batch_size 1 --dataset cifar100
python . --guar --experiment prood --standardized True --batch_size 1 --dataset cifar100
python . --guar --experiment atom --standardized True --batch_size 1 --dataset cifar100
python . --guar --experiment distro --standardized True --batch_size 1 --dataset cifar100
