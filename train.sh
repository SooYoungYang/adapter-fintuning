## Deep coral with adapter
CUDA_VISIBLE_DEVICES=0 python3 Deep_Coral+adapter.py  data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/coral/OfficeHome_Pr --adapter
## Deep coral without adapter
CUDA_VISIBLE_DEVICES=0 python3 Deep_Coral+adapter.py  data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/coral/OfficeHome_Pr
## Mixstyle with adapter
CUDA_VISIBLE_DEVICES=0 python3 mixstyle+adapter.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --mix-layers layer1 layer2 --seed 0 --log logs/mixstyle/OfficeHome_Pr --adapter
## Mixstyle without adapter
CUDA_VISIBLE_DEVICES=0 python3 mixstyle+adapter.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --mix-layers layer1 layer2 --seed 0 --log logs/mixstyle/OfficeHome_Pr
## Group DRO with adapter
CUDA_VISIBLE_DEVICES=0 python3 groupdro+adapter.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/groupdro/OfficeHome_Pr --adapter
## Group DRO without adapter
CUDA_VISIBLE_DEVICES=0 python3 groupdro+adapter.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/groupdro/OfficeHome_Pr
