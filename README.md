This Repo is for synthetic data generation for layerwise convolutional neural network compression


many of the notebooks depend on the pretrained vgg16 and resnet50 fine-tuned on upscaled cifar10. For size reasons .h5 files are not tracked on this repo. If cloning you should download the .h5 files from google drive at the following links

vgg16 - https://drive.google.com/open?id=1856dzQMaqG9FKWwXokm0NTC-kH60rQGY
resnet50 - https://drive.google.com/open?id=18g14OTK7aK3JN3lRmFQlVFovCOr30nBz

## Make Baseline Experiments

- [ ] training with data
- [ ] training with dataset that overlaps classes
- [ ] training with dataset that minimally overlaps classes
- [ ] training on random generated images

## Experiments

- [x] naive maximization
- [ ] augmentation tricks
- [ ] diversity layerwise
- [ ] gradient decent (what not to activate for)
- [ ] gan?

## Related Work

- [ ] Layer/Blockwise comrpession
- [ ] semi supervised training
- [ ] GAN knowledge distillation
