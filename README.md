


many of the notebooks depend on the pretrained vgg16 and resnet50 fine-tuned on upscaled cifar10. For size reasons .h5 files are not tracked on this repo. If cloning you should download the .h5 files from google drive at the following links

- vgg16 https://drive.google.com/open?id=1856dzQMaqG9FKWwXokm0NTC-kH60rQGY
- resnet50 https://drive.google.com/open?id=18g14OTK7aK3JN3lRmFQlVFovCOr30nBz

bash scripts are contained in is repo to build the docker image, start the docker container and start the jupyterlab instance needed for this project.

bash scripts need to be run with sudo permissions.


