# AI: object-detection and image-classification

A ready-to-run object-detection cli tool dockerized together with the [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) model.

The reason for this project is to have a ready-to-use docker image which I can store on my drive and spin it up whenever I want to,
without the need for an internet connection.

See also [my AI list on github.com](https://github.com/stars/andreas-mausch/lists/ai).

# Build, run and save the image

```bash
docker build -t detr-resnet-50 .
docker run -it --rm --network none -v $PWD/images:/home/python/images:ro detr-resnet-50 "./images/**/*.*"
```

Note: I use docker's [none network driver](https://docs.docker.com/network/drivers/none/) to ensure everythings runs locally and no private data is exposed to the internet.
