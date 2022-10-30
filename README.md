# Classifying images of Salmon using the Vision Transformer (ViT) architechture

## Farmed vs Wild Salmon
Salmon is a very popular fish known for its great nutritional qualities. In nature, Salmon have to do a lot of very intense swimming to survive and are known to be swimming against the stream. This type of behavior builds a lot of high quality meat filled with Omega-3 fatty acids (aka fish oil) and other key nutrients that can be consumed by humans. However, due to the large demand for salmon, farming salmon has become a very popular method to produce more salmon for US consumers. Farmed salmon live in a very different habitat than wild salmon mostly differing in diet and activity. Farmed salmon cannot swim as much as wild salmon do and therefore they don't develop the same type of muscle tissue. When raw salmon is eaten it contains a much larger quantity of nutrients than farmed salmon. Farmed salmon can be distinguished from wild salmon due to the difference in color. Raw salmon has the unique red color that people have come to associated with the fish while farmed salmon looks more pink. In the US, it is common practice to feed farmed salmon red dye in their feed to try to achieve the same red color in their meat so that the consumer feels comfortable with consuming it. It would be useful for US consumers to know whether they are buying farmed or raw salmon so they are aware of the nutritional properties of the fish they are consuming. The purpose of this project is to use state-of-the-art algorithms and deploy a model that can distinguish wild salmon from farmed salmon.

## ViT Neural Network Architechture 
![ViT Architechture](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)
The Vision Transformer is a model that uses a Transformer-like approach for patches of an image. An image is split into fixed size patches, each of them linearly embedded, position embeddings are added, and the resulting vectors are to a typical Transformer encoder. For classification to take place, a an extra learnable "classification token" is added. ViT has been demonstrated to attain excellent results compared to state-of-the-art convolutional networks while requiring substantially less computational resources. 

## Repository structure

`AWS deployment` has the notebooks and scripts used to preprocess, train model, and deploy on AWS sagemaker

`Heroku Flask` has the python scripts to deploy model to Heroku using the model state dictionary 

`CPU-VIT.ipynb` the notebook to train model using pytorch locally and extract the state dictionary

# Deployment on AWS SageMaker

The folder for AWS deployment contains a notebook for image preprocessing and model training as well as 3 support scripts used in the notebooks. The preprocessing notebook needs to be run successfully before the training notebook can be used because output dataset will be used to train with.

### Image Preprocessing

Before we pass the data in to fine-tune our transformer model, we need to process it in SageMaker and store it in S3 first. For the preprocessing script to work, you will need to set up an S3 bucket in the following way (as well as adjust the naming in the notebook):
```
bucket    
│
└───train
   │   
   │   
   │
   └───class 1
      │   img1.img
      │   fimg2.jpg
      │  
      │
      └───class 2
      │   img1.img
      │   img2.jpg
```
The preprocessing script will build an appropriate dataset that can be used by the HuggingFace module when training. 

