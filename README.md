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

Before we pass the data in to fine-tune our transformer model, we need to process it in SageMaker and store it in S3 first. For the preprocessing script to work, you will need to set up an S3 bucket in the following way (as well as adjust the names in the notebook):
```
bucket    
│
└───train
   │   
   │   
   │
   └───class 1
      │   img1.jpg
      │   img2.jpg
      │     ......
      │
      └───class 2
      │   img1.jpg
      │   img2.jpg
      |      ......
```
The preprocessing script will build and store an appropriate dataset in that can be used by the HuggingFace module when training.

### AWS SageMaker Deployment

There are two ways we can train our ViT model. The first way is to use the HuggingFace API with the accompanyting HuggingFace trainer script. However, AWS currently does not support containers to deploy HuggingFace models for image classification yet. Therefore, we will instead have to use PyTorch to deploy this model. This can be done with the provided Pytorch Lightning training script. You will have to contact AWS support if you want to use a GPU instance to train (highly recommended). 

After we deployed our model to a real-time inference endpoint, we can use AWS Lambda to trigger an event whenever an image is uploaded to S3 to pass that image through to our endpoint. However, there seems to be some compatibility issues with AWS regarding creating an inference endpoint from HuggingFace models for image classification. In order to clear up this issue we would need to dig deeper into how to pass an image into this type of model. We have to move to a different platform to deploy our model. 

## Heroku Deployment using Flask with Local Training


To deploy our model on a open-source platform (Heroku, Streamlit, etc.) we first need to train our model in a local environment to be able to deploy the model. The neural network architechture is defined in the block below: 

![NNarchitechture](https://i.imgur.com/cU3xzZM.png)

And the data was passed through with the following code block: 

![NNpass](https://i.imgur.com/oGbwvk7.png)

After training the data, it is a good idea to save both the model and the model state dictionary. We will use the model state dictionary later when we load the model in the Flask app. The scripts to deploy the flask app can be found in the apporpriate directory. 

