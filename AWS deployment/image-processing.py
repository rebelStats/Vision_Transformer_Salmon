import argparse, os, subprocess, sys, ast, pickle, boto3
import numpy as np
from PIL import Image
from io import BytesIO

def pip_install(package):
    subprocess.call([sys.executable, '-m', 'pip', 'install', package, '--upgrade'])

pip_install('transformers')
from transformers import ViTFeatureExtractor

pip_install('datasets')
from datasets import Dataset, Features, ClassLabel, Array3D

s3_client = boto3.client('s3')

def s3_object_to_numpy_array(bucket, key):
    image = s3_client.get_object(Bucket=bucket, Key=key)['Body'].read()
    image = Image.open(BytesIO(image))
    image = image.resize((224,224))
    image = np.array(image, dtype=np.uint8)
    image = np.moveaxis(image, source=-1, destination=0)   # channels first for PyTorch
    return image

def get_image_label(key):
    classname = key.split('/')[1]
    label = classes.index(classname)
    return label

def build_image_dict(bucket, prefix):
    image_dict = {}
    image_dict['img'] = []
    image_dict['label'] = []
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    count=0
    for page in pages:
        for obj in page['Contents']:
            image = s3_object_to_numpy_array(bucket, obj['Key'])
            label = get_image_label(obj['Key'])
            image_dict['img'].append(image)
            image_dict['label'].append(label)
    return image_dict

def preprocess_images(batch):
    images = batch['img']
    images = [np.array(image, dtype=np.uint8) for image in images]
    inputs = feature_extractor(images)
    batch['pixel_values'] = inputs['pixel_values']
    return batch

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--classes', type=str)
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k')

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    bucket = args.bucket
    prefix = args.prefix
    classes = ast.literal_eval(args.classes)
    
    # Load images in a Python dictionary
    print('Loading images...')
    image_dict = build_image_dict(bucket, prefix)
    
    # Load dictionary in Hugging Face dataset
    print('Building data set...')
    features = Features({
        'label': ClassLabel(names=classes),
        'img': Array3D(dtype='int64', shape=(3,224,224))
    })
    dataset = Dataset.from_dict(image_dict, features=features)
    
    # Extract image features and add them to dataset
    print('Extracting features...')
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model_name)
    features = Features({
        'label': ClassLabel(names=classes),
        'img': Array3D(dtype='int64', shape=(3,224,224)),
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224))
    })
    preprocessed_dataset = dataset.map(preprocess_images, batched=True, features=features)
        
    print('Splitting data set...')
    splits = preprocessed_dataset.train_test_split(test_size=0.2)
    train_dataset = splits['train']
    splits = splits['test'].train_test_split(test_size=0.5)
    valid_dataset = splits['train']
    test_dataset = splits['test']
    
    print(train_dataset)
    print(valid_dataset)
    print(test_dataset)

    print('Saving data sets...')
    train_dataset.save_to_disk('/opt/ml/processing/train')
    valid_dataset.save_to_disk('/opt/ml/processing/valid')
    test_dataset.save_to_disk('/opt/ml/processing/test')