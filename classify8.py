# -*- coding: utf-8 -*-
"""classify8.ipynb

Automatically generated by Colaboratory. I used Google Colab because my local machine did not have the power
to run torch or torchvision. I have included it in this repository to showcase my code.

Original file is located at
    https://colab.research.google.com/drive/1F0ZZeBAj324ohibT71KZlI-vsNtZSHDn
"""
""" 
# Required packages for model conversion and other utilities are installed
!pip install onnx
!pip install onnx_tf

# Import necessary libraries
import torch
import torchvision
import shutil
import os
import tensorflow as tf

# Define a Dataset class for loading and preprocessing data
class Dataset:
    @classmethod
    def load_dataset(cls):
        # Transforms to resize and normalize images
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Loading the Flowers102 dataset for training and testing
        train_dataset = torchvision.datasets.Flowers102(
            root='data',
            split='train',
            transform=transforms,
            download=True
        )

        test_dataset = torchvision.datasets.Flowers102(
            root='data',
            split='test',
            transform=transforms,
            download=True
        )

        return train_dataset, test_dataset

    @classmethod
    def create_dataloaders(cls):
        # Utilizing the load_dataset method to get train and test datasets
        train_dataset, test_dataset = cls.load_dataset()

        # Creating data loaders for batching, shuffling, and parallel loading
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=True
        )

        return train_loader, test_loader

# Define a classifier for the OxFordFlowers dataset
class OxFordFlowersClassifier:
    @classmethod
    def create_model(cls):
        # Using a pretrained ConvNeXt Tiny model and modifying its last layer to match our class number
        model = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,
            progress=True,
            num_classes=1000
        )

        # Modify the classifier's final layer to have 102 output features (for 102 flower classes)
        model.classifier[2] = torch.nn.Linear(
            in_features=768,
            out_features=102,
            bias=False
        )

        return model.cuda()

    @classmethod
    def train_validate(cls):
        train_loader, test_loader = Dataset.create_dataloaders()
        best_accuracy = 0.0
        best_model = None

        # Create and move the model to the GPU
        model = cls.create_model()
        model = model.to(torch.device('cuda'))

        # Define the loss function and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=3e-4
        )

        epochs = 10
        # Training and validation loop
        for epoch in range(epochs):
            model.train()
            for batch, data in enumerate(train_loader):
                x, label = data
                x, label = x.to(torch.device('cuda')), label.to(torch.device('cuda'))  # Move data to GPU
                
                pred = model(x)
                loss = loss_fn(pred, label)

                # Backpropagation step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch} Batch {batch} Loss {loss}")

            # Evaluation on test data
            num_correct = 0
            num_samples = 0
            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    x, label = data
                    x, label = x.to(torch.device('cuda')), label.to(torch.device('cuda'))  # Move data to GPU

                    pred = model(x)
                    pred_labels = torch.argmax(pred, axis=1)

                    num_correct += (pred_labels == label).sum().item()
                    num_samples += pred_labels.size(0)

                accuracy = float(num_correct) / float(num_samples)
                print(f"Accuracy {accuracy}")

                # Save the best model based on validation accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

        return best_model

# Train the model and get the best version based on validation accuracy
best_model = OxFordFlowersClassifier.train_validate()

# Define a dummy input for the upcoming ONNX export
input_shape = (1, 3, 224, 224)

# Install the onnxruntime for ONNX operations
!pip install onnxruntime

import numpy as np
import onnx_tf
import onnx

# Save the best PyTorch model to a file
model_path = "best_model.pth"
print(f"Saving model to: {model_path}")
torch.save(best_model.state_dict(), model_path)

# Convert the PyTorch model to ONNX format
dummy_input = torch.randn(input_shape).cuda()
onnx_model_path = "best_model_old.onnx"
torch.onnx.export(best_model, dummy_input, onnx_model_path, verbose=False)

# Load the ONNX model and rename its input for compatibility
onnx_model = onnx.load(onnx_model_path)

from onnx import helper

# Rename the model input for compatibility with TensorFlow
name_map = {"input.1": "input_1"}

# Create a list of new input names
new_inputs = []
for inp in onnx_model.graph.input:
    if inp.name in name_map:
        new_inp = helper.make_tensor_value_info(name_map[inp.name],
                                                inp.type.tensor_type.elem_type,
                                                [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
        new_inputs.append(new_inp)
    else:
        new_inputs.append(inp)

# Update the ONNX model with new input names
onnx_model.graph.ClearField("input")
onnx_model.graph.input.extend(new_inputs)

# Rename nodes in the graph that use the old input name
for node in onnx_model.graph.node:
    for i, input_name in enumerate(node.input):
        if input_name in name_map:
            node.input[i] = name_map[input_name]

# Save the updated ONNX model
onnx.save(onnx_model, 'best_model_new.onnx')

# Convert the ONNX model to TensorFlow format
!onnx-tf convert -i best_model_new.onnx -o best_model_new.pb

# Convert the TensorFlow model to TensorFlow Lite format
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS,
  tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
with open('best_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Use Colab's utility to download files to the user's local machine
from google.colab import files
files.download('best_model.tflite')

"""
