# MNIST CNN with CI/CD Pipeline

A deep learning project that trains a Convolutional Neural Network (CNN) on the MNIST dataset with automated testing and deployment.

## Table of Contents
- [Local Setup](#local-setup)
- [Training the Model](#training-the-model)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Model Architecture](#model-architecture)
- [License](#license)

## Local Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

To train the model, run the following command:
```bash
python src/train.py
```

This will initiate the training process, and the model will be trained on the MNIST dataset. The training logs will be saved in the `logs` directory.

## Testing

To run the tests, execute:
```bash
pytest tests/
```

The tests include checks for:
- Total parameter count (should be less than 20,000).
- Use of Batch Normalization layers.
- Use of Dropout layers.
- Use of Global Average Pooling (GAP).

## CI/CD Pipeline

This project includes a CI/CD pipeline configured with GitHub Actions. The pipeline will automatically:
- Install dependencies.
- Train the model.
- Run tests to ensure the model meets the specified architecture requirements.
- Archive model artifacts and logs.

The pipeline is triggered on every push to the repository.

## Model Architecture

The model is a Convolutional Neural Network (CNN) designed for classifying handwritten digits from the MNIST dataset. It includes:
- **Convolutional Layers:** Several convolutional layers to extract features from the input images.
- **Batch Normalization:** Normalizes the output of the previous layer to improve training speed and stability.
- **Dropout:** Regularization technique to prevent overfitting by randomly setting a fraction of input units to 0 during training.
- **Global Average Pooling (GAP):** Reduces the spatial dimensions of the feature maps to a single value per channel, which helps in reducing the number of parameters.

### Model Summary
The model architecture can be summarized as follows:
- Input: 1 channel (28x28 pixels)
- Convolutional layers with increasing depth
- Batch Normalization after each convolutional layer
- Dropout layers for regularization
- Global Average Pooling layer
- Fully connected layer to output 10 classes (digits 0-9)


## TEST LOGS

INGUR1M0071:assign6 dhamenit$ python3 src/train.py
2024-12-24 17:48:35,528 - INFO - Using device: cpu
2024-12-24 17:48:35,529 - INFO - Loading MNIST dataset...
2024-12-24 17:48:35,653 - INFO - Number of training samples: 60000
2024-12-24 17:48:35,654 - INFO - Number of testing samples: 10000
2024-12-24 17:48:35,654 - INFO - Dataset loaded. Training samples: 60000, Test samples: 10000
2024-12-24 17:48:35,677 - INFO - Model initialized with 13940 parameters
2024-12-24 17:48:35,677 - INFO - Model architecture:
MNIST_CNN(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
  (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout1): Dropout(p=0.1, inplace=False)
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
  (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout2): Dropout(p=0.1, inplace=False)
  (conv1x1_1): Conv2d(32, 10, kernel_size=(1, 1), stride=(1, 1))
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(10, 16, kernel_size=(3, 3), stride=(1, 1))
  (bn3): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout3): Dropout(p=0.1, inplace=False)
  (conv4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
  (bn4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout4): Dropout(p=0.1, inplace=False)
  (conv5): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
  (bn5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout5): Dropout(p=0.1, inplace=False)
  (conv6): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn6): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout6): Dropout(p=0.1, inplace=False)
  (gap): AdaptiveAvgPool2d(output_size=1)
  (conv7): Conv2d(16, 10, kernel_size=(1, 1), stride=(1, 1))
)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
       BatchNorm2d-2           [-1, 16, 26, 26]              32
           Dropout-3           [-1, 16, 26, 26]               0
            Conv2d-4           [-1, 32, 24, 24]           4,640
       BatchNorm2d-5           [-1, 32, 24, 24]              64
           Dropout-6           [-1, 32, 24, 24]               0
            Conv2d-7           [-1, 10, 24, 24]             330
         MaxPool2d-8           [-1, 10, 12, 12]               0
            Conv2d-9           [-1, 16, 10, 10]           1,456
      BatchNorm2d-10           [-1, 16, 10, 10]              32
          Dropout-11           [-1, 16, 10, 10]               0
           Conv2d-12             [-1, 16, 8, 8]           2,320
      BatchNorm2d-13             [-1, 16, 8, 8]              32
          Dropout-14             [-1, 16, 8, 8]               0
           Conv2d-15             [-1, 16, 6, 6]           2,320
      BatchNorm2d-16             [-1, 16, 6, 6]              32
          Dropout-17             [-1, 16, 6, 6]               0
           Conv2d-18             [-1, 16, 6, 6]           2,320
      BatchNorm2d-19             [-1, 16, 6, 6]              32
          Dropout-20             [-1, 16, 6, 6]               0
AdaptiveAvgPool2d-21             [-1, 16, 1, 1]               0
           Conv2d-22             [-1, 10, 1, 1]             170
================================================================
Total params: 13,940
Trainable params: 13,940
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.81
Params size (MB): 0.05
Estimated Total Size (MB): 0.87
----------------------------------------------------------------
2024-12-24 17:48:35,789 - INFO - Model Summary:
None
Epoch 1/20: 100%|████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:05<00:00, 14.33it/s, loss=1.1858, train_acc=70.64%]
2024-12-24 17:49:41,308 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 52.64it/s, test_acc=94.73%]
2024-12-24 17:49:44,292 - INFO - Epoch 1/20:
2024-12-24 17:49:44,292 - INFO - Training Loss: 1.1858, Training Accuracy: 70.64%
2024-12-24 17:49:44,292 - INFO - Test Loss: 0.3432, Test Accuracy: 94.73%
2024-12-24 17:49:44,301 - INFO - New best model saved: models/model_acc94.73_params13940_20241224_174944.pth
Epoch 2/20: 100%|████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:05<00:00, 14.38it/s, loss=0.2839, train_acc=94.62%]
2024-12-24 17:50:49,516 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 51.40it/s, test_acc=97.73%]
2024-12-24 17:50:52,571 - INFO - Epoch 2/20:
2024-12-24 17:50:52,571 - INFO - Training Loss: 0.2839, Training Accuracy: 94.62%
2024-12-24 17:50:52,571 - INFO - Test Loss: 0.1175, Test Accuracy: 97.73%
2024-12-24 17:50:52,576 - INFO - New best model saved: models/model_acc97.73_params13940_20241224_175052.pth
Epoch 3/20: 100%|████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:02<00:00, 15.12it/s, loss=0.1606, train_acc=96.43%]
2024-12-24 17:51:54,633 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 63.17it/s, test_acc=98.56%]
2024-12-24 17:51:57,120 - INFO - Epoch 3/20:
2024-12-24 17:51:57,120 - INFO - Training Loss: 0.1606, Training Accuracy: 96.43%
2024-12-24 17:51:57,120 - INFO - Test Loss: 0.0676, Test Accuracy: 98.56%
2024-12-24 17:51:57,124 - INFO - New best model saved: models/model_acc98.56_params13940_20241224_175157.pth
Epoch 4/20: 100%|████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:06<00:00, 14.00it/s, loss=0.1195, train_acc=97.13%]
2024-12-24 17:53:04,113 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 48.79it/s, test_acc=98.49%]
2024-12-24 17:53:07,332 - INFO - Epoch 4/20:
2024-12-24 17:53:07,332 - INFO - Training Loss: 0.1195, Training Accuracy: 97.13%
2024-12-24 17:53:07,332 - INFO - Test Loss: 0.0588, Test Accuracy: 98.49%
Epoch 5/20: 100%|████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:02<00:00, 15.06it/s, loss=0.0994, train_acc=97.51%]
2024-12-24 17:54:09,620 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 40.20it/s, test_acc=98.85%]
2024-12-24 17:54:13,526 - INFO - Epoch 5/20:
2024-12-24 17:54:13,526 - INFO - Training Loss: 0.0994, Training Accuracy: 97.51%
2024-12-24 17:54:13,526 - INFO - Test Loss: 0.0449, Test Accuracy: 98.85%
2024-12-24 17:54:13,529 - INFO - New best model saved: models/model_acc98.85_params13940_20241224_175413.pth
Epoch 6/20: 100%|████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:13<00:00, 12.79it/s, loss=0.0862, train_acc=97.71%]
2024-12-24 17:55:26,846 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 54.57it/s, test_acc=98.97%]
2024-12-24 17:55:29,724 - INFO - Epoch 6/20:
2024-12-24 17:55:29,724 - INFO - Training Loss: 0.0862, Training Accuracy: 97.71%
2024-12-24 17:55:29,724 - INFO - Test Loss: 0.0399, Test Accuracy: 98.97%
2024-12-24 17:55:29,729 - INFO - New best model saved: models/model_acc98.97_params13940_20241224_175529.pth
Epoch 7/20: 100%|████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:04<00:00, 14.64it/s, loss=0.0765, train_acc=97.98%]
2024-12-24 17:56:33,810 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 46.04it/s, test_acc=98.91%]
2024-12-24 17:56:37,220 - INFO - Epoch 7/20:
2024-12-24 17:56:37,221 - INFO - Training Loss: 0.0765, Training Accuracy: 97.98%
2024-12-24 17:56:37,221 - INFO - Test Loss: 0.0377, Test Accuracy: 98.91%
Epoch 8/20: 100%|████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:08<00:00, 13.62it/s, loss=0.0693, train_acc=98.14%]
2024-12-24 17:57:46,081 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 49.67it/s, test_acc=99.11%]
2024-12-24 17:57:49,243 - INFO - Epoch 8/20:
2024-12-24 17:57:49,243 - INFO - Training Loss: 0.0693, Training Accuracy: 98.14%
2024-12-24 17:57:49,243 - INFO - Test Loss: 0.0306, Test Accuracy: 99.11%
2024-12-24 17:57:49,259 - INFO - New best model saved: models/model_acc99.11_params13940_20241224_175749.pth
Epoch 9/20: 100%|████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:05<00:00, 14.25it/s, loss=0.0660, train_acc=98.15%]
2024-12-24 17:58:55,106 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 50.04it/s, test_acc=99.17%]
2024-12-24 17:58:58,244 - INFO - Epoch 9/20:
2024-12-24 17:58:58,244 - INFO - Training Loss: 0.0660, Training Accuracy: 98.15%
2024-12-24 17:58:58,245 - INFO - Test Loss: 0.0282, Test Accuracy: 99.17%
2024-12-24 17:58:58,259 - INFO - New best model saved: models/model_acc99.17_params13940_20241224_175858.pth
Epoch 10/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:06<00:00, 14.14it/s, loss=0.0613, train_acc=98.29%]
2024-12-24 18:00:04,580 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 49.60it/s, test_acc=98.87%]
2024-12-24 18:00:07,746 - INFO - Epoch 10/20:
2024-12-24 18:00:07,747 - INFO - Training Loss: 0.0613, Training Accuracy: 98.29%
2024-12-24 18:00:07,747 - INFO - Test Loss: 0.0412, Test Accuracy: 98.87%
Epoch 11/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:03<00:00, 14.66it/s, loss=0.0573, train_acc=98.41%]
2024-12-24 18:01:11,726 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 50.84it/s, test_acc=99.17%]
2024-12-24 18:01:14,815 - INFO - Epoch 11/20:
2024-12-24 18:01:14,815 - INFO - Training Loss: 0.0573, Training Accuracy: 98.41%
2024-12-24 18:01:14,815 - INFO - Test Loss: 0.0269, Test Accuracy: 99.17%
Epoch 12/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:06<00:00, 14.06it/s, loss=0.0556, train_acc=98.39%]
2024-12-24 18:02:21,515 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 49.51it/s, test_acc=99.14%]
2024-12-24 18:02:24,687 - INFO - Epoch 12/20:
2024-12-24 18:02:24,687 - INFO - Training Loss: 0.0556, Training Accuracy: 98.39%
2024-12-24 18:02:24,687 - INFO - Test Loss: 0.0276, Test Accuracy: 99.14%
Epoch 13/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:05<00:00, 14.37it/s, loss=0.0515, train_acc=98.53%]
2024-12-24 18:03:29,963 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 48.28it/s, test_acc=99.32%]
2024-12-24 18:03:33,216 - INFO - Epoch 13/20:
2024-12-24 18:03:33,216 - INFO - Training Loss: 0.0515, Training Accuracy: 98.53%
2024-12-24 18:03:33,216 - INFO - Test Loss: 0.0245, Test Accuracy: 99.32%
2024-12-24 18:03:33,221 - INFO - New best model saved: models/model_acc99.32_params13940_20241224_180333.pth
Epoch 14/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:10<00:00, 13.32it/s, loss=0.0512, train_acc=98.52%]
2024-12-24 18:04:43,621 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 46.13it/s, test_acc=99.25%]
2024-12-24 18:04:47,026 - INFO - Epoch 14/20:
2024-12-24 18:04:47,026 - INFO - Training Loss: 0.0512, Training Accuracy: 98.52%
2024-12-24 18:04:47,026 - INFO - Test Loss: 0.0244, Test Accuracy: 99.25%
Epoch 15/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:08<00:00, 13.76it/s, loss=0.0498, train_acc=98.56%]
2024-12-24 18:05:55,193 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 48.10it/s, test_acc=99.33%]
2024-12-24 18:05:58,458 - INFO - Epoch 15/20:
2024-12-24 18:05:58,459 - INFO - Training Loss: 0.0498, Training Accuracy: 98.56%
2024-12-24 18:05:58,459 - INFO - Test Loss: 0.0227, Test Accuracy: 99.33%
2024-12-24 18:05:58,463 - INFO - New best model saved: models/model_acc99.33_params13940_20241224_180558.pth
Epoch 16/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:08<00:00, 13.71it/s, loss=0.0480, train_acc=98.61%]
2024-12-24 18:07:06,895 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 51.63it/s, test_acc=99.32%]
2024-12-24 18:07:09,937 - INFO - Epoch 16/20:
2024-12-24 18:07:09,937 - INFO - Training Loss: 0.0480, Training Accuracy: 98.61%
2024-12-24 18:07:09,937 - INFO - Test Loss: 0.0228, Test Accuracy: 99.32%
Epoch 17/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:06<00:00, 14.01it/s, loss=0.0453, train_acc=98.64%]
2024-12-24 18:08:16,884 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 47.56it/s, test_acc=99.30%]
2024-12-24 18:08:20,187 - INFO - Epoch 17/20:
2024-12-24 18:08:20,187 - INFO - Training Loss: 0.0453, Training Accuracy: 98.64%
2024-12-24 18:08:20,187 - INFO - Test Loss: 0.0206, Test Accuracy: 99.30%
Epoch 18/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:12<00:00, 12.86it/s, loss=0.0436, train_acc=98.72%]
2024-12-24 18:09:33,138 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 46.24it/s, test_acc=99.41%]
2024-12-24 18:09:36,535 - INFO - Epoch 18/20:
2024-12-24 18:09:36,535 - INFO - Training Loss: 0.0436, Training Accuracy: 98.72%
2024-12-24 18:09:36,535 - INFO - Test Loss: 0.0206, Test Accuracy: 99.41%
2024-12-24 18:09:36,546 - INFO - New best model saved: models/model_acc99.41_params13940_20241224_180936.pth
Epoch 19/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:08<00:00, 13.63it/s, loss=0.0443, train_acc=98.68%]
2024-12-24 18:10:45,342 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 47.62it/s, test_acc=99.27%]
2024-12-24 18:10:48,640 - INFO - Epoch 19/20:
2024-12-24 18:10:48,640 - INFO - Training Loss: 0.0443, Training Accuracy: 98.68%
2024-12-24 18:10:48,640 - INFO - Test Loss: 0.0224, Test Accuracy: 99.27%
Epoch 20/20: 100%|███████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:13<00:00, 12.81it/s, loss=0.0430, train_acc=98.70%]
2024-12-24 18:12:01,876 - INFO - Starting evaluation...
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 50.38it/s, test_acc=99.44%]
2024-12-24 18:12:04,993 - INFO - Epoch 20/20:
2024-12-24 18:12:04,993 - INFO - Training Loss: 0.0430, Training Accuracy: 98.70%
2024-12-24 18:12:04,993 - INFO - Test Loss: 0.0197, Test Accuracy: 99.44%
2024-12-24 18:12:05,000 - INFO - New best model saved: models/model_acc99.44_params13940_20241224_181204.pth
2024-12-24 18:12:05,000 - INFO - ==================================================
2024-12-24 18:12:05,000 - INFO - Training Summary:
2024-12-24 18:12:05,000 - INFO - Total Parameters: 13,940
2024-12-24 18:12:05,000 - INFO - Best Test Accuracy: 99.44%
2024-12-24 18:12:05,000 - INFO - Best Model Path: models/model_acc99.44_params13940_20241224_181204.pth
2024-12-24 18:12:05,000 - INFO - Model Size: 70.85 KB
2024-12-24 18:12:05,000 - INFO - ==================================================

## License

This project is licensed under the MIT License.