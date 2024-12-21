# MNIST CNN with CI/CD Pipeline

A deep learning project that trains a CNN on the MNIST dataset with automated testing and deployment.

## Local Setup

1. Create a virtual environment: 
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

To train the model, run the following command:



python src/train.py

pytest tests/

## CI/CD Pipeline

This project includes a CI/CD pipeline configured with GitHub Actions. The pipeline will automatically:
- Install dependencies
- Train the model
- Run tests
- Archive model artifacts and logs

The pipeline is triggered on every push to the repository.

## Model Architecture

The model is a Convolutional Neural Network (CNN) designed for classifying handwritten digits from the MNIST dataset. It includes several convolutional layers, batch normalization, dropout for regularization, and a global average pooling layer.

## License

This project is licensed under the MIT License.