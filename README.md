# Image Classification with ResNet152

This is a Python application for image classification using a pre-trained ResNet152 model. The program loads a dataset, applies data transformations, trains the model, and evaluates its performance.

## Features
- Uses a pre-trained ResNet152 model from `torchvision.models`.
- Applies advanced data augmentation techniques.
- Supports training and evaluation on custom datasets.
- Implements training and testing functions with accuracy and loss tracking.
- Provides a function to predict and visualize results for single images.

## Requirements
To run this script, install the required libraries:

```bash
pip install torch torchvision matplotlib numpy Pillow
```

## How to Use

1. **Prepare the dataset**: Place your images inside the `people/` directory with the following structure:
   ```
   people/
   ├── train/
   │   ├── class1/
   │   ├── class2/
   │   └── ...
   ├── test/
   │   ├── class1/
   │   ├── class2/
   │   └── ...
   ```

2. **Run the script**: Execute the Python file to train the model.
   ```bash
   python script.py
   ```

3. **Train the Model**: The script will train for 5 epochs by default. You can modify the `NUM_EPOCHS` variable to adjust training duration.

4. **Test and Evaluate**: After training, the model evaluates performance on the test dataset.

5. **Predict a Single Image**: Use the `pred_and_plot_image()` function to make a prediction on a single image.
   ```python
   pred_and_plot_image(model, 'path/to/image.jpg', class_names)
   ```

## Model Training Details
- Optimizer: Adam (`torch.optim.Adam`)
- Loss function: CrossEntropyLoss (`nn.CrossEntropyLoss`)
- Batch size: 32
- Image size: 224x224
- Data Augmentation: Resize, Random Crop, Horizontal Flip, Rotation, Color Jitter, Grayscale

## Saving the Model
You can save the trained model using:
```python
torch.save(model.state_dict(), 'resNet152.pth')
```
To load the model later:
```python
model.load_state_dict(torch.load('resNet152.pth'))
```

## Example Output
Sample training output:
```
Epoch 1/5: Train Loss = 0.3456, Train Acc = 85.4%, Test Loss = 0.2765, Test Acc = 89.3%
Epoch 2/5: Train Loss = 0.2789, Train Acc = 88.9%, Test Loss = 0.2501, Test Acc = 91.2%
...
```

## License
This project is open-source and can be modified freely.

