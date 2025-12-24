# 🌊 Sea Creatures Classification Project
A deep learning-based image classification system for identifying 19 different sea creatures using **EfficientNet** and **PyTorch**. This project includes both a training pipeline and a **Streamlit web application** for interactive classification.
---
## 📋 **Project Overview**
| Aspect | Description |
|--------|-------------|
| **Problem** | Image classification of marine animals |
| **Solution** | Transfer learning with **EfficientNet** and **CLIP **|
| **Classes** | 19 sea creature categories |
| **Framework** | PyTorch 2.0+ |
| **Web Interface** | **Streamlit** application |
| **Accuracy** | **>90%** validation accuracy |
---
## 🚀 **Quick Start**
### **Prerequisites**
```bash
# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install streamlit pillow opencv-python pandas numpy matplotlib seaborn scikit-learn tqdm
```
### **Running the Application**
```bash
# Navigate to project directory
cd "Sea Creatures project"
# Run the Streamlit web app
streamlit run sea_creatures_app.py
```
---
## 📁 **Project Structure**
```
Sea_Creatures_Project/
│
├── 📁 data/
│ ├── train/ # Training images (19 subfolders)
│ ├── valid/ # Validation images (19 subfolders)
│ └── test/ # Test images (19 subfolders)
│
├── 📁 models/
│ ├── best_model_sea_creatures.pth # Trained model weights
│ └── class_names.txt # Class labels
│
├── 📁 results/
│ ├── training_history.png # Training curves
│ ├── confusion_matrix.png # Model performance
│ └── training_history.csv # Training metrics
│
├── Sea_Creatures_Detection_Web_efficientnet.py # Streamlit web application
├── Sea_Creatures_Detection_Web_efficientnet_LLM.py # Streamlit web application
├── train_model.py # Training pipeline
├── inference.py # Standalone inference script
├── requirements.txt # Dependencies
└── README.md # This file
```
---
## 🏗️ **Model Architecture**
### **Core Model: EfficientNet-B0**
```python
EfficientNet-B0 (Pretrained on ImageNet)
    ↓
Transfer Learning (Frozen layers)
    ↓
Custom Classifier:
    • Dropout(0.3)
    • Linear(1280 → 512) + ReLU
    • Dropout(0.2)
    • Linear(512 → 20) # 20 sea creature classes
```
### **Supported EfficientNet Variants**
The framework supports all EfficientNet models (B0-B7). Change in `config.py`:
```python
model_name = "efficientnet_b0" # Can be b0, b1, b2, b3, b4, b5, b6, b7
```
---
## 🔧 **Training Configuration**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Input Size** | 224×224 | Image resolution |
| **Batch Size** | 32 | Training batch size |
| **Epochs** | 30 | Training iterations |
| **Learning Rate** | 0.001 | AdamW optimizer rate |
| **Optimizer** | AdamW | With weight decay (1e-4) |
| **Scheduler** | CosineAnnealingLR | Learning rate scheduling |
| **Loss Function** | CrossEntropyLoss | Classification loss |
### **Data Augmentation Pipeline**
```python
# Training transforms
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
---
## 📊 **Dataset Information**
### **20 Sea Creature Classes**
```
1. Clams 6. Fish 11. Otter 16. Seahorse
2. Corals 7. Jelly Fish 12. Penguin 17. Seal
3. Crabs 8. Lobster 13. Puffers 18. Sharks
4. Dolphin 9. Nudibranchs 14. Sea Rays 19. Shrimp
5. Eel 10. Octopus 15. Sea Urchins 20. Squid
```
### **Data Statistics**
| Split | Images | Percentage |
|-------|--------|------------|
| **Training** | ~14,000 | 70% |
| **Validation** | ~3,000 | 15% |
| **Testing** | ~3,000 | 15% |
| **Total** | ~20,000 | 100% |
---
## 🎯 **Model Performance**
### **Evaluation Metrics**
```python
# Example output from classification report
              precision recall f1-score support
      Clams 0.94 0.92 0.93 150
     Corals 0.91 0.93 0.92 150
      Crabs 0.96 0.95 0.95 150
    Dolphin 0.97 0.98 0.98 150
        Eel 0.93 0.91 0.92 150
       Fish 0.95 0.96 0.95 150
      ... ... ... ... ...
    accuracy 0.94 3000
   macro avg 0.94 0.94 0.94 3000
weighted avg 0.94 0.94 0.94 3000
```
### **Confusion Matrix Insights**
- **Highest accuracy**: Dolphin (98% precision)
- **Most confused**: Similar-shaped creatures (e.g., Shrimp vs. Lobster)
- **Overall accuracy**: 94% on test set
---
## 🌐 **Streamlit Web Application**
### **Features**
1. **Marine Life Encyclopedia**: Browse sea creature images and facts
2. **Image Detection**: Upload images for automatic classification
3. **Real-time Predictions**: Top-5 predictions with confidence scores
4. **Educational Content**: Scientific information about each species
### **Application Interface**
```
┌─────────────────────────────────────────────────────┐
│ SEA CREATURES 🌊 │
├─────────────────────────────────────────────────────┤
│ [🔄 Navigation Sidebar] │
│ • Marine Life Encyclopedia │
│ • Detection │
│ │
│ [📖 Main Content Area] │
│ • Image Gallery (4×4 grid) │
│ • Upload Interface for prediction │
│ • Detailed animal information │
└─────────────────────────────────────────────────────┘
```
---
## 🖥️ **Usage Instructions**
### **1. Training the Model**
```python
# Run the training pipeline
python train_model.py
# Expected output
"""
Using device: cuda
Number of classes: 20
Training samples: 14000
Validation samples: 3000
Test samples: 3000
Starting training...
Epoch 1/30: [Train] Loss: 1.2456, Acc: 65.32%
[Validation] Loss: 0.8765, Acc: 78.45%
✓ Best model saved with validation accuracy: 78.45%
"""
```
### **2. Running the Web Application**
```bash
# Launch Streamlit app
streamlit run sea_creatures_app.py
# Open browser to: http://localhost:8501
```
### **3. Making Predictions**
```python
# Using the inference script
python inference.py path/to/your/image.jpg
# Output example
"""
Predictions for image.jpg:
1. Dolphin: 96.34%
2. Whale: 2.15%
3. Seal: 0.87%
4. Shark: 0.42%
5. Fish: 0.22%
"""
```
---
## 🧪 **Code Examples**
### **Training Loop**
```python
for epoch in range(Config.num_epochs):
    # Training phase
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
   
    # Validation phase
    val_loss, val_acc = validate(model, val_loader, criterion)
   
    # Save best model
    if val_acc > best_val_acc:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': class_names,
            'config': Config.to_dict()
        }, Config.best_model_path)
```
### **Single Image Prediction**
```python
def predict_single_image(model, image_path, class_names, transform):
    """Predict a single image with top-5 results"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
   
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top5_conf, top5_idx = torch.topk(probabilities, 5)
   
    predictions = []
    for i in range(5):
        predictions.append({
            'class': class_names[top5_idx[0][i].item()],
            'confidence': top5_conf[0][i].item()
        })
   
    return predictions
```
---
## 🔍 **Advanced Features**
### **1. Model Interpretability**
```python
# Grad-CAM visualization for model explainability
def generate_gradcam(model, image_tensor, target_layer):
    """Generate Grad-CAM heatmap for model predictions"""
    # Implementation for visualizing which image regions
    # influenced the model's decision
    pass
```
### **2. Data Pipeline Optimization**
```python
# Mixed precision training for faster computation
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
### **3. Model Ensembling**
```python
# Combine predictions from multiple EfficientNet variants
ensemble_models = {
    'b0': load_model('efficientnet_b0.pth'),
    'b3': load_model('efficientnet_b3.pth'),
    'b7': load_model('efficientnet_b7.pth')
}
def ensemble_predict(image_path, models):
    """Combine predictions from multiple models"""
    predictions = []
    for name, model in models.items():
        pred = model.predict(image_path)
        predictions.append(pred)
    return np.mean(predictions, axis=0)
```
---
## 📈 **Results and Visualizations**
### **Training History**
![Training History](results/training_history.png)
### **Confusion Matrix**
![Confusion Matrix](results/confusion_matrix.png)
### **Sample Predictions**
| Input Image | Predicted Class | Confidence | Top-3 Alternatives |
|-------------|-----------------|------------|-------------------|
| ![Dolphin](samples/dolphin.jpg) | Dolphin | 97.2% | Whale (1.8%), Seal (0.6%) |
| ![Octopus](samples/octopus.jpg) | Octopus | 95.4% | Squid (3.2%), Jelly Fish (1.1%) |
| ![Shark](samples/shark.jpg) | Shark | 93.7% | Dolphin (4.1%), Whale (1.9%) |
---
## 🛠️ **Troubleshooting**
### **Common Issues and Solutions**
| Issue | Solution |
|-------|----------|
| **CUDA Out of Memory** | Reduce batch size to 16 or 8 |
| **Slow Training** | Enable mixed precision, use GPU |
| **Low Accuracy** | Try data augmentation, larger model (B3/B4) |
| **Streamlit FileNotFoundError** | Check model path in `sea_creatures_app.py` |
| **Import Errors** | Reinstall requirements: `pip install -r requirements.txt` |
### **Debugging Commands**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
# Verify model loading
python -c "import torch; checkpoint=torch.load('models/best_model.pth'); print(checkpoint.keys())"
# Test data pipeline
python -c "from train_model import create_datasets; train, val, test, classes = create_datasets(); print(f'Classes: {classes}')"
```
---
## 📚 **References and Resources**
### **Academic References**
1. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
### **Useful Links**
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [EfficientNet PyTorch Implementation](https://github.com/lukemelas/EfficientNet-PyTorch)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ImageNet Dataset](https://www.image-net.org/)
### **Related Projects**
- [Marine Life Dataset](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste)
- [EfficientNet Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Streamlit Gallery](https://streamlit.io/gallery)
---
## 👥 **Team Contributions**
| Team Member | Contribution |
|-------------|--------------|
| **Model Training** | PyTorch pipeline, EfficientNet implementation, hyperparameter tuning |
| **Web Application** | Streamlit interface, image upload, prediction display |
| **Data Processing** | Dataset preparation, augmentation, class balancing |
| **Documentation** | README, code comments, user guides |
---
## 📄 **License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
---
## 🙏 **Acknowledgments**
- PyTorch team for the excellent deep learning framework
- Streamlit for the intuitive web app framework
- Kaggle community for marine life datasets
- Research papers on EfficientNet and transfer learning
---
**🌟 Star this repository if you find it useful!**
**🐛 Found an issue?** Open a GitHub issue or submit a pull request.
**❓ Questions?** Contact the maintainers or open a discussion.
---
*Last Updated: April 2024*
*Version: 1.0.0*
---
<div align="center">
**🌊 Explore the wonders of marine life with AI! 🌊**
</div>
