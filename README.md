🧠 Generative Adversarial Network (GAN) on CIFAR-10

A deep learning project implementing a Generative Adversarial Network (GAN) trained on the CIFAR-10 dataset to generate realistic-looking images belonging to various object categories (airplanes, cars, birds, cats, etc.).

🚀 Project Overview

This project implements a vanilla GAN (Generative Adversarial Network) from scratch using PyTorch to generate 32×32 RGB images similar to those in the CIFAR-10 dataset.

The GAN consists of:

A Generator that learns to produce fake but realistic-looking images.

A Discriminator that learns to distinguish between real and fake images.

Over time, the generator improves its ability to fool the discriminator, resulting in high-quality synthetic images.

📚 Dataset

CIFAR-10 is a standard benchmark dataset containing:

60,000 images (32×32 pixels, RGB)

10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

50,000 training + 10,000 testing images

Dataset is automatically downloaded using:

torchvision.datasets.CIFAR10(download=True)

🧩 Model Architecture
🧠 Generator

Input: Random noise vector (latent dimension z_dim)

Layers: Fully connected + transposed convolutional layers

Activation: ReLU (hidden), Tanh (output)

Output: Generated RGB image (3×32×32)

🔍 Discriminator

Input: Real or fake image

Layers: Convolutional + LeakyReLU + Dropout

Output: Single scalar (probability that image is real)

⚙️ Training Setup
Parameter	Value
Optimizer	Adam (lr=0.0002, β1=0.5, β2=0.999)
Loss Function	Binary Cross-Entropy (BCE)
Epochs	100+ (depending on convergence)
Batch Size	128
Latent Vector (z_dim)	100
Framework	PyTorch
Dataset	CIFAR-10
Hardware	GPU recommended (CUDA)
🧮 Training Process

Discriminator Step:

Train D on real images (label = 1) and fake images (label = 0).

Generator Step:

Train G to fool D — make D classify fake images as real.

Alternate between G and D updates.

Save generated samples after each epoch to monitor visual improvements.

🖼️ Results
🎨 Generated Samples

As training progresses, generated images start resembling real CIFAR-10 images.

Epoch	Example Output
1	Random noise-like images
25	Coarse object shapes visible
50	More detailed textures
100	Realistic class-specific structures



📈 Loss Curves

You can visualize training dynamics using:

plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.legend()
plt.show()

📦 Project Structure
├── data/                 # CIFAR-10 dataset (auto-downloaded)
├── models/
│   ├── generator.py      # Generator architecture
│   ├── discriminator.py  # Discriminator architecture
├── train.py              # Training loop
├── utils.py              # Helper functions
├── results/              # Saved generated images
├── README.md
└── requirements.txt

🧰 Requirements

Install dependencies:

pip install torch torchvision matplotlib tqdm

▶️ How to Run
python train.py


Optional arguments:

--epochs 100
--batch_size 128
--z_dim 100
--lr 0.0002

📊 Evaluation

Visual inspection of generated images.

Inception Score (optional advanced metric).

FID (Fréchet Inception Distance) for quantitative evaluation.

🧠 Key Learnings

Understanding adversarial training and stability issues.

Balancing generator and discriminator training.

Handling mode collapse and vanishing gradients.

Exploring the impact of hyperparameters and normalization layers.

🔮 Future Improvements

Implement DCGAN (Deep Convolutional GAN) with BatchNorm and strided conv layers.

Add Wasserstein GAN (WGAN) with gradient penalty for more stable training.

Try conditional GANs (cGAN) for class-specific image generation.

Evaluate using FID scores to quantify realism.

🏁 Conclusion

This project demonstrates how GANs can learn to synthesize realistic images from random noise.
Despite training on small 32×32 images, the model captures color, shape, and texture distributions of CIFAR-10 classes, showing the power of adversarial learning.
