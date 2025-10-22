# VQVAE Generative Model of the HipMRI Study on Prostate Cancer

## Project Description

In this project, I have integrated a **Vector Quantised Variational Autoencoder (VQVAE)** for learning latent representations of prostate MRI scans from the HipMRI study dataset. This project’s objective is to generate and reconstruct a high-fidelity 2D prostate MRI slices with a high similarity to the original images.

The discrete latent space provided by the VQ-VAE framework [1], enabled the model to produce reconstructions that are visually like the original scans while retaining critical structural features. The model was trained on 2D axial slices extracted from 3D NIfTI volumes, achieving a **Structural Similarity Index Measure (SSIM)** of 0.838.

This project was completed as part of the COMP3710 assignment, under the “**Hard Difficulty**” section.


## How It Works

The **Vector Quantized Variational Autoencoder (VQ-VAE)** model functions by compressing high-dimensional 2D prostate MRI images into discrete, lower-dimensional latent representations and subsequently reconstructing them with high fidelity. This process enables the extraction of clinically meaningful features while minimizing reconstruction blur which is a critical factor in medical imaging applications [1].

## Model Architecture

### Dataloader

The data loader prepares 2D prostate MRI slices for training, validation and testing. It automatically reads all the .png images from the dataset folders and converts them to grayscale tensors which resizes them to **128x128** pixels for consistent input to the model. The images are then normalized to the [0,1] range to ensure stable learning. Pytorch [4] is then used to return the batches of preprocessed images to enable efficiency in training in small batches.

### Encoder

In this project, the task of the encoder is to take the grayscale images with one input channel and progressively downsamples them to a smaller size while increasing the number of feature channels to capture complex patterns.

The encoder consists of several convolutional layers followed by ReLU activations, with kernel size 4, stride 2, and padding 1 and it effectively halve the image dimensions at each step. The number of feature channels increases from 64 up to 128, allowing the network to extract a deeper set of features from each MRI slice.

After that, the residual layers [3] are added, and it allows gradient flow directly across layers to help with the encoder. The residual structure prevents loss of small textural details.
The encoder outputs a compressed latent feature map (z_e) and is then passed to the Codebook for quantisation.


### Vector Quantiser

The codebook module converts continuous latent representations into descrete embeddings; In this project, **512 embedding vectors** each consisting of **64 dimensions** are in the codebook. When the encoder produces its output, every vector in the map is compared with all 512 embeddings. Then the embedding with the smallest Euclidean distance is selected and is used to replace the original vector.

It ensures that the model repesents images using a set of learned features. It improves the sharpness and reduces the blur in the reconstructed images [1][2]. Overall, this module helps the model to learn repeatable structural features. 


### Decoder

The Decoder reconstructs the MRI slice from the quantised latent representation produced by the codebook. It is used for upsampling the feature maps back to original size.

In this project, the decoder will use a **1x1 convolutional layer** that prepares the quantised embeddings and it passes the features through a series of **transposed convolutional layers**(ConvTranspose2d) with kernel size 4, stride 2, and padding 1 for a progressive upsampling. In between these layers, Residual block are then used to refine and stabilise the feature maps.

Finally, a **3x3 convutional layer** produces a single-channel grayscale image with pixel intensities in the normalized [0, 1] range, resulting in a reconstructed image that closely approximates the original MRI slice.

### VQVAE

The VQVAE class integrates encoder, codebook and decoder(previously described class) into a single end-to-end model. It defines the forward pass and computes the total loss during training. 

The flow of the process in this project is as follows:
1. The encoder compresses the MRI slice into a latent feature map.
2. The codebook quantises the vectors into discrete embeddings and produces a quantised latent map.
3. The Decoder reconstructs the original MRI slice from the quantised map and generates the reconstructed output.


This design makes the system both interpretable and effective for capturing high-level prostate MRI structures.

### Training

In this project, I used the Adam optimizer for training with a learning rate of 1e-4 for a total of 60 epochs. The Structural Similarity Index (SSIM) was selected as the main evaluation metric to assess reconstruction similarity. The dataset was divided into three subsets: training, validation, and testing. Each image was first normalized to a [0, 1] range, then resized to 128 × 128 pixels before being loaded through a PyTorch DataLoader [4].

In each epoch, batches of MRI slices were passed through the model, and the total loss was calculated and backpropagated. After every epoch, I checked the validation SSIM to see whether the model was improving on unseen data. I noticed that the SSIM began to stabilise after around 30 epochs, suggesting that the model had mostly converged. When the model is done training, the highest SSIM value achieved, was saved automatically as **best.pt**.

The validation SSIM plot shown below shows a steady upward trend throughout training. In the early stages of training, the SSIM increased rapidly from roughly 0.35 to 0.7, which indicates that the model learned the fundamental structural pattern quite quickly. After around 30 epochs, the improvement rate slowed and by the 60th epoch, the SSIM had stopped at roughly 0.8-0.85 similarity.

![ssim_curve](readme_images/ssim_curve.png)
![loss_curve](readme_images/loss_curve.png)

### Prediction

For prediction, the trained VQ-VAE model was first initialised and then loaded with the **best.pt** checkpoint. This ensured that inference began from the best-performing trained weights rather than from scratch. The script automatically selected the GPU when available and transferred both the model and test data onto it for faster computation. The model was then switched to evaluation mode using model.eval(), and inference was performed inside a torch.no_grad() block to disable gradient tracking and minimise memory usage.

The HipMRISliceSet was used to load the test data. Each slice was converted to a grayscale **128 × 128 image** and normalised to a **[0, 1]** range to match the preprocessing conditions used during training. The **DataLoader** prepared the test data in small, non-shuffled batches to ensure that the results were reproducible across runs.

During reconstruction, the vector-quantisation layer mapped each encoded feature to one of 512 discrete codebook entries, converting the continuous encoder output into a quantised latent representation. The decoder then upsampled these latent codes back to the original image resolution using a combination of transposed convolutions and residual blocks.

To evaluate the reconstruction quality, the **Structural Similarity Index** (SSIM) was computed using the skimage.metrics.structural_similarity function with a data range of 1.0. The average SSIM was calculated across all test batches to give a single performance score. This final SSIM value was saved automatically to final_ssim.txt, and a visual comparison of the original and reconstructed MRI slices was stored in the readme_images/ directory.

![reconstructed_images](readme_images/test_recons.png)

## Dependencies

- python 3.10.11
- torch 2.2.1
- torchvision 0.17.1
- numpy 1.26.4
- opencv-python 4.10.0
- nibabel 5.2.0
- matplotlib 3.9.0
- scikit-image 0.23.2
- tqdm 4.66.4
- Pillow 10.2.0

## Results

The final SSIM score for the model was **0.8389** as per recorded in the final_ssim.txt

![reconstructed_images](readme_images/test_recons.png)

## Reproducibility

This project was designed to ensure that all results can be reproduced.
1. All random seeds were fixed to ensure consistent results across runs
2. Same preprocessing, hyperparameters and model settings were used for training, validation and testing.
   
To reproduce the result:
1. Install the dependencies
2. Have the proper dataset
3. Run train.py
4. Then run predict.py
5. Finally, the script will save the new image into readme_images/ and then print the Test SSIM as well as save it into final_ssim.txt

## Future improvements

Architecture wise, I could increase codebook size or try VQVAE2 for sharper detail. Furthermore, adding perceptual losses alongside MSE to reduce blur.

Data processing wise, a slice-quality filter could be implemented to ensure that only high-quality MRI slices are used for training and evaluation. This would improve the overall reconstruction accuracy.


## Reference

1. Van den Oord, Vinyals, Kavukcuoglu. Neural Discrete Representation Learning. NeurIPS 2017. https://arxiv.org/abs/1711.00937
2. Razavi, Van den Oord, Vinyals. VQ-VAE-2. NeurIPS 2019 https://arxiv.org/abs/1906.00446 (hierarchical extension).
3. He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. CVPR 2016 https://arxiv.org/abs/1512.03385 (residual blocks).
4. PyTorch Official Documentation – “Training a VAE/VQ-VAE in PyTorch.” https://pytorch.org/tutorials/
5. GitHub Docs – “About READMEs.” https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
6. ChatGPT (OpenAI, 2025) – used for assistance in formatting, proofreading, and clarifying explanations within the README.md file. 