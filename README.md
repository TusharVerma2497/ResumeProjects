# PROJECTS
## Building Height Estimation

<p> A multitask model for building height Estimation and shadow, footprint mask generation, trained on the unbalanced GRSS-2023 dataset. Domain adapted for India with Google Open Building data, Introduced a novel seam carving based augmentation technique. Trained on UNet architecture with ASSP based encoder and two decoder one for height estimation known as regression deoder and other for shadow mask and footprint generation known as segmentation decoder, regression decoder uses windowed cross attention to query about shadow and footprint information from the segmentation decoder.<br><a href="1. Building Height Detection/Report.pdf">Report</a></p>



<!-- #### Result Image -->

<img src="1. Building Height Detection/Delhi_Estimates.png" alt="Result Image" style="display:block; margin:auto;"/>


#
## Context Aware Image Resizing
<p> The project focuses on seam carving, a content-aware image resizing technique that adjusts image dimensions by removing or inserting seams, preserving important content. It explains the implementation of seam carving using the Discrete Cosine Transform (DCT) and dynamic programming to find the minimum energy seams. Performance improvements were achieved by removing multiple seams simultaneously, reducing time complexity by 68%. The report also covers a deep learning-based seam carving approach using a VGG16-based UNet model and perceptual loss, showing promising results with a high structural similarity index (SSIM) of 0.8186.<br><a href="2. Context Aware Image Resizing/Report.pdf">Report</a>​</p>


<!-- #### Result Image -->

<img src="2. Context Aware Image Resizing/Seam_Carving.gif" alt="Result Image" style="display:block; margin:auto;"/>

#
#
#
# MINI PROJECTS

## GAN
<p> In this project, a Generative Adversarial Network (GAN) was implemented from scratch to generate images using the CIFAR-10 dataset. The architecture featured a generator that produces images from random noise and a discriminator that classifies them as real or fake. The model included batch normalization and LeakyReLU layers, with an Inception Score of 3.5, indicating moderate image quality and diversity. While the model performed reasonably well, there is room for improvement in terms of image sharpness and variety across classes.<br><a href="3. Mini Projects/GAN/Report.pdf">Report</a></p>

<!-- ### Result Image -->

<img src="3. Mini Projects/GAN/GAN Adversarial_Learning.gif" alt="Result Image" style="display:block; margin:auto;"/>

## Super-Resolution
<p> In this project, a Super Resolution Generative Adversarial Network (SRGAN) was implemented to upsample satellite images. The model enhances image resolution while preserving texture and detail. The architecture consists of a generator using convolutional layers and residual blocks for upsampling and a discriminator that classifies images as real or fake. Loss functions like adversarial, perceptual, and total variation losses were combined to improve image quality. The SRGAN effectively generates high-resolution satellite images, enabling better interpretation of geographical features and improving the utility of satellite data.<br><a href="3. Mini Projects/Super-Resolution/Report.pdf">Report</a></p>

<!-- ### Result Image -->

<img src="3. Mini Projects/Super-Resolution/SRGAN_results.png" alt="Result Image" style="display:block; margin: -10% auto 0; transform: rotate(90deg);width: 60%;"/>


## Inversion and Adverserial Attack
<p> This project demonstrates neural network inversion and adversarial attacks on the MNIST dataset. It utilizes a simple sigmoid-based convolutional neural network (CNN) model trained on this dataset. The project employs image inversion through gradient descent to reconstruct input images that yield desired outputs from the model. Additionally, it showcases a targeted adversarial attack, where images are manipulated to resemble a specific target class, fooling the model into consistently predicting this class. The effectiveness of the attack is evaluated using a confusion matrix, highlighting the model's vulnerabilities.<br><a href="3. Mini Projects/Inversion and Adverserial Attack/Report.pdf">Report</a></p>


<div style="margin: auto; width: 50%;">
<table style="text-align: center;">
    <tr>
    <td><b>0</b></td>
    <td><b>1</b></td>
    <td><b>2</b></td>
    <td><b>3</b></td>
    <td><b>4</b></td>
    <td><b>5</b></td>
    <td><b>6</b></td>
    <td><b>7</b></td>
    <td><b>8</b></td>
    <td><b>9</b></td>
</tr>
    <tr>
        <td>858</td><td>0</td><td>4629</td><td>188</td><td>4</td><td>142</td><td>48</td><td>0</td><td>53</td><td>1</td>
    </tr>
    <tr>
        <td>1</td><td>0</td><td>6692</td><td>37</td><td>0</td><td>0</td><td>0</td><td>6</td><td>6</td><td>0</td>
    </tr>
    <tr>
        <td>36</td><td>53</td><td>5405</td><td>81</td><td>91</td><td>16</td><td>60</td><td>85</td><td>100</td><td>31</td>
    </tr>
    <tr>
        <td>232</td><td>3</td><td>5009</td><td>385</td><td>12</td><td>102</td><td>236</td><td>7</td><td>135</td><td>10</td>
    </tr>
    <tr>
        <td>16</td><td>4</td><td>5193</td><td>217</td><td>144</td><td>18</td><td>13</td><td>3</td><td>161</td><td>73</td>
    </tr>
    <tr>
        <td>52</td><td>0</td><td>4173</td><td>963</td><td>7</td><td>53</td><td>53</td><td>8</td><td>89</td><td>23</td>
    </tr>
    <tr>
        <td>31</td><td>0</td><td>5755</td><td>45</td><td>4</td><td>18</td><td>52</td><td>0</td><td>13</td><td>0</td>
    </tr>
    <tr>
        <td>144</td><td>7</td><td>5686</td><td>95</td><td>8</td><td>58</td><td>47</td><td>21</td><td>63</td><td>136</td>
    </tr>
    <tr>
        <td>35</td><td>3</td><td>5494</td><td>138</td><td>10</td><td>39</td><td>37</td><td>2</td><td>87</td><td>6</td>
    </tr>
    <tr>
        <td>28</td><td>3</td><td>5362</td><td>390</td><td>108</td><td>6</td><td>0</td><td>33</td><td>12</td><td>7</td>
    </tr>
</table>
<p>The following confusion matrix illustrates the effectiveness of the targeted attack on the model,
which consistently predicts the inputs belonging to the <b>target class 2</b>.</p>
</div>


## Judging a Book by its Cover
<p> The project develops a multi-modal model for a book cover dataset to predict book genres, combining BERT for text and ResNet for images, this effectively integrates
textual and visual information in the latent space before final classification. The model achieved a training accuracy of 54%, with Class 15 performing best (precision 0.87, recall 0.95), while Class 21 struggled significantly (precision 0.35, recall 0.02). On test data, overall accuracy achieved was 48%, with an average precision and recall of 49%.<br><a href="3. Mini Projects/Judging a Book by its Cover/Report.pdf">Report</a></p>


## Artistic Image Enhancement and Style Transfer, Image Quantization
<p> This assignment had three parts.
<br><a href="3. Mini Projects/Judging a Book by its Cover/Report.pdf">Report</a>

#### Part 1
The goal was to transform real photos into non-photorealistic, painting-like images by enhancing light-shadow contrast, emphasizing lines, and using vivid colors. This process consists of two main steps: Artistic Enhancement and Color Adjustment.

<ul>
    <li><strong>Artistic Enhancement Step:</strong>
        <ul>
            <li><strong>Shadow Map Generation:</strong> Convert the input image from RGB to HSI color space and create a shadow map by assigning flags based on light or shadow areas. Merge this shadow map with the original image.</li>
            <li><strong>Line Draft Generation:</strong> Convert the original image to grayscale, apply bilateral filtering to smooth it while preserving edges, and use Sobel filters to detect edges. Produce a binary line draft image through thresholding to emphasize important lines.</li>
        </ul>
    </li>
    <li><strong>Color Adjustment Step:</strong> Create a chromatic map by decomposing the image in LAB color space to remove lightness influence and enhance the RGB components with this map. Apply linear saturation correction to address brightness changes.</li>
    <li><strong>Final Output:</strong> Combine the enhanced image and the line draft to create the final artistic rendered image, effectively balancing color and line emphasis.</li>
</ul></p>


<!-- <div style="display: flex; justify-content: center;">
    <img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/input_img.jpg" alt="Result Image 1" style="margin: 0 10px; width:50%"/>
    <img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/Artistic_img.jpg" alt="Result Image 2" style="margin: 0 10px;width:50%"/>
</div> -->
<table style="text-align: center;">
    <tr>
    <td><img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/input_img.jpg" alt="Result Image 1" style="margin: 0 10px; width:100%"/></td>
    <td><img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/Artistic_img.jpg" alt="Result Image 2" style="margin: 0 10px;width:100%"/></td>
    </tr>
</table>

#
#### Part 2 
Focuses on analyzing a mechanism for rendering output through image quantization, specifically using the median-cut method for color quantization. The process involves working with a 24-bit input image, which contains 8 bits each for the red, green, and blue components. For comparison, results are generated both with and without Floyd-Steinberg dithering.

The assignment requires implementing both the median-cut algorithm and Floyd-Steinberg dithering from scratch, avoiding the use of direct built-in functions. The median-cut quantization is approached using a divide-and-conquer strategy. 

The result below shows 12 bit quantization of the original 24 bit image.
<!-- <div style="display: flex; justify-content: center;">
    <img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/Artistic_img.jpg" alt="Result Image 1" style="margin: 0 10px;width:50%"/>
    <img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/quantization12colors.png" alt="Result Image 2" style="margin: 0 10px; width:50%"/>
</div> -->
<table style="text-align: center;">
    <tr>
    <td><img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/Artistic_img.jpg" alt="Result Image 1" style="margin: 0 10px;width:100%"/></td>
    <td><img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/quantization12colors.png" alt="Result Image 2" style="margin: 0 10px; width:100%"/></td>
    </tr>
</table>

#### Part 3
This part focuses on transferring color from a source image to a grayscale image using swatches. The objective is to utilize the similar intensity values between the source and target images within the respective swatches to effectively paint the entire target image. 

A basic GUI application was developed to effectively select n swatches from both the source and the target image as shown below.
<div style="text-align:center;">
<img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/swatches.png" alt="Result Image" style="display:block; margin:auto;"/>
Swatch Selection

<!-- </div>
<div style="margin: auto;width:50%; text-align:center;">
<img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/colorTransferred.png" alt="Result Image" />
Grayscale image after color transfer.
</div> -->
<table style="text-align: center;">
    <tr>
    <td><img src="3. Mini Projects/Artistic Image Enhancement and Style Transfer, Image Quantization/colorTransferred.png" alt="Result Image 1" style="margin: 0 10px;width:60%"/></td>
    </tr>
    <td>Grayscale image after color transfer.</td>
</table>

#
#
#
# ASSIGNMENTS

<p><b>Implementing Linear Regression, Logistic Regression, GDA.<b> 
<a href="4. Course Assignments/Linear Regression, Logistic Regression, GDA/Report.pdf">Report</a>
</p>

<p><b>Classification using Naive Bayes, SVM.<b> 
<a href="4. Course Assignments/Naive Bayes, SVM/Report.pdf">Report</a>
</p>

<p><b>Exploring Decision Trees, Random Forests, Gradient Boosted Trees and Implementing Neural Network.<b> 
<a href="4. Course Assignments/Decision Trees, Random Forests, Gradient Boosted Trees, Neural Network/Report.pdf">Report</a>
</p>

<p><b>Exploring Activation Functions, Transfer Learning, Optimisers.<b> 
<a href="4. Course Assignments/Exploring Activation Functions, Transfer Learning, Optimisers /Report.pdf">Report</a>
</p>

<p><b>UC Merced Classifier.<b> 
<a href="4. Course Assignments/UC Merced Classifier/Report.pdf">Report</a>
</p>


<p><b>Implementing Bilinear Interpolation.<b> 
<a href="4. Course Assignments/Implementing Bilinear Interpolation/Report.pdf">Report</a>
</p>

<p><b>Face Expression Transfer.<b> 
<a href="4. Course Assignments/Face Expression Transfer/Report.pdf">Report</a>
</p>

