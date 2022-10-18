# Brain Tumor Segmentation and Classification

The idea behind this project was to develop a process similar to what would happen at a hospital. Simply put a patient would get an MRI a deep learning model would then be tasked to find the tumor in the MRI if it exists. Once the tumor is found we proceed by doing some feature extraction and with these features we can apply a Machine Learning Model to classify the type of tumors.

### Dataset

The dataset [[1]](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) contains 3064 T1-weighted contrast-enhanced images with three kinds of brain tumor. In addition to the MRI's images we also have access to the tumor mask which is necessary to train the Segmentation model. But also the label so which type of tumor is in the MRI which we need to train the Classification model.

Below an example of the types of tumor from the dataset used:

Meningioma      | Glioma |  Pituitary Tumor
:-------------------------:|:-------------------------:|:-------------------------:
![](/assets/glioma.png)  |  ![](/assets/meningioma.png) |  ![](/assets/pituatary.png)

It is important to note that making the difference between each type of tumor is important since they behave differently.

* `Meningiomas` are benign most of the time they arise from the meninges — the membranes that surround the brain and spinal cord.

* `Gliomas` are more complex since if they are malignant the diagnosis can be really bad since they can evolve pretty quickly. However these types of tumor can also be benign in the sense that they don't grow fast enough to be a threat.

* `Pituitary tumors` are abnormal growths that develop in your pituitary gland. Some pituitary tumors result in hormone imbalance which can impact some important functions in the patient's body.

### Segmentation Model

In this application I decided to use the U-Net [[2]](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) model which is frequently used for segmentation when the object that needs to be detected is fairly small in the image, which is our case.

![](/assets/u-net-architecture.png)

As for all deep learning problems the choice of Loss function is extremely important, in this case using "traditional" loss function such as MSE or Binary Cross Entropy would lead to terrible results. This is what lead me to the choice of the *Dice Loss Function* [[3]](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/](https://arxiv.org/pdf/2006.14822.pdf)) which is frequently used in these types of problems.

The formula is the following:

$$Dice = \frac{2\times TP}{(TP+FP) + (TP+FN)}$$
with :

`TP` True positive

`FP` False Positive

`FN` False Negative

Once the model is set up and the loss function is defined it is finally time to train the model to do so we used Google Colab to train our model in the cloud using GPU's. The first training session was done on 90% of the dataset for 250 epochs using the 10% leftover for our validation.

Underneath we can see the prediction made by our model after the first training session:

![](/assets/first_training.png)

The results aren't very good and this was predictable because we trained a fairly big model from scratch on only 1000 images. It seems that the model is having a hard time differentiating the tumor from the skull. To solve some of these problems I decided to use data augmentation to increase the size of my dataset with the hope that it will lead to better performance.

### Data Augmentation

Using the albumentations library the dataset was increased to roughly 40 000 images using a combination of the transformation described in the table below.

Zoom             |  Holes |  Rotated
:-------------------------:|:-------------------------:|:-------------------------:
![](/assets/data_augmented/rotated.png)  |  ![](/assets/data_augmented/holed.png) |  ![](/assets/data_augmented/zoomed.png)
![](/assets/data_augmented/mask_zoomed.png)  |  ![](/assets/data_augmented/mask_holed.png) |  ![](/assets/data_augmented/mask_rotated.png)

Now using this enhanced dataset on 4 times 100000 random images for 250 epochs we notice some better results as seen in the table below :

Input             |  Ground Truth |  Prediction
:-------------------------:|:-------------------------:|:-------------------------:
![](/assets/glioma.png)  |  ![](/assets/mask_12.png) |  ![](/assets/mask_github.png)

Note: The artifacts on the prediction are due to the fact that the model was trained on collaboration so the images have to be shrunk down from 512x512 to 256x256 for time sake. But this proves that with more computing power this method would work on real size images.

### Classification

Once the tumor has been isolated in the brain we can proceed to a feature extraction and with these features we can then train Machine Learning models to classify the tumor.

Using Various function on the masks I came up with the following sets of features for my first draft at classification:

- `Width` of the tumor which is an indication of its shape and size.
- `Height` of the tumor which is an indication of its shape and size.
- `avgPosX` the position on the X axis of the tumor in the brain.
- `avgPosY` the position on the Y axis of the tumor in the brain.
- `ratioCircular` the ratio between the tumor's height and which gives indication on the shape of the tumor.
- `area` the number of pixels tagged as tumor in the mask which gives information about the size of the tumor.

Here are the confusion matrices comparing two Classifiers, Random Forest and Support Vector Machine (SVM) trained on the features shown above:

Random Forest             |  SVM
:-------------------------:|:-------------------------:
![](/assets/random_forest.png)  |  ![](/assets/SVM_poly_cut.png)

The results of the SVM are really bad since the model can't make the difference between meningiomas and gliomas. Whereas the Random Forest does a better job at differentiating the two however **56%** Meningiomas are still diagnosed as Gliomas.

To try and fix this error it's important to understand when and why the model is making a mistake. Here are some examples of Gliomas and Meningiomas that are misdiagnosed.

Meningioma             |  Glioma |  Necrotic Glioma
:-------------------------:|:-------------------------:|:-------------------------:
![](/assets/glioma.png)  |  ![](/assets/early_glioma.png) |  ![](/assets/necro_glioma.png)


We can see that Gliomas and Meningiomas have a fairly similar shape and size especially for early stage Gliomas eg non necrotic. If the Glioma is necrotic the texture of the tumor becomes less uniform this indicates that using texture features might help with the diagnosis. To do we use the Grey Level Co-occurrence matrix (GCLM) to extract some new features related to the texture of the tumor as described in [[4]](https://www.researchgate.net/publication/285737882_GLCM_textural_features_for_Brain_Tumor_Classification) the new features used are the following :

- `entropy` Measure of uncertainty within an image
- `homogeneity` Measures the smoothness of the image texture
- `contrast` Overall amount of local gray level variation within a window
- `dissimilarity` Measure of the local variation
- `energy` Provides the sum of squared elements in the GLCM. Also known as uniformity or 
the angular second moment.
- `correlation` Measurement of linear dependency of gray levels within an image

Before adding all these features to the model I thought using feature selection might be helpful to improve the precision of the model. Using Backward Elimination which consists in checking the model performance and then removing the worst performing features until the model performs well. The metric used to evaluate the usefulness of a feature was the p-value whether it was above 0.05 or not.

Using this method we selected the following features which optimize the performance of the model:

`['width', 'height', 'avgPosY', 'ratioCircular', 'area', 'entropy', 'homogeneity', 'dissimilarity', 'energy', 'correlation']
`

The final results of the model are summarized in the Confusion Matrix Bellow, the model performance has slightly improved reaching an accuracy of roughly `85%` however there are still a lot of Meningiomas which are Diagnosed as Gliomas. But as described above this is explainable by the fact that Meningiomas and Grade I Gliomas are fairly similar in their appearance. And using follow up MRI's the model would see whether the tumor was evolving which is an indicator of malignancy which are usually linked to Gliomas.

Final Results             |
:-------------------------:|
![](/assets/Final_Results.png)

# Conclusion

The model developed in this project performs averagely at best which is explainable by the fact that this project was done in a short amount of time. There are a lot of improvements that could be done to this project for example using a larger dataset to improve the performance of the models. Using a mix of human selected features and deep learning to make the classification better. Finally using the model on several brain slices from the same MRI and selecting the type on these different slices.




