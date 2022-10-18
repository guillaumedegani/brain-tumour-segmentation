# Brain Tumour Segmentation and Classification

The idea behind this project was to developpe a process similar to what would happen at a hospital. Simply put a patient would get an MRI a deep learning model would then be tasked to find the tumor in the MRI if it exsist. Once the tumour is found we proceed by doing some feature extraction and with these feature we can apply a Machine Learning Model to classifie the type of tumours.

### Dataset

The dataset [[1]](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) contains 3064 T1-weighted contrast-inhanced images with three kinds of brain tumor. In addition to the MRI's images we also have access to the toumor mask which is necessary to train de Segmentation model. But also the label so which type of tumour is in the MRI which we need to train the Classification model.

Below an example of the types of tumour from the dataset used:

Meningioma      | Glioma |  Pituatary Tumour
:-------------------------:|:-------------------------:|:-------------------------:
![](/assets/glioma.png)  |  ![](/assets/meningioma.png) |  ![](/assets/pituatary.png)

It is important to note that making the diffrence between each type of tumor is important since they behave diffrently.

* `Meningiomas` are benign most of the time they arise from the meninges — the membranes that surround the brain and spinal cord.

* `Gliomas` are more complexe since if they are malignant the diagnosis can be really bad since they can evolve pretty quickly. However these type of tumor can also be benign in the sens that they don't grow fast enough to be a threat. 

* `Pituatary tumors` are abnormal growths that develop in your pituitary gland. Some pituitary tumors result in hormone imbalance wich can impact some important functions in the patients body. 

### Segmentation Model

In this application I decided to use the U-Net [[2]](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) model which is frequently used for segmentation when the object that needs to be detected is fairly small in the image, which is our case.

![](/assets/u-net-architecture.png) 

As for all deep learning problems the choice of Loss function is extremly important, in this case using "traditional" loss function such as MSE or Binary Cross Entropy would leave to terrible results. This is what lead me to the choice of the *Dice Loss Function* [[3]](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/](https://arxiv.org/pdf/2006.14822.pdf)) which is frequently used in these types of problems.

The formula is the followning:

$$Dice = \frac{2\times TP}{(TP+FP) + (TP+FN)}$$ 
with :

`TP` True positve

`FP` False Positive

`FN` False Negative

Once the model is setup and the loss function is defined it is finally time to train the model to do so we used Google Colab to train our model in the could using GPU's. The first training session was donne on 90% of the dataset for 250 epochs using the 10% leftover for our validation.

Underneath we can see the prediction made by our model after the first training session:

![](/assets/first_training.png)

The results aren't very good and this was predictable because we trained a fairly big model from scrath on only 1000 images. It seems that the model is having a hard time differenciating the tumour from the skull. To solve some of these problems I decided to use data augmentation to increase the size of my dataset with the hope that it will lead to better performance. 

### Data Augmentation

Using the albumentations library the dataset was increased to roughly 40 000 images using a combination of the tranformation described in the table bellow. 

Zoom             |  Holes |  Rotated
:-------------------------:|:-------------------------:|:-------------------------:
![](/assets/data_augmented/rotated.png)  |  ![](/assets/data_augmented/holed.png) |  ![](/assets/data_augmented/zoomed.png)
![](/assets/data_augmented/mask_zoomed.png)  |  ![](/assets/data_augmented/mask_holed.png) |  ![](/assets/data_augmented/mask_rotated.png)

Now using this enhanced dataset on 4 times 100000 random images for 250 epochs we notice some better results as seen in the table bellow :

Input             |  Ground Truth |  Prediction
:-------------------------:|:-------------------------:|:-------------------------:
![](/assets/glioma.png)  |  ![](/assets/mask_12.png) |  ![](/assets/mask_github.png)

Note: The artefacts on the prediction a due to the fact that the model was train on collab so the images that to be shrunk down from 512x512 to 256x256 for time sake. But this proves that with more computing power this method would work on real size images.

### Classification

Once the tumor has been isoloated in the brain we can proceed to a feature extraction and with these features we can then train Machine Learning models to classfie the tumor.

Using Various function on the masks I came up with the following sets of features for my first draft at classification:

- `Width` of the tumor which is an indication of it's shape and size.
- `Height` of the tumor which is an indication of it's shape and size.
- `avgPosX` the position on the X axis of the tumour in the brain.
- `avgPosY` the position on the Y axis of the tumour in the brain.
- `ratioCircular` the ration between the tumors height and with which tells gives indecation on the shape of the tumor.
- `area` the number of pixel tagged as tumor in the mask which gives information about the size of the tumour.

Here are the confusion matrixes comparing two Classifiers, Random Forest and Support Vector Machine (SVM) trainded on the features showed above:

Random Forest             |  SVM
:-------------------------:|:-------------------------:
![](/assets/random_forest.png)  |  ![](/assets/SVM_poly_cut.png)

The results of the SVM are really bad sice the model can't make the diffrence between meningiomas and gliomas. Whereas the Random Forest does a better job at differanciating the two however **56%** of Meningiomas are still diagnosed as Gliomas.

To try and fix this error it's important to understand when and why the model is making a mistake here bellow are some example Gliomas and Meningiomas that are missdiagnosed.

Meningioma             |  Glioma |  Necrotic Glioma
:-------------------------:|:-------------------------:|:-------------------------:
![](/assets/glioma.png)  |  ![](/assets/early_glioma.png) |  ![](/assets/necro_glioma.png)


We can see that Gliomas and Meningiomas have a fairly similar shape and size especially for early stage Gliomas eg non necrotic. If the Glioma is necrotic the texture of the tumour becomes less uniform this indecates that using texture feature might help with the diagnosis. To do we use the Grey Level Co-occurrence matrix (GCLM) to extract some new features related to the texture of the tumor as described in [[4]](https://www.researchgate.net/publication/285737882_GLCM_textural_features_for_Brain_Tumor_Classification) the new features used are the following :

- `entropy` Measure of uncertainty withinan image
- `homogeneity` Measures the smoothness of the image texture
- `contrast` Overall amount of local grey level variation within a window
- `dissimilarity` Measure of the local variation
- `energy` Provides the sum of squared elements in the GLCM. Also known as uniformity or the angular second moment.
- `correlation` Measurement of linear dependency of grey levels within an image

Before adding all these features to the model I thought using feature selection might be helpfull to improve the precision of the model. Using Bacward Elimination wich consists in cheking the model performance and then removing the worst performing features until the model perform well. The metric to used to evaluate the usefulness of a feature was the p-value wehter it was abovz 0.05 or not.

Using this method we selected the following features which optimize the performance of the model:

`['width', 'height', 'avgPosY', 'ratioCircular', 'area', 'entropy', 'homogeneity', 'dissimilarity', 'energy', 'correlation']
`

The final results of the model are summerized in the Confusion Matrix Bellow, the model performance has slightly improved reaching an accuarcy of roughly `85%` however there still alot of Meningiomas which are Diagnosed as Gliomas. But as described above this in explainable by the fact that Meningiomas and Grade I Gliomas are fairly similar in there appearence. And using follow up MRI's the model would see wether the tumour was evolving which is an indicator of malignancy which are usually linked to Gliomas.

Final Results             |
:-------------------------:|
![](/assets/Final_Results.png)

# Conclusion

The model developped in this project perform averagly at best which is explainable by the fact that this project was done in short amount of time. There are alot of improvment that could be done to this project for example using a larger dataset to improve the performance of the models. Using a mix of human selected features and deep learning to make the classification better. Finnaly using the model on several brain slices from the same MRI and selcting the type on these diffrent slices.














