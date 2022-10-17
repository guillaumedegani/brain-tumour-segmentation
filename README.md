# Brain Tumour Segmentation and Classification

The idea behind this project was to developpe a process similar to what would happen at a hospital. Simply put a patient would get an MRI a deep learning model would then be tasked to find the tumor in the MRI if it exsist. Once the tumour is found we proceed by doing some feature extraction and with these feature we can apply a Machine Learning Model to classifie the type of tumours.

### Dataset

The dataset [[1]](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) contains 3064 T1-weighted contrast-inhanced images with three kinds of brain tumor. In addition to the MRI's images we also have access to the toumor mask which is necessary to train de Segmentation model. But also the label so which type of tumour is in the MRI which we need to train the Classification model.

Below an example of the types of tumour from the dataset used:

Glioma             |  Meningioma |  Pituatary Tumour
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

# Data Augmentation

Rotation             |  Holes |  Zoom
:-------------------------:|:-------------------------:|:-------------------------:
![](/assets/data_augmented/rotated.png)  |  ![](/assets/data_augmented/holed.png) |  ![](/assets/data_augmented/zoomed.png)









