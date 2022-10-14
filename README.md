# Brain Tumour Segmentation and Classification

The idea behind this project was to developpe a process similar to what would happen at a hospital. Simply put a patient would get an MRI a deep learning model would then be tasked to find the tumor in the MRI if it exsist. Once the tumour is found we proceed by doing some feature extraction and with these feature we can apply a Machine Learning Model to classifie the type of tumours.

### The Dataset

The dataset [[1]](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) contains 3064 T1-weighted contrast-inhanced images with three kinds of brain tumor. In addition to the MRI's images we also have access to the toumor mask which is necessary to train de Segmentation model. But also the label so which type of tumour is in the MRI which we need to train the Classification model.

Below an example of the types of tumour from the dataset used:

Glioma             |  Meningioma |  Pituatary Tumour
:-------------------------:|:-------------------------:|:-------------------------:
![](/assets/glioma.png)  |  ![](/assets/meningioma.png) |  ![](/assets/pituatary.png)

It is important to note that making the diffrence between each type of tumor is important since they behave diffrently.

* **Meningiomas** are benign most of the time they arise from the meninges — the membranes that surround the brain and spinal cord.

* **Gliomas** are more complexe since if they are malignant the diagnosis can be really bad since they can evolve pretty quickly. However these type of tumor can also be benign in the sens that they don't grow fast enough to be a threat. 

* **Pituatary tumors** are abnormal growths that develop in your pituitary gland. Some pituitary tumors result in hormone imbalance wich can impact some important functions in the patients body. 
