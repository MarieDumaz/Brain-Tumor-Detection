# Brain-Tumor-Detection

In this repository, you will find the work I have done for my class on Pattern Recognition. 

The goal was to use Machine Learning to automatically detect if an MRI scan shows tumorous tissues or not. I used data from The Cancer Image Archives with a total of 917 brain MRI pictures [1]. You can find those pictures in the folder "original_images", where I manually classified them, to be able to label them. However the pictures are randomly sampled when training and testing the models.

I compared the performances of Support Vector Machine (SVM) and K-Nearest Neighbors algorithms, as well as the impact of different pre-processing methods such as resizing and skull-stripping.

Bahadure, Nilesh Bhaskarrao, et al [2] inspired the methodology for skull-stripping and analysis while the python code offered on GitHub [3] by Muhammad Fathy gave a base to implement the code.

### References

[1] Schmainda KM, Prah M (2018). Data from Brain-Tumor-Progression. The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2018.15quzvnb

[2] Bahadure, Nilesh Bhaskarrao, et al. “Image Analysis for MRI Based Brain Tumor Detection and Feature Extraction Using Biologically Inspired BWT and SVM.” International Journal of Biomedical Imaging, vol. 2017, 6 Mar. 2017, pp. 1–12., doi:10.1155/2017/9749108.

[5] MuhammadFathy. “MuhammadFathy/Study-of-Detection-Brain-Tumor-with-Machine-Learning.” GitHub, github.com/MuhammadFathy/Study-of-Detection-Brain-Tumor-with-machine-learning.
