<h1 align="center"><a name="section_name">Surveillance GUI</a></h1>

<div align="justify">
A GUI model to aid long range surveillance.  With our literature, we understand this model performs better  without training dataset or ground truth values. We wish to improve the accuracy of the model and reduce the time complexity.  In our model, we utilised few steps from the references cited below.
  
</div>


## Work under Progress

Enabling IP Camera in-built modules for better monitoring and control

## Help Required

1. Automatic detection of the connected surveillance devices to be enabled as a list
2. Modify the code to utilise GPU resources (if available) to reduce time complexity

## Build Status

<img src="https://img.shields.io/badge/build-passing-brightgreen"/> <img src="https://img.shields.io/badge/code-latest-orange"/> <img src="https://img.shields.io/badge/langugage-python-blue"/>


## Intended Use

<img src="https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white"/> <img src="https://img.shields.io/badge/linux-E95420?style=for-the-badge&logo=linux&logoColor=white"/>

## Screenshot


GUI

<img src="https://raw.githubusercontent.com/Surveillance-NWB/Surveillance/main/GUI_Screenshots/GUI_Choice.png" alt="check box functionality" style="height: 350px; width:650px;"/>

<img src="https://raw.githubusercontent.com/Surveillance-NWB/Surveillance/main/GUI_Screenshots/GUI_updated.png" alt="check box functionality" style="height: 350px; width:650px;"/>


## Hardware and Software Requirements

<div align="justify">
The following frameworks were utilized in the building of this code.
</div>


<img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white"/>  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>


<div align="justify">
The following tools were utilized in the building of this code.
</div>


<img src="https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white"/> <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white"/>


<br>

>  Pro Tip: Press 'q' to release the connected suveillance device from capturing input for the selected module / feature



</div>

## Installation of Dependencies / Development Setup

### Pre-requisites


<div align="justify">

The code is developed with Python ver.3.8 and `pip` ver.21.0.1 in Windows OS and the same is tested on linux OS too. The necessary packages and frameworks can be installed from the *Requirements* directory.  However, one can follow the below mentioned steps in case of any errors.


```Python
pip install -r requirements.txt
``` 


Firstly, check the version of Python on your system using:


```Python
python --version
``` 

If you wish to change / upgrade the version or install Python afresh, visit https://www.python.org/downloads/. 

`pip` is a package-management system written in Python used to install and manage software packages. It connects to an online repository of public packages, called the Python Package Index. pip can also be configured to connect to other package repositories.  One can check `pip` version using:

```Python
pip --version
```

If you wish to install `pip` afresh, do:

```Python
python3 -m pip install --upgrade pip
```

or

```Python
sudo apt install python3-pip
```

Installing the necessary packages and depencies is a pre-requisite.  The setup itself varies according to the OS, though the code is really the same.  Yet, the GUI is builded with different libraries in runtime, hence it results in differrent appearances of the same GUI accroding to OSs.


<details>
<summary>Windows OS</summary>

---

The `tkinter` package (“Tk interface”) is the standard Python interface to the Tk GUI toolkit. The `Tk interface` is located in a binary module named `_tkinter`. It is usually a shared library (or DLL), but might in some cases be statically linked with the Python interpreter.  The `cffi` module is used to invoke `callback` methods inside the program.

```Python
pip install tk
python3 -m pip install cffi
```

`Pillow` is a Python Imaging Library (`PIL`), which adds support for opening, manipulating, and saving images. The current version identifies and reads a large number of formats.  It supports wide variety of images such as “jpeg”, “png”, “bmp”, “gif”, “ppm”, “tiff”.

```Python
python3 -m pip install --upgrade Pillow
```

`OpenCV` is a huge open-source library for computer vision, machine learning, and image processing. `OpenCV` supports a wide variety of programming languages like Python, C++, Java, etc. It can process images and videos to identify objects, faces, and so on. The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms.


```Python
pip install opencv-python
```


`NumPy` is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.  By reading the image as a `NumPy` array ndarray, various image processing can be performed using NumPy functions.


```Python
pip3 install numpy
```


`Imutils` are a series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with `OpenCV` in Python.


```Python
pip3 install imutils
```


</details>


<details>
<summary>Linux OS</summary>

---

The `tkinter` package (“Tk interface”) is the standard Python interface to the Tk GUI toolkit. The `Tk interface` is located in a binary module named `_tkinter`. It is usually a shared library (or DLL), but might in some cases be statically linked with the Python interpreter.  The `cffi` module is used to invoke `callback` methods inside the program.

```Python
apt-get install python-tk 
sudo apt-get install python-setuptools
sudo apt-get install -y python-cffi
```

`Pillow` is a Python Imaging Library (`PIL`), which adds support for opening, manipulating, and saving images. The current version identifies and reads a large number of formats.  It supports wide variety of images such as “jpeg”, “png”, “bmp”, “gif”, “ppm”, “tiff”.

```Python
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow
```


`OpenCV` is a huge open-source library for computer vision, machine learning, and image processing. `OpenCV` supports a wide variety of programming languages like Python, C++, Java, etc. It can process images and videos to identify objects, faces, and so on. The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms.


```Python
sudo apt-get install python3-opencv
```

`NumPy` is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.  By reading the image as a `NumPy` array ndarray, various image processing can be performed using NumPy functions.


```Python
pip3 install numpy
```


`Imutils` are a series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with `OpenCV` in Python.


```Python
pip3 install imutils
```



</details>



</div>


## References

1. https://github.com/SYangChen/Detection-of-Moving-Object 
2. https://github.com/subeeshvasu/Awesome-Deblurring
3. https://github.com/samuelt121/Turbulence-Mitigation


[Go to Top](#section_name)
