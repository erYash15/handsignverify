handwritten-signature-verification

Argument passed are image1, image2 & modelname(VGG16, ResNet, AlexNet) to return match/unmatch. 

Handwritten Signature Verification is the demonstration library for matching 2 signature images on perticulary dataset.

### Disclaimer

May not be used for production.


- **Homepage:** https://github.com/eryash15/handsignverify 
- **Website of Gesis:** https://pypi.org/project/handwritten-signature-verification

## Installation

You have multiple options for the installation on your local machine.

If you want to become a contributor in this library and you want to apply changes or add new functions to the library you can take a look on the development guide from the website.
Contact me at eryash15@gmail.com.

If you just want to install the library on your system to work with it, you can do so with:
    
        $ pip install handwritten-signature-verification=0.0.3
        
## Tutorials

If you are new to this library and you want to learn about how to use the included methods or you want to learn how they work you can use the tutorials. For a short description of the different tutorials you can find more information in the tutorial README [Tutorial README](/tutorial/README.md). If you want to test the tutorials right now you can click on the binder badge on top of this README.md to open the tutorials in the Gesis Notebooks. 

[Here](/tutorial) you get to the tutorial folder.

        $ from handsignverify import model
        $ model.match_sign(path_img1,path_img2)



## Current State

The face2face toolbox at its current state is an alpha version. If you find any bugs or you have suggestions about how to extend the toolbox with useful new functionalities please let us know by creating a new issue thread on our <a href="https://github.com/eryash15/handsignverify/issues">GitHub Page</a> or contribute a fix or additional methods by yourself as described in the developer guide in the online documentation.
