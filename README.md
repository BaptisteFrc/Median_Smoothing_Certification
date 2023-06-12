# Certification and Robustness of Neural Networks by Gaussian Smoothing

## Description of the project

This python code enable the user the apply a Gaussian smoothing to any function from R^d to R including such neural networks.
The bench test provides several graphic representations of the smoothed functions and their certification bounds as it is described in "article en question". It also contains many other indicators of the performances of the Gaussian smooting mainly by the quantil method but also by the mean method. A quantifier of the robustness of a function has also been implemented.

## Implementing rules

The python modules mandatory to run the code are described in requirements.txt. There is no need for any other gadget.

## How to use ?

The only file that provides robustness indicators is user_interface.py. The folder "smoothing" manages the smoothing process of a function. The folder "models_neural_network" provides two neural networks that can be used to run tests in the interface section. The folder adversarial_attacks contains a hand-written fast gradiant attack that can be used to test the robustness of functions in the interface section as well.

## Support

If you need any further information, you can contact us at this email adress : aurele.gallard@student-cs.fr

## Credits

This project has been carried out in the context of CentraleSupelec AI project pole. Our team was composed of 5 first year students :
Sarah Lamik, Mathilde Morhain, Yi Zhong, Baptiste François and Aurèle Gallard ; in partnership with a CEA researcher : Aurélien Benoît-Lévy.

## Licence

GPL

## Projet Status

This project, in its academic context, has come to its end.