# TransWi

## News
- [2025-07-31] Code has been released.


## Abstract
Wireless sensing-based human activity recognition (HAR) is a promising technology for smart home and healthcare applications. Recently, transformer-based HAR methods have attracted significant attention because of their powerful ability to capture long-range dependencies. However, traditional transformer models are often large and computationally inefficient. Moreover, they typically rely on pre-trained word embeddings and ignore local spatial relationships in 2D space, which are crucial in spectrograms and have been proved by convolution-based models. In this paper, we present TransWi, a lightweight transformer model designed for the edge computing scenario. TransWi includes a local-aware fusion mechanism and a window attention block to better capture local spatial features. It also uses a dual attention block to learn both temporal and frequency relationships at the same time. To further improve performance, a custom data augmentation strategy is proposed. Extensive experiments on three public datasets—Widar3.0, GeCro, and ISAC—demonstrate the effectiveness of TransWi, achieving in-domain accuracies of 99.54\%, 97.51\%, and 98.45\%, and out-of-domain accuracies of 93.85\%, 93.21\%, and 97.59\%, respectively. On a Raspberry Pi 5, TransWi-S recognizes activities in just 3.77 ms with 10.13 MB of memory, making it ideal for edge computing. The code will be publicly available at https://anonymous.4open.science/r/TransWi-ADF4.
