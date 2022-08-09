## Enhancing ResNet Image Classification Performance by using PHM
<p align="justify">The advantages of using hypercomplex inspired networks have not been studied to the densely connected backend of hypercomplex Networks. This paper studies ResNet architectures and 
incorporates parameterized hypercomplex multiplication (PHM) into the backend of residual, quaternion, and vectormap convolutional neural networks to assess the effect. We show that PHM does improve classification accuracy performance on several image datasets, including small, low-resolution CIFAR 10/100 and large high-resolution ImageNet and ASL, and can achieve state-of-the-art accuracy for hypercomplex networks.</p>

### Proposed QPHM Networks
<p align="justify">We propose a new fully hypercomplex model in lieu of hypercomplex CNNs that use a real-valued backend dense layer. That is, we replace the dense layer with a PHM layer to enjoy the benefits of hypercomplex weight sharing throughout the network. To match dimensions with frontend networks, we used a PHM layer at four dimensions with the quaternion network and a PHM layer at five dimensions with the three dimensional vectormap network. In some cases, we also needed to use a PHM layer at five dimensions with quaternion networks. But we couldn't use a three dimensional PHM layer as the output classes must be divisible by the dimensions in the PHM operation. Similarly, this PHM (both 4D, and 5D) dense layer is applied in the backend of original ResNet \cite{he2016deep} which we named RPHM (ResNet-with-PHM). </p>
<p align="center"> <img src="https://github.com/nazmul729/QPHM-VPHM/blob/main/figures/QARNET.PNG" width="550" title="Full Hypercomplex Network"></p>

### Experimental Results
Image classification performance on the CIFAR benchmarks for 50-layer architectures-

| Model    | Dataset  | Parameters | FLOPS |  Latency | Validation Accuracy |
| -------- | -------- | :---------:|:-----:|:--------:|:-------------------:|
| RPHM-50  | Cifar10  |   20.6M    | 1.29G |  0.46ms  |      95.59          | 
| QPHM-50  | Cifar10  |   18.07M   | 1.44G |  0.96ms  |      95.59          | 
| VPHM-50  | Cifar10  |   15.5M    | 1.15G |  0.76ms  |      95.48          | 
| QPHM-18-2| Cifar10  |   -        | -     |  -       |      96.24          | 
| VPHM-50-2| Cifar10  |   -        | -     |  -       |      96.63          | 
| RPHM-50  | Cifar100 |   20.6M    | 1.29G |  0.46ms  |      79.21          | 
| QPHM-50  | Cifar100 |   18.07M   | 1.44G |  0.96ms  |      80.25          | 
| VPHM-50  | Cifar100 |   15.5M    | 1.15G |  0.76ms  |      79.91          | 
| QPHM-18-2| Cifar100 |   -        | -     |  -       |      81.45          | 
| VPHM-50-2| Cifar100 |   -        | -     |  -       |      82.00          | 

