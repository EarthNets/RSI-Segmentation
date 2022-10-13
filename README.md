
# RSI-Segmentation
<div  align="center">    
 <img src="resources/seg.png" width = "400" height = "140" alt="segmentation" align=center />
</div>


## Changelog
Initial version.

## Benchmark and model zoo

Results and models are available in the [model zoo].

Supported datasets:

- [x] [DFC2020](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest)
- [x] [LandSlide4Sense](https://www.iarai.ac.at/landslide4sense/)
- [x] [GeoNRW](https://github.com/gbaier/geonrw)
- [x] [LoveDA](https://github.com/Junjue-Wang/LoveDA)
- [x] [RSUSS](https://github.com/EarthNets/RSI-MMSegmentation)
- [x] [SEASONET](https://zenodo.org/record/5850307#.Y0cayXbP1D8)
- [x] [SSL4EO_S12](https://github.com/zhu-xlab/SSL4EO-S12)
- [x] [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)

Continually updating


Supported backbones:

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] [HRNet (CVPR'2019)](configs/hrnet)
- [x] [ResNeSt (ArXiv'2020)](configs/resnest)
- [x] [MobileNetV2 (CVPR'2018)](configs/mobilenet_v2)
- [x] [MobileNetV3 (ICCV'2019)](configs/mobilenet_v3)
- [x] [Vision Transformer (ICLR'2021)](configs/vit)
- [x] [Swin Transformer (ICCV'2021)](configs/swin)
- [x] [Twins (NeurIPS'2021)](configs/twins)
- [x] [BEiT (ICLR'2022)](configs/beit)
- [x] [ConvNeXt (CVPR'2022)](configs/convnext)
- [x] [MAE (CVPR'2022)](configs/mae)

Supported methods:

- [x] [FCN (CVPR'2015/TPAMI'2017)](configs/fcn)
- [x] [ERFNet (T-ITS'2017)](configs/erfnet)
- [x] [UNet (MICCAI'2016/Nat. Methods'2019)](configs/unet)
- [x] [PSPNet (CVPR'2017)](configs/pspnet)
- [x] [DeepLabV3 (ArXiv'2017)](configs/deeplabv3)
- [x] [BiSeNetV1 (ECCV'2018)](configs/bisenetv1)
- [x] [PSANet (ECCV'2018)](configs/psanet)
- [x] [DeepLabV3+ (CVPR'2018)](configs/deeplabv3plus)
- [x] [UPerNet (ECCV'2018)](configs/upernet)
- [x] [ICNet (ECCV'2018)](configs/icnet)
- [x] [NonLocal Net (CVPR'2018)](configs/nonlocal_net)
- [x] [EncNet (CVPR'2018)](configs/encnet)
- [x] [Semantic FPN (CVPR'2019)](configs/sem_fpn)
- [x] [DANet (CVPR'2019)](configs/danet)
- [x] [APCNet (CVPR'2019)](configs/apcnet)
- [x] [EMANet (ICCV'2019)](configs/emanet)
- [x] [CCNet (ICCV'2019)](configs/ccnet)
- [x] [DMNet (ICCV'2019)](configs/dmnet)
- [x] [ANN (ICCV'2019)](configs/ann)
- [x] [GCNet (ICCVW'2019/TPAMI'2020)](configs/gcnet)
- [x] [FastFCN (ArXiv'2019)](configs/fastfcn)
- [x] [Fast-SCNN (ArXiv'2019)](configs/fastscnn)
- [x] [ISANet (ArXiv'2019/IJCV'2021)](configs/isanet)
- [x] [OCRNet (ECCV'2020)](configs/ocrnet)
- [x] [DNLNet (ECCV'2020)](configs/dnlnet)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [CGNet (TIP'2020)](configs/cgnet)
- [x] [BiSeNetV2 (IJCV'2021)](configs/bisenetv2)
- [x] [STDC (CVPR'2021)](configs/stdc)
- [x] [SETR (CVPR'2021)](configs/setr)
- [x] [DPT (ArXiv'2021)](configs/dpt)
- [x] [Segmenter (ICCV'2021)](configs/segmenter)
- [x] [SegFormer (NeurIPS'2021)](configs/segformer)
- [x] [K-Net (NeurIPS'2021)](configs/knet)


## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Please see [train.md](docs/en/train.md) and [inference.md](docs/en/inference.md) for the basic usage.
There are also tutorials for [customizing dataset](docs/en/tutorials/customize_datasets.md), [designing data pipeline](docs/en/tutorials/data_pipeline.md), [customizing modules](docs/en/tutorials/customize_models.md), and [customizing runtime](docs/en/tutorials/customize_runtime.md).
We also provide many [training tricks](docs/en/tutorials/training_tricks.md) for better training and [useful tools](docs/en/useful_tools.md) for deployment.

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb) on Colab.

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.


## Contributing

We appreciate all contributions. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Projects in EarthNets

- [Dataset4EO](https://github.com/DeepAI4EO/Dataset4EO): Datasets downloading and loading for Remote sensing community.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{earthnets4eo,
    title={EarthNets: Empowering AI in Earth Observation},
    author={Zhitong Xiong, Fahong Zhang, Yi Wang, Yilei Shi, Xiao Xiang Zhu},
    journal = {arXiv:2210.04936},
    year={2022}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
