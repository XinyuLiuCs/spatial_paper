# The official implementation of the paper "CA-ATP: Content-Adaptive Image Retouching Guided by Attribute-Based Text Representation".


## abstract

Image retouching has garnered significant attention due to its ability to achieve professional-level visual enhancement. Existing approaches mainly rely on uniform pixel-wise color mapping across entire images, neglecting the inherent color variations induced by image content. This limitation consequently hinders existing approaches from achieving adaptive retouching that accommodates both diverse color distributions and user-defined style preferences. To address these challenges, we propose a novel Content-Adaptive image retouching method guided by Attribute-based Text Representation (CA-ATP). Specifically, we propose a content-adaptive curve mapping module, which leverages a series of basis curves to establish multiple color mapping relationships and learns the corresponding weight maps, enabling content-aware color adjustments. The proposed module can capture color diversity within the image content, allowing identical color values to receive distinct transformations based on their spatial context. In addition, we propose an attribute text prediction module that generates text representations from multiple image attributes to quantitatively represent user-defined style preferences. These attribute-based text representations are subsequently integrated with visual features via a multimodal model, guiding image retouching to align with user preferences. Extensive experiments on several public datasets demonstrate that our method achieves state-of-the-art performance. Our source code can be found in the supplementary material.
\end{abstract}


## Getting Started

### Training

```bash
python train.py
```

### Testing

```bash
python test.py
```

## Citation
