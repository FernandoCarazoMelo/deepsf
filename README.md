
<!-- ABOUT THE PROJECT -->
# DeepSF

A Deep Neural Network to predict cell transcriptome from genome.

![](/docs/deepsf_dnn.drawio.svg)


# Introduction

The main objective of this project is to build a deep neural network (DNN) that models how the transcriptome of cancer related genes is expressed from the splicing factors and the genome. An autoencoder was also trained and added to the final model architecture to study the cellular context of protein coding genes in different tumor tissues. Finally, we tried to identify which splicing factors regulate which genes by interpreting the parameters of the final model with a tool called Deep Learning of Features (DeepLIFT).

Figure 2. The Deep Neural Networks developed in this project.
Deep learning models has already been used to identify gene expression level (Lirong Zhang, 2022), discover disease-relevant modules of genes (Sanjiv K. Dwivedi, 2020), inferring transcription factor (TF) binding sites (Peter K.Koo, 2020), predict TFs from protein sequences (Gi Bae Kim, 2020) or predict transcriptomic profiles from TF expression (Rasmus Magnusson, 2022). However, the use of the gene expression of known SFs to unveil the transcriptome regulation is a novel approach.


<!-- GETTING STARTED -->
# Deep SF architecture

The DNN model we have developed has three main parts.


## 1. Main DNN

This model has two inputs colored in blue: the expression of the SFs that are the main input of the model and the gene expression of each one of the isoforms. Between the input with the SFs expression and the output there are 2 hidden layers (L1 and L2) with 183 nodes and 82 nodes respectively. All nodes are densely connected by linear functions with ReLU activation function. After L1 and L2, two batch normalizations were performed and the gene expression for each isoform was added to the last linear function between L2 and the output.


## 2. Autoencoder to capture cell context

In (Sanjiv K. Dwivedi, 2020) the researchers were able to discover disease- relevant modules of genes by training a Deep autoencoder from large
transcriptional data.

An autoencoder is a neural network (NN) model that is used to capture the most relevant information from the input data (encodings). The input data and the output to be predicted is the same, which is why it is considered unsupervised learning. Since the prediction is not compared to a label but to the input data itself. In the end, what is obtained with this NN is a representation of the input data in a lower dimension (Bandyopadhyay, 2022).

In our case, in order to obtain a model capable of learning particular expression characteristics of each tumor tissue, an autoencoder was trained based on the [paper] architecture with 3 hidden layers and 512 nodes per layer (with which they managed to capture around 95% of the variance of the gene expression). To do so the expression of 19,594 protein coding genes was used as an input. Two
batch normalizations were also introduced in the autoencoder architecture.
The trained model with its weights was saved to be used as transfer learning
when designing the final model.


## 3. DeepSF & AEensemble
Finally, a model was developed that joined the 2 hidden layers of the first model with the knowledge acquired in the training of the autoencoder. To achieve this, the last layer of the autoencoder was removed and it was also connected to the last linear layer of the model.




# Prerequisites

This is an example of how to list things you need to use the software and how to install them.
<!-- * npm
  ```sh
  npm install npm@latest -g
  ``` -->

## Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
<!-- 3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ``` -->





<!-- USAGE EXAMPLES -->
# Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.




<!-- CONTRIBUTING -->
# Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request





<!-- LICENSE -->
# License

Distributed under the MIT License. See `LICENSE.txt` for more information.





<!-- CONTACT -->
## Contact

<!-- Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name) -->

<!-- ACKNOWLEDGMENTS
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p>


