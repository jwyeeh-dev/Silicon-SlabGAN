# Inverse Design of Surface Geometries using Generative AI Models (WGAN)

<img width="1000" align="center" alt="System Overview" src="https://github.com/jwyeeh-dev/Silicon-SlabGen/assets/99489807/439d766e-0268-42a6-a2ad-b20dc2568a89">


## Motivation
- Considering the material space for designing new materials is laborintensive and inefficient to manually find materials with desired properties.
- Inverse design technology using AI generative models[1-5] that can create new structures based on target physical properties is able to overcome this problem.


## Objectives
- Constructing the high-quality slab structure database (DB).
- Developing the Si slab generator with the targeted ionization energy using AI generative models.


## Datasets
- Generating Si Slab DB with surface properties obtained by the extended Hubbard U+V approach[6].
- 23,137 Silicon Slab geometry with JSON file format.
- To enhance the performance of generative models, we are augmented to 80,000 Databases by Supercell operation and Translation operation method to insert at the generative model.

<div align="center">
  <img width="750" alt="스크린샷 2023-10-13 오후 1 55 29" src="https://github.com/jwyeeh-dev/Silicon-SlabGen/assets/99489807/b42b2aca-de58-4b06-a93b-caa99cc0ba61">
</div>


## Network Structures

### 1. Conditional Variational AutoEncoder (CVAE)
<div align="center">
  <img width="1000" alt="스크린샷 2023-10-13 오후 1 30 28" src="https://github.com/jwyeeh-dev/Silicon-SlabGen/assets/99489807/657e5bae-7e84-4510-975a-03efeac5e747">
</div>
<br>

### 2. Wesserstain Generative Adverserial Network (WGAN)
<div align="center">
  <img width="1000" align="center" alt="스크린샷 2023-10-13 오후 1 31 59" src="https://github.com/jwyeeh-dev/Silicon-SlabGen/assets/99489807/d34da36e-46fa-4202-9c2d-d09dab0b907d">
</div>
<br>

## Usage
### command for WGAN
```
$ cd pytorch
$ python train.py --n_epochs 301 --batch_size 32 --load_generator models.py --load_discriminator models.py --load_q models.py 
```

### Options
```
parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=301, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--d_lr', type=float, default=0.00005, help='adam: learning rate')
    parser.add_argument('--q_lr', type=float, default=0.000025)
    parser.add_argument('--g_lr', type=float, default=0.00005)
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--latent_dim', type=int, default=512, help='dimensionality of the latent space')
    parser.add_argument('--model_save_dir', type = str, default = './model_cwgan_slab/')
    parser.add_argument('--load_model', type = bool, default = False)
    parser.add_argument('--load_generator', type = str)
    parser.add_argument('--load_discriminator', type = str)
    parser.add_argument('--load_q', type = str)
    parser.add_argument('--constraint_epoch', type = int, default = 10000)
    parser.add_argument('--gen_dir', type=str, default='./gen_image_cwgan_slab/')
    parser.add_argument('--trainingdata', type=str, default='./slab_1000.pickle')
    parser.add_argument('--input_dim', type=str, default=512+1002+1)
    opt = parser.parse_args()
```


## References
- *Sanchez-Lengeling and Aspuru-Guzik, Science 361, 360 (2018).*
- *Noh, Kim, Stein, Sanchez-Lengeling, Gregoire, Aspuru- Guzik, Jung, Matter 1, 1370 (2019).*
- *Kim, Lee, and Kim, Sci. Adv. 6, 9324 (2020).*
- *Court, Yildirim, Jain, and Cole, J. Chem. Inf. Model. 60, 4518 (2020).*
- *Kim, Noh, Gu, Aspuru-Guzik, and Jung, ACS Cent. Sci. 6, 1412 (2020).*
- *Lee and Son, Phys. Rev. Res 2, 043410 (2020).*


## Citation
```
@misc{HWANG_2022_ESCW,
  author = {Jae-Yeong Hwang, Weon-Gyu Lee, Sang-Hoon Lee, Young-Woo Son, and Jung-Hoon Lee},
  title = {Inverse Design of Surface Geometries using Generative AI Models},
  howpublished = {Presented at \textit{The KIAS Electronic Structure Calculation Workshop 2022}},
  year = {2022},
  note = {Date of Presentation: July 7, 2022. Accessed: Date}
}
```
