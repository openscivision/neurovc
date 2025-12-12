[![PyPI - Version](https://img.shields.io/pypi/v/neurovc)](https://pypi.org/project/neurovc/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neurovc)](https://pypi.org/project/neurovc/)
[![PyPI - License](https://img.shields.io/pypi/l/neurovc)](LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/neurovc)](https://pypistats.org/packages/neurovc)
[![Documentation Status](https://readthedocs.org/projects/neurovc/badge/?version=latest)](https://neurovc.readthedocs.io/en/latest/?badge=latest)

## 🚧 Under Development

This project is still in an **alpha stage**. Expect rapid changes, incomplete features, and possible breaking updates between releases.

- The API may evolve as we stabilize core functionality.
- Documentation and examples are incomplete.
- Feedback and bug reports are especially valuable at this stage.

# NeuroVC

![Fig1](https://phflot.github.io/img/fig1.jpg)

Toolbox with utility functions for computer vision setups in neuroscience. The core module contains classes for motion magnification and camera io.

## Citation

If you use this code in work for publications, please cite in the following way.

**1. Camera routines**:

  > Flotho, P., Bhamborae, M., Grun, T., Trenado, C., Thinnes, D., Limbach, D., & Strauss, D. J. (2021). Multimodal Data Acquisition at SARS-CoV-2 Drive Through Screening Centers: Setup Description and Experiences in Saarland, Germany. J Biophotonics.

  BibTeX entry
  ```bibtex
  @article{flotea2021b,
      author = {Flotho, P. and Bhamborae, M.J. and Grün, T. and Trenado, C. and Thinnes, D. and Limbach, D. and Strauss, D. J.},
      title = {Multimodal Data Acquisition at SARS-CoV-2 Drive Through Screening Centers: Setup Description and Experiences in Saarland, Germany},
      year = {2021},
    journal = {J Biophotonics},
    pages = {e202000512},
    doi = {https://doi.org/10.1002/jbio.202000512}
  }
  ```

**2. Motion magnification**:

  > Flotho, P., Heiss, C., Steidl, G., & Strauss, D. J. (2023). Lagrangian motion magnification with double sparse optical flow decomposition. Frontiers in Applied Mathematics and Statistics, 9, 1164491.

  ```bibtex
  @article{flotho2023lagrangian,
    title={Lagrangian motion magnification with double sparse optical flow decomposition},
    author={Flotho, Philipp and Heiss, Cosmas and Steidl, Gabriele and Strauss, Daniel J},
    journal={Frontiers in Applied Mathematics and Statistics},
    volume={9},
    pages={1164491},
    year={2023},
    publisher={Frontiers Media SA}
  }
  ```

  and for facial landmark-based decomposition:

  > Flotho, P., Heiß, C., Steidl, G., & Strauss, D. J. (2022, July). Lagrangian motion magnification with landmark-prior and sparse PCA for facial microexpressions and micromovements. In 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 2215-2218). IEEE.

  ```bibtex
  @inproceedings{flotho2022lagrangian,
    title={Lagrangian motion magnification with landmark-prior and sparse PCA for facial microexpressions and micromovements},
    author={Flotho, Philipp and Hei{\ss}, Cosmas and Steidl, Gabriele and Strauss, Daniel J},
    booktitle={2022 44th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
    pages={2215--2218},
    year={2022},
    organization={IEEE}
  }
  ```

and for using compressive function approaches:

  > Flotho, P., Bhamborae, M. J., Haab, L., & Strauss, D. J. (2018). Lagrangian motion magnification revisited: Continuous, magnitude driven motion scaling for psychophysiological experiments. In 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 2215-2218). IEEE.

  ```bibtex
  @inproceedings{flotho2018lagrangian,
    title={Lagrangian motion magnification revisited: Continuous, magnitude driven motion scaling for psychophysiological experiments},
    author={Flotho, Philipp and Bhamborae, Mayur J and Haab, Lars and Strauss, Daniel J},
    booktitle={2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
    pages={3586--3589},
    year={2018},
    organization={IEEE}
  }
  ```


**3. Dense and sparse thermal landmarks:**

  > Flotho, P., Piening, M., Kukleva, A., & Steidl, G. (2025). T-FAKE: Synthesizing Thermal Images for Facial Landmarking. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 26356–26366).

  ```bibtex
  @inproceedings{flotho2025t,
    title={T-FAKE: Synthesizing Thermal Images for Facial Landmarking},
    author={Flotho, Philipp and Piening, Moritz and Kukleva, Anna and Steidl, Gabriele},
    booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
    pages={26356--26366},
    year={2025}
  }
  ```

## Third-Party Code

This distribution bundles selected research code from external projects. Please cite the original authors when using functionality derived from these components.

**RAFT optical flow**:

  > Teed, Z., & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In *European Conference on Computer Vision* (pp. 402–419). Springer.

  ```bibtex
  @inproceedings{teed2020raft,
    title={Raft: Recurrent all-pairs field transforms for optical flow},
    author={Teed, Zachary and Deng, Jia},
    booktitle={European conference on computer vision},
    pages={402--419},
    year={2020},
    organization={Springer}
  }
  ```

**FlowMag motion magnification**:

  > Pan, Z., Geng, D., & Owens, A. (2023). Self-supervised motion magnification by backpropagating through optical flow. *Advances in Neural Information Processing Systems*, 36, 253–273.

  ```bibtex
  @article{pan2023self,
    title={Self-supervised motion magnification by backpropagating through optical flow},
    author={Pan, Zhaoying and Geng, Daniel and Owens, Andrew},
    journal={Advances in Neural Information Processing Systems},
    volume={36},
    pages={253--273},
    year={2023}
  }
  ```

The FlowMag implementation is provided in `neurovc.contrib.flowmag` (MIT; see `src/neurovc/contrib/flowmag/LICENSE`). Supporting helpers live in `neurovc.contrib.flowmag_util`.

**Thermal face detection (TFW)**:

  > Kuzdeuov, A., Aubakirova, D., Koishigarina, D., & Varol, H. A. (2022). TFW: Annotated Thermal Faces in the Wild Dataset. *IEEE Transactions on Information Forensics and Security*, 17, 2084–2094. https://doi.org/10.1109/TIFS.2022.3177949

  ```bibtex
  @article{9781417,
    author={Kuzdeuov, Askat and Aubakirova, Dana and Koishigarina, Darina and Varol, Huseyin Atakan},
    journal={IEEE Transactions on Information Forensics and Security},
    title={TFW: Annotated Thermal Faces in the Wild Dataset},
    year={2022},
    volume={17},
    pages={2084-2094},
    doi={10.1109/TIFS.2022.3177949}
  }
  ```

### Licensing Notice:
This project contains code derived from RAFT, which is licensed under the BSD 3-Clause License. See `neurovc/raft/LICENSE` for details.
The FlowMag contrib module (`neurovc.contrib.flowmag`) is distributed under the MIT License; see `src/neurovc/contrib/flowmag/LICENSE`.
The rest of this project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 1.0 license (CC BY-NC-SA 1.0). See `LICENSE` for details.

When using or redistributing this project, you must comply with each respective license.
