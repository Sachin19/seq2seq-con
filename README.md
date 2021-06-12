This repository contains the code for the following two papers

* [Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs](https://arxiv.org/pdf/1812.04616.pdf) _Sachin Kumar_ and _Yulia Tsvetkov_
* [Machine Translation in Low-resource Language Varieties](#) _Sachin Kumar_, _Antonios Anastasopoulos_, _Shuly Wintner_ and _Yulia Tsvetkov_

If you use this repository for academic research, [please cite the relevant papers](#Publications)

# Dependencies

* __Pytorch >= 1.6__
* Python >= 3.6

# Instructions to reproduce results from the papers

Please look at the [examples](examples) directory for detailed instructions.

## Publications

If you use this code, please cite the following paper

```
@inproceedings{kumar2018vmf,
title={Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs},
author={Sachin Kumar and Yulia Tsvetkov},
booktitle={Proc. of ICLR},
year={2019},
url={https://arxiv.org/pdf/1812.04616.pdf},
}

@inproceedings{kumar2021langvar,
title={Machine Translation into Low-resource Language Varieties},
author={Sachin Kumar, Antonios Anastasopoulos, Shuly Wintner and Yulia Tsvetkov},
booktitle={Proc. of ACL},
year={2021},
url={},
}
```

## Acknowledgements

This code base has been adapted from [open-NMT](https://github.com/OpenNMT/OpenNMT-py) toolkit.

scripts/compare_mt.py has been taken from [here](https://github.com/neulab/compare-mt)

## License 

This code is freely available for non-commercial use, and may be redistributed under these conditions. Please, see the [license](https://github.com/Sachin19/seq2seq-con/blob/master/LICENSE) for further details. Interested in a commercial license? Please contact the authors

