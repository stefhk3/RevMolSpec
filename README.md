# RevMolSpec
Reversible spectrum prediction/structure elucidation

The following programs use an iRevNet inspired by https://github.com/jhjacobsen/pytorch-i-revnet to write a program which is trained to predict spectra, but can also do structure elucidation (within constraints).

##RevMolSpec01.py

This projects a 4x16x16 (=1024) structure prediction onto a 1024-bit binned spectrum. It works reversible for outputs coming from a trained input. It does not work correctly for "random" inputs. The reason is that the entropy of the input is higher than the entropy of the output. Those need to match in order to work for any outputs as inputs.

##RevMolSpec02.py

This reduces the actual information in the spectrum to 128 bits (roughly the same as entropy in the structure input). The network is fully reversible and stable. Should in principle work, tested a few cases.

##RevMolSpec03.py

This learns a prior an Z_free and has X-magnitude regularization. This means the network works an a half-way sensible fashion even for Y inputs it has not seen during learning.

##RevMolSpec04.py

This adds aggregate statistics and a decoder for reconstructed inputs. Largely works as expected.
