# RevMolSpec
Reversible spectrum prediction/structure elucidation

The following programs use an iRevNet inspired by https://github.com/jhjacobsen/pytorch-i-revnet to write a program which is trained to predict spectra, but can also do structure elucidation (within constraints).

##RevMolSpec01.py

This projects a 4x16x16 (=1024) structure prediction onto a 1024-bit binned spectrum. It works reversible for outputs coming from a trained input. It does not work correctly for "random" inputs. The reason is that the entropy of the input is higher than the entropy of the output. Those need to match in order to work for any outputs as inputs.







