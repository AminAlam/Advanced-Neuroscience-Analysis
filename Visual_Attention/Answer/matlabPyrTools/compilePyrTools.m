% This is a script file for compiling the mex versions of the Steerable
% Pyramid Tools.
% 
% Usage:>> compilePyrTools
%
% Tested for gcc and lcc.
%
% Rob Young, 9/08
clc
mex -setup C++
mex -setup C++ upConv.c convolve.c wrap.c edges.c 
mex -setup C++ corrDn.c convolve.c wrap.c edges.c
mex -setup C++ histo.c
%mex innerProd.c
mex pointOp.c
mex range2.c
