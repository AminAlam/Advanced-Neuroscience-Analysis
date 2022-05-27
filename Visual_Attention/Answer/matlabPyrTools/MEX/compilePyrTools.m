% This is a script file for compiling the mex versions of the Steerable
% Pyramid Tools.
% 
% Usage:>> compilePyrTools
%
% Tested for gcc and lcc.
%
% Rob Young, 9/08
clc
mex -setup CPP upConv.c convolve.c wrap.c edges.c
mex -setup CPP corrDn.c convolve.c wrap.c edges.c
mex -setup CPP histo.c
%mex innerProd.c
mex -setup CPP pointOp.c
mex -setup CPP range2.c
