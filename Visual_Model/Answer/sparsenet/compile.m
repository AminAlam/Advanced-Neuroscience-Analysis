%%
%compile mex files
mex -setup C
mex -v cgf.c nrf/brent.c nrf/frprmn.c nrf/linmin.c nrf/mnbrak.c nrf/nrutil.c -Inrf

