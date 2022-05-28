function a = mat2gray(a)

m = min(a(:));
a = a - m;

M = max(a(:));
if ( M ~= 0 )
  a = a / M;
end
