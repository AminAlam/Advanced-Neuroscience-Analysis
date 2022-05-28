
function imgr = imresize( img , sz , type )

if ( strcmp(class(img),'uint8') == 1 ) img = double(img)/255; end

Worig = size(img,2);
Horig = size(img,1);
z = size(img,3);

h = sz(1);
w = sz(2);

x = repmat( [ 1 : Worig ] , [ Horig 1 ] );
y = repmat( [ 1 : Horig ]', [ 1 Worig ] );

xi = repmat( linspace( 1 , Worig , w ) , [ h 1 ] );
yi = repmat( linspace( 1 , Horig , h )' , [ 1 w ] );
imgr = repmat( img(1)  , [ h w z ] );

% blur image a bit before resampling
% here blur is gaussian with std = sqrt(2)
k = [ 0.0301    0.1050    0.2223    0.2854    0.2223    0.1050    0.0301 ];
k = k / sum(k);
for i = 1 : z
  img(:,:,i) = myconv2( myconv2( img(:,:,i) , k ) , k' );
  imgr(:,:,i) = interp2( x , y , img(:,:,i) , xi , yi );
end

