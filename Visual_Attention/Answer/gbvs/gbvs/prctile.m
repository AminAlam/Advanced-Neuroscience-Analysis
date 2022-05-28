function v = prctile( a , p )

a = sort( a(:) );
l = round( p / 100 * numel(a) );
v = a(l);
