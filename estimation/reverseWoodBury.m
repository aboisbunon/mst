function [res]=reverseWoodBury(A,invA)
%
%function [res]=reverseWoodBury(A,invA)
%
%   calculate the inverse of the Matrix A(1:n-1,1:n-1) knowing A and A^(-1)
%   using The Woodbury Matrix Identity
%
%   A and invA are matrices of size n*n
%   The resulting matrix res is of size (n-1)*(n-1)
%
%   By J. Delporte and A. Boisbunon
%   01/2012

    [n,p]=size(A);
    U=[zeros(n-1,1) A(1:n-1,n);
            1       (A(n,n)-1)/2 ];
    C=[0 1;1 0];
    res=invA+(invA*U)*inv(C-U'*invA*U)*U'*invA;
    res=res(1:n-1,1:n-1);


