function A = getPolynomialBasis(d,T,nt,basis)

if nargin==0
    T = 2.0;
    nt = 101;
    d = 3;
    tt =linspace(0,T,nt);
    A = feval(mfilename,d,T,nt,'monomial');
    figure(2); clf;
    for k=1:size(A,2)
        plot(tt,A(:,k));
        hold on
    end
end

if not(exist('basis','var')) || isempty(basis)
    basis = 'monomial';
end

tt = linspace(0,T,nt);

switch basis
    case 'monomial'
        A = tt(:).^(0:d);
    case 'qr'
        B = tt(:).^(0:d);
        [A,R] = qr(B,'econ');
    case 'Legendre'
        tb = linspace(-1,1,nt);
        A = zeros(nt,d+1);
        for k=0:d
            A(:,k+1) = reshape(legendreP(k,tb),[],1);
        end
    otherwise
        error('invalid basis')
end