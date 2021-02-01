%% ============EBOwithCMAR============
% This code is taken from UMOEA-II
% =========================================================================

function x = han_boun (x, xmax, xmin, x2, PopSize,hb)

switch hb;
    case 1 % for DE
        x_L = repmat(xmin, PopSize, 1);
        pos = x < x_L;
        x(pos) = (x2(pos) + x_L(pos)) / 2;
        
        x_U = repmat(xmax, PopSize, 1);
        pos = x > x_U;
        x(pos) = (x2(pos) + x_U(pos)) / 2;
        
    case 2 % for CMA-ES
        x_L = repmat(xmin, PopSize, 1);
        pos = x < x_L;
        x_U = repmat(xmax, PopSize, 1);
        x(pos) = min(x_U(pos),max(x_L(pos),2*x_L(pos)-x2(pos)))  ;
        pos = x > x_U;
        x(pos) = max(x_L(pos),min(x_U(pos),2*x_L(pos)-x2(pos)));
        
end  
