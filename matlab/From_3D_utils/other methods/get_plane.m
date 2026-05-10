function [X, Y, Z, P] = get_plane(pnts_3D)
    x=1;y=2;z=3;
    Xs = pnts_3D(:, x); Ys = pnts_3D(:, y); Zs = pnts_3D(:, z);
    P = get_plane_params(pnts_3D);
    d = P(4);
    normal = P(1:3);
    [X,Y] = meshgrid(linspace(min(Xs),max(Xs)),linspace(min(Ys),max(Ys))); % Create a grid of x and y values
    Z = (-normal(1)*X - normal(2)*Y - d)/normal(3); % Solve for z values on the plane
    Z(Z > max(Zs) | Z < min(Zs)) = nan; % ignore all Zs above and below points   
end