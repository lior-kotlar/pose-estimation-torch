function phi = get_phi(head_tail_vec, wing_vec)
    % returns phi (the angle in the xy axis between the wing vector and the fly x axis) 
    x=1;y=2;z=3;
    head_tail_vec(z) = 0; wing_vec(z) = 0;
    head_tail_vec = head_tail_vec/norm(head_tail_vec);
    wing_vec = wing_vec/norm(wing_vec);
    phi = acos(dot(head_tail_vec, wing_vec)); 
    phi = rad2deg(phi);
end