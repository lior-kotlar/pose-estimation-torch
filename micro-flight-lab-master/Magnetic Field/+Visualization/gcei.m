function cel=gcei(kc, p, c, s)
    cel=zeros(size(kc));
    for i=1:size(cel,1)
        for j=1:size(cel,2)
            if kc(i,j)==0
                exit
            end
            errtol = 1e-6;
            k = abs(kc(i,j));
            pp = p(i,j);
            cc = c;
            ss = s(i,j);
            em = 1;
            if p(i,j)>0
                pp = sqrt(p(i,j));
                ss = s(i,j)/pp;
            else
                f = kc(i,j)*kc(i,j);
                q = 1. - f;
                g = 1. - pp;
                f = f - pp;
                q = q*(ss - c*pp);
                pp = sqrt( f/g );
                cc = (c - ss)/g;
                ss = - q/(g*g*pp) + cc*pp;
            end
            f = cc;
            cc = cc + ss/pp;
            g = k/pp;
            ss = 2*(ss + f*g);
            pp = g + pp;
            g = em;
            em = k + em;
            kk = k;
            while (abs(g-k)>g*errtol)
                k = 2*sqrt(kk);
                kk = k*em;
                f = cc;
                cc = cc + ss/pp;
                g = kk/pp;
                ss = 2*(ss + f*g);
                pp = g + pp;
                g = em;
                em = k + em;
            end
            cel(i,j) = (pi/2.)*(ss + cc*em)/( em*(em + pp) );
        end
    end
end
