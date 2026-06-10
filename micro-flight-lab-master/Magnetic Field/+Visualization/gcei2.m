function cel=gcei2(kc, p, a, b)
    cel=zeros(size(kc));
    for i=1:size(cel,1)
        for j=1:size(cel,2)
            if kc(i,j)==0
                exit
            end
            ca = 1e-6;
            e = abs(kc(i,j));
            em = 1;
            if p(i,j)>0
                pp = sqrt(p(i,j));
                bb = b(i,j)/pp;
            else
                exit
            end
            f = a;
            a = a + bb/pp;
            g = e/pp;
            bb = 2*(bb + f*g);
            pp = g + pp;
            g = em;
            em = kc(i,j) + em;
            k=abs(kc(i,j));
            while (abs(g-k)>g*ca)
                k = 2*sqrt(e);
                e = k*em;
                f = a;
                a = a + bb/pp;
                g = e/pp;
                bb = 2*(bb + f*g);
                pp = g + pp;
                g = em;
                em = k + em;
            end
            cel(i,j) = (pi/2.)*(bb + a*em)/( em*(em + pp) );
        end
    end
end
