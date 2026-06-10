I_bpp=10^-13*[2.9,0,-0.3;0,6.3,0;-0.3,0,5.7];
I_fly=10^-13*diag([1.1,5,5]);
I_scalar=5e-13;

mu0=(4*pi)*10^-7;
Br=0.15;
mag_dens=7e3;

% mag_vol=(0.25e-3)^3;
mag_vol=pi*(0.1e-3)^2*0.5e-3;

mag_mom=mag_vol*Br/mu0;

B_field=1e-2;


Fs = 100000;            % Sampling frequency                    
dT = 1/Fs;             % Sampling period       
signal_Length = 10000;             % Length of signal
t = (0:signal_Length-1)*dT;        % Time vector
tot_time=5e-2;

field_time=zeros(size(t));
field_time(t<5e-3)=B_field;


freq=sqrt(mag_mom*B_field/I_scalar)/(2*pi);

initial_ang=pi/2;
initial_angmom=0;
ang(1)=initial_ang;
L(1)=initial_angmom;
omega(1)=initial_angmom/I_scalar;
ang_drag=1e-15;


for time_step=1:signal_Length
    L(time_step+1)=L(time_step)-mag_mom*field_time(time_step)*sin(ang(time_step))*dT-ang_drag*omega(time_step);
    omega(time_step+1)=L(time_step+1)/I_scalar;
    ang(time_step+1)=ang(time_step)+omega(time_step+1)*dT;
end

% plot((1:time_step+1)*dT,ang)
plot(1000*t,rad2deg(ang(1:length(t))))
title('Signal Corrupted with Zero-Mean Random Noise')
xlabel('t (milliseconds)')
ylabel('X(t)')

Y=fft(ang);
P2 = abs(Y/signal_Length);
P1 = P2(1:signal_Length/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(signal_Length/2))/signal_Length;
figure
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

figure
plot(1000*t,omega(1:length(t))/(2*pi)*360)

figure
plot(1000*t,diff(omega)/(2*pi)*360/dT)

rms(diff(rad2deg(omega))/dT)