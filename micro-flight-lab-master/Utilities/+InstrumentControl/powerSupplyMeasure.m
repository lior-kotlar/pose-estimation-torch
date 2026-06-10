%% Instrument Connection
% Find a VISA-USB object.


% Create the VISA-USB object if it does not exist
% otherwise use the object that was found.
if isempty(PowerSup)
    PowerSup = visa('NI', 'USB0::0x0AAD::0x0135::033613137::0::INSTR');
else
    fclose(PowerSup);
    PowerSup1 = PowerSup(1);
    PowerSup2 = PowerSup(2);
end

% Connect to instrument object, obj1.
fopen(PowerSup2);
%%
clear('time','curr','voltage')

c=0;
record_time=10;
tic
fprintf(PowerSup2, 'OUTP ON');
while toc<record_time
    c=c+1;
    time(c)=toc;
    curr(c) = str2double(query(PowerSup2, 'MEAS:CURR?'));
    voltage(c) = str2double(query(PowerSup2, 'MEAS:VOLT?'));
end
fprintf(PowerSup2, 'OUTP OFF');
hold on
plot(time,curr,'.')
plot(time,voltage,'.')
%% Disconnect and Clean Up
% Disconnect from instrument object, obj1.
fclose(PowerSup2);