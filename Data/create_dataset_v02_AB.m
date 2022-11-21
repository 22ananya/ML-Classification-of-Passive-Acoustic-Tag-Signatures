%% Creating Dataset - Loading Simulated Waveforms, Preprocessing and Creating noisy and cluttered realizations

close all
clear all

%% load data files - simulated waveforms
x_sim10 = load('sim_signatures_10C.mat');

X = x_sim10.y_10C;
Y1 = [1:8];

x_sim20 = load('sim_signatures_20C.mat');
X = cat(2, X, x_sim20.y_20C);
Y = cat(2, Y1, Y1);

x_sim30 = load('sim_signatures_30C.mat');
X = cat(2, X, x_sim30.y_30C);
Y = cat(2, Y, Y1);

%% Crop into useful part of signal
start = 3001;
stop = 10000; 
X = X(start:stop, :);
[a, b] = size(X);

%% Create replica waveforms with signals randomly distributed along time
len = 10000;
X2 = zeros(len, b);

% create random starting point for signals
r = round(3000*rand(1,b));
for i = 1:b
    X2(r(i):a+r(i)-1,i) = X(:,i);
end

% normalize signals to a max amplitude of 1
X2 = X2./max(abs(X2));

X = X2;

% Plot to verify
figure;
plot(X);
title('Check that max amplitude is 1')

%% Randomly add strong reflectors and Noise to the signal 
% reflector parameters
fs = 10e6; % signal sample rate
fc = 1e6; % assuming the source is a 1MHz source, so we'd expect a strong reflection at this frequency from some random object
tau = 1e-6;
t = 0:1/fs:(len-1)*(1/fs); % time vector equal in length to total signal
RL = [4 2 0];


% create noise arrays at -6dB -3dB, 0dB, 3dB and 6dB -  Signal to Noise Amplitude ratio
NL = [2, sqrt(2), 1, (1/sqrt(2)), 1/2]; % noise amplitude vector

% To create a duplicate dataset with much better SNR if needed
NL = NL./2;
    
X = repmat(X,1,50);
Y = repmat(Y,1,50); 
len1 = size(X,2);

% Reflector
% create random posiition in time for interferer/reflector
t0 = round(len*rand(1,len1))*(1/fs);

% create a strong interferer 
inf = zeros(size(X));
for i = 1:len1
    if mod(i,3) == 0
        inf(:,i) = RL(1)*sin(2*pi*fc.*t).* exp(-((t - t0(i))./tau).^2);
    elseif mod(i,3) == 1
        inf(:,i) = RL(2)*sin(2*pi*fc.*t).* exp(-((t - t0(i))./tau).^2);
    else
        inf(:,i) = RL(3)*sin(2*pi*fc.*t).* exp(-((t - t0(i))./tau).^2);
    end
end

% shuffle order of interferer vector
ind = size(inf,2);
inf2 = inf(:,randperm(ind));

% Noise
agn = -1 + 2*rand(size(X)); % Added gaussian noise - zero mean
len = size(X,2);

for i = 1:length(NL)
    X(:,(i-1)*(len/5) + 1 : i*(len/5)) = X(:,(i-1)*(len/5) + 1 : i*(len/5)) + NL(i)*agn(:,(i-1)*(len/5) + 1 : i*(len/5));
end

X = X + inf2;

%% normalize signals to a max amplitude of 1 
X = X./max(abs(X)); 

%% Save data for importing
save('Dataset_v02.mat', 'X', 'Y')
