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

%% Randomly add strong reflectors to the signal 
fs = 10e6; % signal sample rate
fc = 1e6; % assuming the source is a 1MHz source, so we'd expect a strong reflection at this frequency from some random object
tau = 0.1;
t = 0:1/fs:(len-1)*(1/fs); % time vector equal in length to total signal

% create random posiition in time for interferer/reflector
t0 = round(len*rand(1,b));

% create a strong reflector
y = sin(2*pi*fc.*t) .* exp(-((t - t0)./tau).^2);


% repeat the matrix a few times 


%% Create noisy realizations of all data



