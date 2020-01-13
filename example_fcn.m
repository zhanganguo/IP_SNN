dbstop if error
%% Train an example FC network to achieve very high classification, fast.
%    Load paths
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
%% Load data
rand('state', 0);

dataset = 'fashion_mnist';

if strcmp(dataset, 'mnist') == 1
    load('mnist_uint8.mat');
    train_x = double(train_x)/255;
    train_y = double(train_y);
    test_x = double(test_x)/255;
    test_y = double(test_y);
    load('fnn_2hidden_mnist.mat');
elseif strcmp(dataset, 'fashion_mnist') == 1
    load('fashion_mnist_double.mat');
    load('nn_fashion_mnist_90.75.mat');
else
end

% load('D:\OneDrive\Achievement\Paper\1.COMPLETED\Journal\2019 - Neurocomputing - Fast and Robust Learning in SFNNs based on IP Mechanism\Simulation\dlt_cnn_map_dropout_nobiasnn\data\cifar-10-matlab\data_batch_1.mat');
% imData = uint8(zeros(10000, 32, 32, 3));
% grayData = uint8(zeros(10000, 32, 32));
% dataColor = uint8(zeros(32, 32, 3));
% hwait=waitbar(0,'计算中...');
% for i = 1:10000
%     value = 100 * i / 10000;
%     waitbar(i/10000, hwait, sprintf('计算中:%3.2f%%',value));
%     data1 = data(i,:);
%     data1 = data1';
%     dataColor = reshape(data1, [32, 32, 3]);
%     imData(i,:,:,:) = dataColor;
%     grayData(i,:,:) = rgb2gray(dataColor);
% end
% train_x = double(grayData) / 255;

%% Noise 
noise_type = 'gaussian';

if strcmp(noise_type, 'none') == 1
    train_noise = zeros(size(train_x));
    test_noise = zeros(size(test_x));
elseif strcmp(noise_type, 'gaussian') == 1  
%     a = 0;
%     b = 0.33465;
    a = 0;
    b = 0.33465;
    train_noise = a + b * randn(size(train_x));
    test_noise = a + b * randn(size(test_x));
elseif strcmp(noise_type, 'rayleigh') == 1
    a = 0;
    b = 0.1;
    train_noise = a + b * (-log(1-rand(size(train_x)))).^0.5 .* sign(rand(size(train_x))-0.5);
    test_noise = a + b * (-log(1-rand(size(test_x)))).^0.5 .* sign(rand(size(test_x))-0.5);
elseif strcmp(noise_type, 'uniform') == 1
    a = 0;
    b = 0.17;
    train_noise =  a + (b - a) * (rand(size(train_x))-0.5) * 2;
    test_noise =  a + (b - a) * (rand(size(test_x))-0.5) * 2;
elseif strcmp(noise_type, 'gamma') == 1
    a = 18.78;
    b = 4;
    train_noise = zeros(size(train_x));
    test_noise = zeros(size(test_x));
    for i = 1 : b
        train_noise = train_noise + (-1/a) .* log(1 - rand(size(train_noise)));
        test_noise = test_noise + (-1/a) .* log(1 - rand(size(test_noise)));
    end
elseif strcmp(noise_type, 'pepper') == 1
    a = 0.06;
    b = 0.06;
    x = rand(size(train_x));
    train_noise = zeros(size(train_x));
    train_noise(x<=a) = -1;
    train_noise(x>a & x<(a+b)) = 1;
    t_x = rand(size(test_x));
    test_noise = zeros(size(test_x));
    test_noise(t_x<=a) = -1;
    test_noise(t_x>a & t_x<(a+b)) = 1;
end

noisy_train_x = train_x + train_noise;
noisy_train_x(noisy_train_x > 1) = 1;
noisy_train_x(noisy_train_x < 0) = 0;
noisy_test_x = test_x + test_noise;
test_x(test_x > 1) = 1;
test_x(test_x < 0) = 0;

s = snr(train_x, noisy_train_x)

%% Training an FNN
% nn = nnsetup([784 1200 1200 10]);
% % Rescale weights for ReLU
% for i = 2 : nn.n   
%     % Weights - choose between [-0.1 0.1]
%     nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
%     nn.vW{i - 1} = zeros(size(nn.W{i-1}));
% end
% % Set up learning constants
% nn.activation_function = 'relu';
% nn.output ='relu';
% nn.learningRate = 1;
% nn.momentum = 0.5;
% nn.dropoutFraction = 0.5;
% nn.learn_bias = 0;
% opts.numepochs =  200;
% opts.batchsize = 100;
% % Train - takes about 15 seconds per epoch on my machine
% nn = nntrain(nn, noisy_train_x, train_y, opts);
% % Test - should be 98.62% after 15 epochs
% [er, train_bad] = nntest(nn, noisy_train_x, train_y);
% fprintf('TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);

%% Test Accuracy
[er, bad] = nntest(fnn, noisy_test_x, test_y);
fprintf('Test Accuracy of FNN: %2.2f%%.\n', (1-er)*100);

%% Spike-based Testing of Fully-Connected NN
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =  2;
t_opts.dt           = 0.001;
t_opts.duration     = 0.300;
t_opts.report_every = 0.001;
t_opts.max_rate     =   100;
t_opts.reset        = 0;

nn_nonip = nnlifsim(fnn, noisy_test_x, test_y, t_opts);

%% IP rule of Chunguang Li.
% From paper:
% C. Li, Y. Li. A spike-based model of neuronal intrinsic plasticity. 
% IEEE Transactions on Autonomous Mental Development, Vol. 5, No. 1, pp.
% 62-73, 2013.
% Parameter setting from:
% A. Zhang, H. Zhou, X. Li, and W. Zhu, “Fast and robust learning in Spiking 
% Feed-forward Neural Networks based on Intrinsic Plasticity mechanism,” 
% Neurocomputing, vol. 365, pp. 102–112, 2019.
li_ip_opts.tau_ip       = 20;
li_ip_opts.beta         = 0.6;
li_ip_opts.eta          = 0.5;
li_ip_opts.initial_rC   = 2;
li_ip_opts.initial_rR   = 2;
nn_li_ip = nnlifsim_Li_ip(fnn, noisy_test_x, test_y, t_opts, li_ip_opts);

%% IP rule of Wenrui Zhang
% From paper:
% Wenrui Zhang, Peng Li. Information-Theoretic Intrinsic Plasticity for 
% Online Unsupervised Learning in Spiking Neural Networks. Frontiers in 
% Neuroscience, Vol. 13, No. Feb, pp.1-14, 2019.
zhang_ip_opts.eta           = 0.1;
zhang_ip_opts.initial_R     = 1;
zhang_ip_opts.initial_tau_m = 1;

nn_zhang_ip = nnlifsim_Zhang_ip(fnn, noisy_test_x, test_y, t_opts, zhang_ip_opts);

%%

t_opts.R            = 1;
t_opts.tau_m        = 1;
t_opts.k            = 100;


ip_opts.eta_alpha = 0.5;
ip_opts.eta_beta = 0.00;
ip_opts.mu = 0.5;
ip_opts.initial_alpha = 1.05;
ip_opts.initial_beta = 1;
nn_ip = nnlifsim_ip(fnn, noisy_test_x, test_y, t_opts, ip_opts);
fprintf('Done.\n');

%% Show the difference
figure;
plot((t_opts.dt:t_opts.dt:t_opts.duration)*1000, nn_nonip.performance,'r');
hold on;
plot((t_opts.dt:t_opts.dt:t_opts.duration)*1000, nn_zhang_ip.performance,'black');
hold on;
plot((t_opts.dt:t_opts.dt:t_opts.duration)*1000, nn_li_ip.performance,'b');
% hold on;
% plot((t_opts.dt:t_opts.dt:t_opts.duration)*100, nn_our_ip.performance,'m');
grid;
legend('SFNN-noIP', 'SFNN-ZHANG-IP', 'SFNN-LI-IP');
ylim([0 100]);
xlabel('Time [s]');
ylabel('Accuracy [%]');