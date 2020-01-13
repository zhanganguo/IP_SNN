function nn=nnlifsim_Zhang_ip(nn, test_x, test_y, opts, ip_opts)
% From paper:
% Wenrui Zhang, Peng Li. Information-Theoretic Intrinsic Plasticity for 
% Online Unsupervised Learning in Spiking Neural Networks. Frontiers in 
% Neuroscience, Vol. 13, No. Feb, pp.1-14, 2019.
dt = opts.dt;
nn.performance = [];
num_examples = size(test_x,1);
mu = 0.5;

eta = ip_opts.eta;
initial_R = ip_opts.initial_R;
initial_tau_m = ip_opts.initial_tau_m;

% Initialize network architecture
for l = 1 : numel(nn.size)
    blank_neurons = zeros(num_examples, nn.size(l));
    one_neurons = ones(num_examples, nn.size(l));
    nn.layers{l}.mem = blank_neurons;
    nn.layers{l}.refrac_end = blank_neurons;
    nn.layers{l}.sum_spikes = blank_neurons;

    nn.layers{l}.R = one_neurons * initial_R;
    nn.layers{l}.tau_m = one_neurons * initial_tau_m;

end

% Precache answers
[~,   ans_idx] = max(test_y');

for t=dt:dt:opts.duration
    % Create poisson distributed spikes from the input images
    %   (for all images in parallel)
    rescale_fac = 1/(dt*opts.max_rate);
    spike_snapshot = rand(size(test_x)) * rescale_fac;
    inp_image = spike_snapshot <= test_x;
    
    nn.layers{1}.spikes = inp_image;
    nn.layers{1}.sum_spikes = nn.layers{1}.sum_spikes + inp_image;
    for l = 2 : numel(nn.size)
        % Get input impulse from incoming spikes
        I = nn.layers{l-1}.spikes * nn.W{l-1}';
        
        dv = nn.layers{l}.R .* I ./ nn.layers{l}.tau_m;
        
        % Add input to membrane p otential
        nn.layers{l}.mem = nn.layers{l}.mem + dv;
        % Check for spiking
        nn.layers{l}.spikes = nn.layers{l}.mem >= opts.threshold;
        % Reset
        nn.layers{l}.mem(nn.layers{l}.spikes) = 0;
        % Ban updates until....
        nn.layers{l}.refrac_end(nn.layers{l}.spikes) = t + opts.t_ref;
        % Store result for analysis later
        nn.layers{l}.sum_spikes = nn.layers{l}.sum_spikes + nn.layers{l}.spikes;
        
        % IP update rule:
        y = 1 ./ (opts.t_ref + nn.layers{l}.tau_m .* log(nn.layers{l}.R.*I./(nn.layers{l}.R.*I-opts.threshold)));
        W = opts.threshold ./ (exp(1./nn.layers{l}.tau_m.*(1./y-opts.t_ref)) -1);
        delta_R = zeros(size(y));
        delta_tau_m = zeros(size(y));
        t_y = y > 1;
        t_y_inv = y <= 1;
        delta_R(t_y) = (2*y(t_y).*nn.layers{l}.tau_m(t_y).*opts.threshold - W(t_y) - opts.threshold - 1/mu * opts.threshold*nn.layers{l}.tau_m(t_y).*y(t_y).^2)./(nn.layers{l}.R(t_y) .* W(t_y));
        delta_R(t_y_inv) = 0.1;
        delta_tau_m(t_y) = (2*opts.t_ref - 1 - 1/mu*(opts.t_ref.*y(t_y).^2 - y(t_y))) ./ nn.layers{l}.tau_m(t_y);
        delta_tau_m(t_y_inv) = 0.1;
        nn.layers{l}.R = nn.layers{l}.R + eta * delta_R;
        nn.layers{l}.tau_m = nn.layers{l}.tau_m + eta * delta_tau_m;
    end
    
    if(mod(round(t/dt),round(opts.report_every/dt)) == round(opts.report_every/dt)-1)
        [~, guess_idx] = max(nn.layers{end}.sum_spikes');
        acc = sum(guess_idx==ans_idx)/size(test_y,1)*100;
        fprintf('Time: %1.3fs | Accuracy: %2.2f%%.\n', t, acc);
        nn.performance(end+1) = acc;
    else
        fprintf('.');
    end
end


% Get answer
[~, guess_idx] = max(nn.layers{end}.sum_spikes');
acc = sum(guess_idx==ans_idx)/size(test_y,1)*100;
fprintf('\nFinal spiking accuracy: %2.2f%%\n', acc);

end


