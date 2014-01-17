%% Define Network.
T = 10000;  % Total time interval for data sequence
lambda1=1;  % Rate of non-infected message passage. 
k=0.1;  % Rate of infected message passage.
nodes=4; % Number of nodes.
Network = zeros(nodes); %Matrix of rates between the various nodes.
Network(1,2)=1/10000; % Rate at wich Node 1 sends infected messages to Node 2
Network(1,3)=1/10000; % Rate at wich Node 1 sends infected messages to Node 3
Network(2,3)=1/10; % Rate at wich Node 2 sends infected messages to Node 3
Network(2,4)=1/10; % Rate at wich Node 2 sends infected messages to Node 4
Network(3,4)=1/10; % Rate at wich Node 3 sends infected messages to Node 4
%% Generate data.
Data = Generate_Data(T,lambda1,Network);  % Call function to generate simulated data for this network.
% The output of the function "Generate_Data" is a structure called 'Data'.
% The four elements of this structure are:
% Data.time_n = times at which each node became infected.
% Data.num = number of messages sent along each link before t = [1,2,...,T]
% Data.t = number of time points.
% Data.links(i).vec = actual times of messages sent along the ith link. 

%% Specify Time.
time_array = [0,10000*rand(1,3)];  %Initial guess for the times at which each node became infected.
% time_array = Data.time_n; %% Give the correct value of the infections times as the initial guess.

%% Compute Probability of that Timing
Prob_Model_Given_Data = Get_Prob_Model_Given_Data(time_array,Network,Data,lambda1,T);  
% Call function to compute the probability of the infections times in
% 'time_array' given the network model and the simulated data sequence.

%% What follows is a preliminary MCMC routine to integrate the probability of the model given the data.
% The basic idea is to guess a new vector of time of infection. Compute the
% probability of the model given the actual data and that proposed time of
% infection.  If the new guess is more probable than the last, we accept.
% If it is less probable, then we accept with some probability. Right now
% the proposal distrivution is normally distributed, and impossible time
% series are given a probability of log(P) = -inf.  
clf
time_array_0 = time_array; 
N_MCMC_Steps = 10000;
time_array_history = zeros(N_MCMC_Steps,length(time_array_0));
fcn_array_history = zeros(N_MCMC_Steps,1);
Prob_Model_Given_Data_0 = Prob_Model_Given_Data;
for i=1:N_MCMC_Steps
    time_array_1 = time_array_0+100.0*randn(size(time_array_0));
    Prob_Model_Given_Data_1 = Get_Prob_Model_Given_Data(time_array_1,Network,Data,lambda1,T);
    if (Prob_Model_Given_Data_1>Prob_Model_Given_Data_0)&&min(time_array_1)>=0
        time_array_0 = time_array_1;
        Prob_Model_Given_Data_0 = Prob_Model_Given_Data_1;
    elseif ((Prob_Model_Given_Data_1-Prob_Model_Given_Data_0)>log(rand))&&min(time_array_1)>=0
        time_array_0 = time_array_1;
        Prob_Model_Given_Data_0 = Prob_Model_Given_Data_1;
    else
        time_array_0 = time_array_0;
        Prob_Model_Given_Data_0 = Prob_Model_Given_Data_0;
    end
    time_array_history(i,:) = time_array_1;
    fcn_array_history(i,1) = Prob_Model_Given_Data_0;
end

% Some plots
scatter(time_array_history(:,2),time_array_history(:,3),[],fcn_array_history,'filled');  
% Plot the trajectory of the MCMC in the time at which nodes 2 and 3 get
% infected.
hold on
plot(Data.time_n(2),Data.time_n(3),'rs','markersize',16,'markerfacecolor','w');
colorbar
% Plot the actual time of infection of nodes 2 and 3.
