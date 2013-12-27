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

%% Record Data For Testing
Data_Except_Links = rmfield(Data,'links');
struct2csv(Data_Except_Links,'test_data/Data_Except_Links.csv');

m = length(Data.links);
for i = 1:m
    name = sprintf('test_data/vector%i.csv',i);
    struct2csv(Data.links,name);
end

fileID = fopen('test_data/Prob_Model_Given_Data.csv','w');
fprintf(fileID, 'Probability Given Data, %d\n', Prob_Model_Given_Data);