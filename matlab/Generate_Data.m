function[Data]=Generate_Data(T,lambda1,Network)
%this function takes inputs:
% T - Final time.
% lambda1 - non-infective message rate.
% Network(i,j)  - matrix of infective message passage rate from node i to j.
% With these inputs the function will simulate the infective and
% non-infected messages sent along each link.  The results of this
% simulation are saved in the output structure "Data" with elements:
% Data.time_n = times at which each node became infected.
% Data.num = number of messages sent along each link before t = [1,2,...,T]
% Data.t = number of time points.
% Data.links(i).vec = actual times of messages sent along the ith link. 

%%
nodes = size(Network,1); % The number of nodes is the size of the network matrix.
N = zeros(nodes,1);
N(1)=1;  % We assume that node 1 is infected at the beginning
% It is the attacker.

nodes=size(Network,1); % Number of nodes
Number_of_Links=0; % Counter to determine how many non-zero links there are.
for i=1:nodes  % Step over the origination node for each link.
    for j=1:nodes % Step over the destination node for each link.
        Number_of_Links=Number_of_Links+(Network(i,j)~=0); 
        % If there is a link for this origin and destination increment link counter
    end
end

%Stoichiometry, 
% In the Gellespie algorithm the stoichiometry is the amount that the state of the
% systems changes as a result of each reaction.  Each column of this matrix
% corresponds to the state change for the given reaction.
S1=zeros(nodes, Number_of_Links);  
% Stoichiometry of non-infective messages (S1) is zero --  no change in
% infectios state.
S2=zeros(nodes, Number_of_Links);  
% Initialize stoichiometry for infective messages (S2).
k=0; % counter for links.
for i = 1 : nodes  % Step over the origination node for each link.
    for j = 1 : nodes % Step over the destination node for each link.
        if Network(i,j)>0 % If there is a link for this origin and destination
            k=k+1; % increment counter.
            S2(j,k)=1; % The jth state increases from 0 to 1 for the kth reaction
        end
    end
end
S=[S1, S2]; % the total stoichiometry is the concatenation of the two stoichiometry
% vectors for the non-infective and infective reacrtions.

t=0; % Intialize time at zero.
s = zeros(20000,1);    % Intitialize vector to record the times of reactions
rxn = zeros(20000,1);  % Intitialize vector to record the types of reactions 
i_count=0; % Initialize counter of reactions.

time_n =inf*ones(1, nodes); % Initialize Times of infection for all nodes. 
% I set this to infinity.

while t<T % Run simulation until the final time.
    
% Compute Propensity Functions, w.  In the Gillespie algorithm, 
% the propensity function is the rate at which each reaction occurs.  
% This typically depends upon the current state of the process.
    W1=lambda1 * ones(Number_of_Links,1);  % the propensity of the first set of
    % reactions for the non-infected messages is costant, lambda1.
    W2 = zeros(Number_of_Links,1); % Initialize the propensities of the second set of
    % reactions for the infected messages.
    k=0; % counter for links.
    for i = 1 : nodes % Step over the origination node for each link.
        for j = 1 : nodes % Step over the destination node for each link.
            if Network(i,j)>0 % If there is a link for this origin and destination
                k=k+1; % increment counter.
                W2(k)=Network(i,j)*(N(i)==1); % The propensity function of this
                % reaction is the rate of infected message passage between
                % the ith and jth link (Network(i,j)) times the boolean
                % operator equal that describes if the ith node is
                % infected (ie., N(i) = 1 --> infected).
            end
        end
    end
    w=[W1; W2]; % Concatenate teh two propensity function vectors.
    
    w0 = sum(w); % Sum of the rates for all reactions
    
    t=t-log(rand)/w0; % Generate the time of next reaction.
    
    % Find the type of this reaction
    tmp = 1; % Set counter to the first reaction
    r2 = rand*w0; % Generate a uniform random number between zero and w0.
    while sum(w(1:tmp))<r2 % Sum over the propensities until the sum exceeds r2.
        tmp=tmp+1;
    end
    % tmp is now the index of the reaction that actually occurs at time 't'
    
    % If we are still less than T, record time and type of reaction.
    % Otherwise, the simulation is complete.
    if t<=T
        i_count=i_count+1; % Increment reaction counter.
        s(i_count) = t; % REcord time of the last reaction.
        rxn(i_count) = tmp; % Record type of the last reaction.
    end
    
    N = max(N,S(:,tmp));  
    % Update state following the last reaction.  Note that the only changes
    % are from 0 to 1.  If N(i)==0 and S(i,tmp)==1, then the state N(i) will
    % change from 0 to one.  Otherwise nothing changes.
    
    for i=1:nodes  % Step through the different nodes.
        if isinf(time_n(i))&&N(i)==1; % if the ith node just got infected,
            % then time_n(i) will be infinity.  If N(i) is now ==1, then we
            % record the time as the time the infection of the N(i) occurs.
            time_n(i)=t;
        end
    end
    
end

% Remove zeros from reaction history.
rxn = rxn(1:i_count);
s = s(1:i_count);

% Separate reactions into the five possible links
num=zeros(T, Number_of_Links);

k=0; % counter for links.
for i = 1 : nodes % Step over the origination node for each link.
    for j = 1 : nodes % Step over the destination node for each link.
        if Network(i,j)>0 % If there is a link for this origin and destination
            k=k+1; % increment counter.
            links(k).vec = s(rxn==k|rxn==(k + Number_of_Links));
            % Record the times of all messages sent along the kth link.
            % This includes both the non-infected messages (rxn==k) and the
            % infected messages (rxn==(k + Number_of_Links)).
            for t = 1:T % Run through the time points from 1 to T. 
                num(t,k)=sum(links(k).vec<=t);
                % Record how many messages were sent along the kth link
                % in the interval tau = [0,t].
            end
            
        end
    end
end

% Define output structure.
Data.time_n=time_n;
Data.num=num;
Data.t=t;
Data.links=links;



