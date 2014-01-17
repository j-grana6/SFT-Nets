function Prob_Model_Given_Data = Get_Prob_Model_Given_Data(time_array,Network,Data,lambda1,T)

%% First we compute the probability of this particular time ordering
[~,Sequence] = sort(time_array);
% Sort the proposed time array into consecutive order.  Record the
% sequences of the events from first to last.

Log_Prob_Sequence = Get_Prob_Sequence_Given_Model(Sequence,Network)
% Call function to compute the probability of this sequence given the
% Network topology and reaction rates.

%% Second we compute the probability of the specific times given the ordering.
Log_Prob_Specific_Times_Given_Sequence = Get_Prob_Time_Given_Sequence(time_array,Sequence,Network)
%Call functions to compute the probability of the specific times given the
%sequence of events and the given network topology and reaction rates.

%% Third we compute the probability of the data given the time array.
Log_Prob_Data_Given_Times = Get_Prob_Data_Given_Time(time_array,Network,Data,lambda1,T)
% Call function to compute the probabilty of the data (message passage
% times) given the specific times of the nodes becoming infected and the
% rates associated with non-infectd and infected nodes.
%% Fourth, we combine the previous three to get the probavility of the model given the data.
Prob_Model_Given_Data = Log_Prob_Data_Given_Times+...
    Log_Prob_Specific_Times_Given_Sequence+...
    Log_Prob_Sequence;

function Log_Prob_Sequence = Get_Prob_Sequence_Given_Model(Sequence,Network)
nodes=length(Network);
State_of_Infection = zeros(nodes,1);  % Initialize the current state.
State_of_Infection(1)=1;  % Node 1 is infected at the beginning, it is the attacker.
Log_Prob_Sequence = 0; %initialize the probability of the sequence at P=1, log(P)=0;
for i=2:length(Sequence)
    % Sequence(i) is the next node to get infected.
    % Network(:,Sequence(i)) is the rate of this node to get infected from
    % neighbors.
    % State_of_Infection is the list of infected (1) and uninfected nodes (0).
    Prob_Next = State_of_Infection'*Network(:,Sequence(i));
    % Probability that the next node is infected by a neighbor.  when this
    % happens sequence(i) must be connected to a node that is already
    % infected.  The total rate for this to occur is the sum over all
    % nodes connected to Sequence(i) that are already infected.

    Prob_Total = State_of_Infection'*Network*(1-State_of_Infection);
    % Probability of all possible events. Only the nodes that are not
    % infected already can become infected; this is represted by the
    % vector (1-State_of_Infection).  Only the infected nodes can pass
    % infection to a neighbor; this is represented by (State_of_infection).
    % The rate in Network define the rates for all combinations.

    Log_Prob_Sequence = Log_Prob_Sequence+log(Prob_Next)-log(Prob_Total);
    % Sum of log probabilities for each event in the sequence. This
    % corresponds to a product of the previous iteration with
    % Prob_Next/Prob_Total.

    State_of_Infection(Sequence(i))=1;  %Update State of infection to reflect the last event.
end

function Log_Prob_Specific_Times_Given_Sequence = Get_Prob_Time_Given_Sequence(time_array,Sequence,Network)
nodes=length(Network);
State_of_Infection = zeros(nodes,1);  % Current state.
State_of_Infection(1)=1;  % Node 1 is infected at the beginning, it is the attacker.
Log_Prob_Specific_Times_Given_Sequence = 0;
for i=2:length(Sequence)
    % Sequence(i) is the next node to get infected.
    % Network(:,Sequence(i)) is the rate of this node to get infected from
    % neighbors.
    % State_of_Infection is the list of infected (1) and uninfected nodes (0).
    Prob_Total = State_of_Infection'*Network*(1-State_of_Infection);
    % Probability of all possible events. Only the nodes that are not
    % infected already can become infected; this is represted by the
    % vector (1-State_of_Infection).  Only the infected nodes can pass
    % infection to a neighbor; this is represented by (State_of_infection).
    % The rate in Network define the rates for all combinations.

    delT = time_array(Sequence(i))-time_array(Sequence(i-1));
    % Difference in time since last reaction.

    Log_Prob_Specific_Times_Given_Sequence = Log_Prob_Specific_Times_Given_Sequence+log(Prob_Total)-(Prob_Total*delT);
    % Probability density of the this specific time is
    % (probtotal)*exp(-probtotal*delT).
    % Use this to update teh total probability.

    State_of_Infection(Sequence(i))=1;  %Update State
end

function Log_Prob_Data_Given_Times = Get_Prob_Data_Given_Time(time_array,Network,Data,lambda1,T)
nodes=length(Network);
k=0;
LogL_Nodes = zeros(1,nodes);
for i = 1 : nodes  % Sending node.
    t_change = time_array(i);
    for j = 1 : nodes  % Receiving node
        if Network(i,j)>0
            k=k+1;
            num(k)=sum(Data.links(k).vec<=t_change)
            N=size(Data.links(k).vec,1)
            lambda2 = lambda1+Network(i,j)
            logL(k)=num(k).*log(lambda1 * t_change) + (N-num(k)) .* log(lambda2 * (T - t_change)) - ...
                logfactorial(num(k))*log(10) - logfactorial(N - num(k))*log(10) - lambda1 .* t_change - ...
                lambda2 .* (T-t_change)
            LogL_Nodes(i) = LogL_Nodes(i) + logL(k);
        end
    end
end
% time_array
% logL
% pause
Log_Prob_Data_Given_Times=sum(LogL_Nodes(2:nodes));
