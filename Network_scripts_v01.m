%% y vals first

load('.\MatFiles\inputs.mat');
load('.\MatFiles\outputs.mat');

outputsy = [];

ins = inputs;
targets = outputs;

% Create a Pattern Recognition Network
hiddenLayerSize = [2500 1000 500 250];
% notes on network:
%hiddenLayerSize = [2500 1000 500 250]; BEST - err of 61

nety_225 = feedforwardnet(hiddenLayerSize,'trainscg');
% nety_225.trainFcn = 'trainscg'; %gradient training

% Set up Division of Data for Training, Validation, Testing
nety_225.divideParam.trainRatio = 80/100;
nety_225.divideParam.valRatio = 10/100;
nety_225.divideParam.testRatio = 10/100;

nety_225.trainparam.epochs = 8000;
nety_225.trainparam.goal = 1e-100000;

% Train the Network
[nety_225,tr] = train(nety_225,ins,targets,'useGPU','yes');

% Test the Network
outputsy = nety_225(ins,'useGPU','yes');
errorsy = abs((abs(gsubtract(targets,outputsy))./targets)*100);

%sum errors greater than 100%
erry = sum(sum(errorsy>100))

% save('.\MatFiles\nety_225.mat','nety_225')

%% plot results

% load nety_225
% outputsy = nety_225(ins,'useGPU','yes');

netouts = [];

for i = 1:length(outputsy)
    %netouts_i = nety_225(inputs(:,i));
    if outputsy(i) < 0
        netouts(i) = 0;
    else
        netouts(i) = outputsy(i);
    end
    
end

x = 1:length(netouts);

plot(x,netouts)
hold on
plot(x,outputs)

