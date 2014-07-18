%%% plot ARMSE


clear;
clc;

BW = [10e6:5e6:100e6];

for i = 1: length(BW),
    i
    [fwf(:,i), ctr(:,i)] = Main2(BW(i));
end


figure;
plot(BW/1e6,ctr(1,:),'b--d','LineWidth',2);hold on;
plot(BW/1e6,fwf(1,:),'r*-','LineWidth',2);grid on;
legend('CTR','Fixed waveform');
ylabel('Acc. RMSE_{Alt.}');
xlabel('Bandwidth (MHz)');


figure;
plot(BW/1e6,ctr(2,:),'b--d','LineWidth',2);hold on;
plot(BW/1e6,fwf(2,:),'r*-','LineWidth',2);grid on;
legend('CTR','Fixed waveform');
ylabel('Acc. RMSE_{Vel.}');
xlabel('Bandwidth (MHz)');



figure;
plot(BW/1e6,ctr(3,:),'b--d','LineWidth',2);hold on;
plot(BW/1e6,fwf(3,:),'r*-','LineWidth',2);grid on;
legend('CTR','Fixed waveform');
ylabel('Acc. RMSE_{Bal.}');
xlabel('Bandwidth (MHz)');