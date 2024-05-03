%% clear section
clc;
clear;
close all;

%% Intialization
global SNR_dB ;
Eb = 1 ;
SNR_dB = -4 : 14 ;
SNR_linear = 10.^(SNR_dB/10);
NumOfBits = 1000032 ;
DataBits = randi([0,1],1,NumOfBits); % Generate random array of 0 & 1.

%% BPSK (Mapper) 
Actual_BER_BPSK = zeros(1,length(SNR_dB));
Theoretical_BER_BPSK = zeros(1,length(SNR_dB));
Demapped_BPSK = zeros(1,length(NumOfBits));
Mapped_BPSK = 2*DataBits-1; % Converting the bits to (1,-1) in new array with the same size of DataBits

%% BPSK (Channel & Demapper)
Noise_Bits = randn(1,size(Mapped_BPSK,2)); % AWGN noise with zero mean , one variance.

for i = 1:1:length(SNR_dB)
 N0 = Eb/(SNR_linear(i));    
 Noise = Noise_Bits.*sqrt(N0/2);  
 Channel_BPSK = Mapped_BPSK + Noise; % channel effect on BPSK_Data array
    
%Demapping recieved signals
for j=1:NumOfBits
 if(Channel_BPSK(j)<=0)
 Demapped_BPSK(j)=0;
 else
 Demapped_BPSK(j)=1;
 end
end

 % Calcilate BER
 BPSK_ErrorBits = abs(Demapped_BPSK - DataBits(1:NumOfBits));
 Actual_BER_BPSK(i) = sum(BPSK_ErrorBits) / NumOfBits;
 Theoretical_BER_BPSK(i) = 0.5*erfc(sqrt(Eb/N0)); 
end

% plot BPSK BER
Plot_BER(1,'Acutal','Theortical','BER BPSK',Actual_BER_BPSK,Theoretical_BER_BPSK);

%% QPSK (Mapper)
Actual_BER_QPSK = zeros(1,length(SNR_dB));  
Theoretical_BER_QPSK = zeros(1,length(SNR_dB));
Demapped_QPSK = zeros(1,NumOfBits);
Mapped_QPSK = zeros(1,length(DataBits)/2); 

%       |
%    01 | 11
% ------|------
%    00 | 10
%       |
for i=1:2:NumOfBits
    Mapped_QPSK((i+1)/2)=( DataBits(i+1)*2-1 )+( DataBits(i)*2-1 )* 1i;
end

%% QPSK (Channel & Demapper)
Noise_Bits = randn(1,size(Mapped_QPSK,2))+ randn(1,size(Mapped_QPSK,2))*1i;

for i=1:1:length(SNR_dB)
  N0 = Eb/(SNR_linear(i));
  Noise = Noise_Bits.*sqrt(N0/2);
  Channel_QPSK = Mapped_QPSK + Noise; % channel effect
 
  %Demapping recieved signals
  for j = 1:1:length(Channel_QPSK)
      if (imag( Channel_QPSK(j))>=0)
         Demapped_QPSK(j*2-1) = 1 ;
      else 
         Demapped_QPSK(j*2-1) = 0 ;
      end
      
      if (real(Channel_QPSK(j))>=0)
          Demapped_QPSK(j*2) = 1 ;
      else 
          Demapped_QPSK(j*2) = 0 ;
      end 
  end
  
 QPSK_ErrorBits = abs(Demapped_QPSK - DataBits(1:NumOfBits));
 Actual_BER_QPSK(i) = sum(QPSK_ErrorBits) / NumOfBits;
 Theoretical_BER_QPSK(i)=0.5*erfc(sqrt(Eb/N0));
end

% plot QPSK BER
Plot_BER(2,'Acutal','Theortical','BER QPSK',Actual_BER_QPSK,Theoretical_BER_QPSK);

%% 8-PSK (Mapper)
Eb = 1/3 ;
Actual_BER_MPSK = zeros(1,length(SNR_dB));   
Theoretical_BER_MPSK = zeros(1,length(SNR_dB)); 
Demapped_MPSK = zeros(1,NumOfBits);
Mapped_MPSK = zeros( 1,ceil(length(DataBits)/3) ); 
MPSK_Constall = [1,(1/sqrt(2))*(1+1i),(1/sqrt(2))*(-1+1i),1i, ...
    (1/sqrt(2))*(1-1i),-1i,-1,(1/sqrt(2))*(-1-1i)];

%    \ 011 /
%  010\   /001
%      \ /
%-110---------000-
%      /  \
%  111/    \100
%    / 101  \
for i=1:3:NumOfBits
    Mapped_MPSK((i+2)/3)= MPSK_Constall( ( DataBits(i)+DataBits(i+1)*2+DataBits(i+2)*4 ) +1 );
end

%% 8-PSK (Channel & Demapper)
Noise_Bits = randn(1,size(Mapped_MPSK,2))+randn(1,size(Mapped_MPSK,2))*1i;
Demapped_MPSK_Vector =  [0  0  0  1  0  0  0  1  0  1   1   0 ...
    0   0   1   1    0    1     0    1    1   1    1   1 ];

for i=1:1:length(SNR_dB)
 N0 = Eb/(SNR_linear(i));
 Noise = Noise_Bits.*sqrt(N0/2);
 Channel_MPSK = Mapped_MPSK + Noise;

 for j = 1:1:length(Channel_MPSK)
   [~ , index] = min(abs(Channel_MPSK(j) - MPSK_Constall));
   Demapped_MPSK((j-1)*3+1 : j*3) = Demapped_MPSK_Vector((index-1)*3+1 : index*3); % Extract and assign the three elements based on index
 end
 
 MPSK_ErrorBits = abs(Demapped_MPSK - DataBits(1:NumOfBits));
 Actual_BER_MPSK(i) = sum(MPSK_ErrorBits) / NumOfBits;
 Theoretical_BER_MPSK(i)=(1/3)*erfc(sqrt(3*Eb/N0)*sin(pi/8));
end

% plot MPSK BER
Plot_BER(3,'Acutal','Theortical','BER MPSK',Actual_BER_MPSK,Theoretical_BER_MPSK);

%% 16-QAM (Mapper)
Eb = 2.5 ;
Actual_BER_QAM = zeros(1,length(SNR_dB));   
Theoretical_BER_QAM = zeros(1,length(SNR_dB)); 
Demapped_QAM = zeros(1,NumOfBits);
Mapped_QAM = zeros( 1,ceil(length(DataBits)/4) ); 
QAM_Constall = [-3-3i,-3-1i,-3+3i,-3+1i,-1-3i,-1-1i,-1+3i,-1+1i,3-3i, ...
    3-1i,3+3i,3+1i,1-3i,1-1i,1+3i,1+1i];

% 0010  |  0110   3   1110    | 1010
%----------------------------------
% 0011  |  0111   1   1111    | 1011
%-(-3)-----(-1)---|-----1---------3--
% 0001  |  0101  -1   1101    | 1001
%----------------------------------
% 0000  |  0100  -3   1100    | 1000
for i=1:4:NumOfBits
    Mapped_QAM((i+3)/4)= QAM_Constall( (DataBits(i)+DataBits(i+1)*2+DataBits(i+2)*4+DataBits(i+3)*8) +1 );
end

%% QAM-PSK (Channel & Demapper)
Noise_Bits = randn(1,size(Mapped_QAM,2))+randn(1,size(Mapped_QAM,2))*1i;
Demapped_QAM_Vector =  [0,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,1,0, ...
    1,0,0,1,1,0,1,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,0,1,0,0,1,1,1,0,1, ...
    1,0,1,1,1,1,1,1,1];

for i=1:1:length(SNR_dB)
 N0 = Eb/(SNR_linear(i));
 Noise = Noise_Bits.*sqrt(N0/2);
 Channel_QAM = Mapped_QAM + Noise;

 for j = 1:1:length(Channel_QAM)
   [~ , index] = min(abs(Channel_QAM(j) - QAM_Constall));
   Demapped_QAM((j-1)*4+1 : j*4) = Demapped_QAM_Vector((index-1)*4+1 : index*4); % Extract and assign the three elements based on index  
 end
 
 QAM_ErrorBits = abs(Demapped_QAM - DataBits(1:NumOfBits));
 Actual_BER_QAM(i) = sum(QAM_ErrorBits) / NumOfBits;
 Theoretical_BER_QAM(i)=(1.5/4)*erfc(sqrt(Eb/(2.5*N0)));
end

% plot QAM BER
Plot_BER(4,'Acutal','Theortical','BER QAM',Actual_BER_QAM,Theoretical_BER_QAM);

%% Ploting BER for all case of PSK
Plot_BER_All(5,'BPSK BER', 'QPSK BER','MPSK BER','QAM BER', ...
        'Actual BER of PSK' ,...
        Actual_BER_BPSK,Actual_BER_QPSK,Actual_BER_MPSK,Actual_BER_QAM);

Plot_BER_All(6,'BPSK BER', 'QPSK BER','MPSK BER','QAM BER', ...
        'Theoretical BER of PSK' ,...
        Theoretical_BER_BPSK,Theoretical_BER_QPSK,Theoretical_BER_MPSK, ...
        Theoretical_BER_QAM);

%% QPSK (Mapper) without gray representation
Eb = 1;
Actual_BER_QPSK = zeros(1,length(SNR_dB));  
Theoretical_BER_QPSK = zeros(1,length(SNR_dB));
Demapped_QPSK = zeros(1,NumOfBits);
Mapped_QPSK = zeros(1,length(DataBits)/2); 
QPSK_Constall = [-1-1i,-1+1i,1+1i,1-1i];

%       |
%    01 | 10
% ------|------
%    00 | 11
%       |
for i=1:2:NumOfBits
    Mapped_QPSK((i+1)/2) = QPSK_Constall( (DataBits(i)+DataBits(i+1)*2) +1 );
end

%% QPSK (Channel & Demapper)
Noise_Bits = randn(1,size(Mapped_QPSK,2))+ randn(1,size(Mapped_QPSK,2))*1i;
Demapped_QPSK_Vector =  [0,0,1,0,0,1,1,1];

for i=1:1:length(SNR_dB)
  N0 = Eb/(SNR_linear(i));
  Noise = Noise_Bits.*sqrt(N0/2);
  Channel_QPSK = Mapped_QPSK + Noise; % channel effect
 
  %Demapping recieved signals
 for j = 1:1:length(Channel_QPSK)
   [~ , index] = min(abs(Channel_QPSK(j) - QPSK_Constall));
   Demapped_QPSK((j-1)*2+1 : j*2) = Demapped_QPSK_Vector((index-1)*2+1 : index*2); % Extract and assign the three elements based on index  
 end
  
 QPSK_ErrorBits = abs(Demapped_QPSK - DataBits(1:NumOfBits));
 Actual_BER_QPSK(i) = sum(QPSK_ErrorBits) / NumOfBits;
 Theoretical_BER_QPSK(i)=0.5*erfc(sqrt(Eb/N0));
end

% plot QPSK BER
Plot_BER(7,'Acutal','Theortical','BER QPSK',Actual_BER_QPSK,Theoretical_BER_QPSK);

%% BFSK (Mapper)
Eb = 1;
Actual_BER_BFSK = zeros(1,length(SNR_dB));  
Theoretical_BER_BFSK = zeros(1,length(SNR_dB));
Demapped_BFSK = zeros(1,NumOfBits);
Mapped_BFSK = zeros(1,length(DataBits));

for i=1:1:NumOfBits
    if ( DataBits(i) == 0 )
    Mapped_BFSK(i) = 1;
    else 
    Mapped_BFSK(i) = 1*1i;
    end 
end

%% BFSK (Demapper && Channel)
Noise_Bits = randn(1,size(Mapped_BFSK,2));

for i=1:1:length(SNR_dB)
  N0 = Eb/(SNR_linear(i));
  Noise = Noise_Bits.*sqrt(N0);
  Channel_BFSK = Mapped_BFSK + Noise; % channel effect
 
  %Demapping recieved signals
for j = 1:1:length(Channel_BFSK)
    Difference = imag(Channel_BFSK(j)) - real(Channel_BFSK(j));
    if (Difference >= 0)
        Demapped_BFSK(j) = 1;
    else 
        Demapped_BFSK(j) = 0;
    end 
end
  
 BFSK_ErrorBits = abs(Demapped_BFSK - DataBits(1:NumOfBits));
 Actual_BER_BFSK(i) = sum(BFSK_ErrorBits) / NumOfBits;
 Theoretical_BER_BFSK(i)=0.5*erfc(sqrt(Eb/(2*N0)) );
end

% plot BFSK BER
Plot_BER(8,'Acutal','Theortical','BER BFSK',Actual_BER_BFSK,Theoretical_BER_BFSK);

%% PSD of BFSK
% Initialization of variables
fs=100;
num_samples = 7;
Tb = 0.07;
t = 0 : 1/fs : Tb;
t = t(1:end-1);
f = 1 / Tb; % frequency of symbols

Ensemble_size = 7500;         % number of waveforms
LengthOfrealization = 100;     % length of each waveform

% Initializing matrices
Data_bits = randi([0, 1], Ensemble_size, LengthOfrealization+1); % Generate random bits
SampledData_bits = repelem(Data_bits, 1, 7); % Repeat each bit 7 times (7 samples per bit)
Delayed_mapped_BFSK = zeros(1,LengthOfrealization*7);
Mapped_BFSK = zeros(Ensemble_size, (LengthOfrealization*7)+7);

% Mapper
for i = 1:Ensemble_size
    for j = 1:7:(LengthOfrealization*7)+7
        if SampledData_bits(i, j) == 1
            Mapped_BFSK(i, j : j+6 ) = sqrt(2/Tb);
        else
            Mapped_BFSK(i, j : j+6 ) = sqrt(2/Tb) * (cos(2 * pi * f * t) + 1i * sin(2 * pi * f * t));
        end
    end
end

% channel effect
T_Delay = randi([0, 6], Ensemble_size, 1); % Random start for each realization
for i = 1:Ensemble_size
    Delayed_mapped_BFSK(i, :) = Mapped_BFSK(i, 1 + T_Delay(i) : (LengthOfrealization*7) + T_Delay(i) );
end

% autocorrelation calculations
Auto_Corr = zeros(1, (LengthOfrealization*7) );
for tau = (-(LengthOfrealization*7)/2+1) : (LengthOfrealization*7)/2 
    % Calculate the product of the signal and its complex conjugate shifted by tau
    x = conj(Delayed_mapped_BFSK(:, (LengthOfrealization*7)/2)) .* Delayed_mapped_BFSK(:, (LengthOfrealization*7)/2 + tau);
    
    % Compute the autocorrelation value for this lag (tau)
    Auto_Corr(tau + (LengthOfrealization*7)/2) = sum(x) / Ensemble_size;
end

% PSD of BFSK
Actual_PSD_BFSK = abs(fftshift(fft(Auto_Corr)))/fs; % actual

% Define the frequency vector based on the length of auto_corr
f = (-(LengthOfrealization*7)/2 : (LengthOfrealization*7)/2 - 1) * fs / (LengthOfrealization*7);

% Calculate theoretical PSD based on the formula
X0 = 1 / Tb;                  % Value of interest for deltaDirac function
Delta1 = deltaDirac(X0, f);   % Calculate deltaDirac for the given frequencies
Delta2 = deltaDirac(-X0, f);  % Calculate deltaDirac for the given frequencies

% Construct the theoretical PSD using the calculated delta values
Theoretical_PSD_BFSK = ((2/Tb) * (Delta1 + Delta2)) + ...
                  ((8 * cos(pi * Tb * f).^2) ./ (pi^2 * (4 * Tb^2 * f.^2 - 1).^2));

% PSD plotting
x_axis = (f*Tb);
plot(x_axis, Theoretical_PSD_BFSK,'r-','LineWidth',1);
xlabel('Frequency normalized');
ylabel('PSD');

% Plotting the statistical PSD
hold on;
plot(x_axis, Actual_PSD_BFSK,'b-');
xlabel('Frequency normalized');
ylabel('PSD');
title("PSD of BFSK");
legend('Theoretical','Actual')
ylim([0,3]);

%% Functions
function Plot_BER(figureNum, text1, text2, text3, Actual_BER, Theoretical_BER)
    global SNR_dB ;
    figure(figureNum);
    semilogy(SNR_dB, Actual_BER, 'b');
    hold on;
    semilogy(SNR_dB, Theoretical_BER, 'r --');
    legend(text1, text2);
    title(text3);
    xlabel('Eb/No (dB)');
    ylabel('BER');
    ylim([1e-5, 1e0]);
    grid on;
    hold off;
end

function Plot_BER_All(figureNum, text1, text2, text3 , text4 , text5 , BER1 , BER2 , BER3 , BER4)
    global SNR_dB ;
    figure(figureNum);
    semilogy(SNR_dB, BER1, 'b--');
    hold on;
    semilogy(SNR_dB, BER2, 'r--o');
    hold on;
    semilogy(SNR_dB, BER3, 'g');
    hold on;
    semilogy(SNR_dB, BER4, 'y');
    legend(text1', text2,text3,text4);
    title(text5);
    xlabel('Eb/No (dB)');
    ylabel('BER');
    ylim([1e-5, 1e0]);
    grid on;
    hold off;
end

% deltaDirac function to make impluse response at specific x_values axis by x
function y = deltaDirac(x, x_values)
    % Function to compute discrete approximation of Dirac delta function

    % Compute the delta function approximation
    y = 99 * double(abs(x_values - x) < 0.01);  % Adjust the tolerance as needed
end