% Obtain input image
inputImage = imread('lenna_halftone.png');

% Ensure input is grayscale
if (size(size(inputImage),2) == 3)
    grayImage = rgb2gray(inputImage);
else
    grayImage = inputImage;
end

% Frequency Transformation
F_t0 = fft2(grayImage);
F_t1 = fftshift(F_t0);
F_t2 = abs(F_t1);
F_t3 = log(F_t2+1);
freqImg = mat2gray(F_t3);

% Create a square mask
[row_count,col_count] = size(freqImg)
larger_size = col_count;
if (row_count > col_count)
    larger_size = row_count;
end

% The center frequency is the center of the image
fNy = larger_size/2;       % Nyquist frequency [Hz]
fs = fNy*2;                % Number of samples
Ts = 1/fs;                 % Sampling period [s]
duration = 1-1/fs;         % Duration [s]
t = 0:Ts:duration; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize Mask
mask = freqImg;

% Show data about mask
% figure(6)
% imhist(mask)

% Reoccurring frequencies will be masked
% FUTURE IMPROVEMENT: Thresholding method must be changed
mask(mask<0.63) = 0;
mask(mask>=0.63) = mask(mask>=0.63);

mask2 = mask(1:round(row_count/2),round(col_count/2:end));
row_mask_sum = flip(sum(mask2,2));
[y_min, y_index_min] = min(row_mask_sum)
[y_max, y_index_max] = max(row_mask_sum(y_index_min:end))
y_index_max = row_count/2 - y_index_max - y_index_min
[x_max, x_index_max] = max(mask2(y_index_max,:));

rgbm2 = cat(3, mask2, mask2, mask2);
rgbm2 = insertMarker(rgbm2,[x_index_max y_index_max]);

rgbImage = cat(3, freqImg, freqImg, freqImg);
rgbImage = insertMarker(rgbImage,[row_count/2+x_index_max y_index_max],'color','red');
rgbImage = insertMarker(rgbImage,[row_count/2+x_index_max col_count-y_index_max],'color','red');
rgbImage = insertMarker(rgbImage,[row_count/2-x_index_max y_index_max],'color','red');
rgbImage = insertMarker(rgbImage,[row_count/2-x_index_max col_count-y_index_max],'color','red');


figure(4)
imshow(rgbImage)

figure(5)
imshow(rgbm2)

% Create a 2-D representation of frequency in the x
% FUTURE IMPROVEMENT: Search frequencies in both X and Y instead of just X
harm_mask_x = sum(mask,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the harmonic order
harm_order = 4;
max_index = zeros(harm_order,2);

% First Harmonic
[max_index(1,1),max_index(1,2)] = max(harm_mask_x(:));

% First signal
first_sig = max_index(1,1).*sin(2*pi*t);
xn = first_sig;

for i = 2:harm_order
    % FUTURE IMPROVEMENT: Assumption that a minimum will occur before a
    % maximum for the offset
    [~,offset] = min(harm_mask_x(max_index(i-1,2):end));                                    % set offset
    [max_index(i,1),max_index(i,2)] = max(harm_mask_x((max_index(i-1,2) + offset):end));    % find next harmonic

    max_index(i,2) = max_index(i,2) + max_index(i-1,2) + offset;                            % set new maxima
    % FUTURE IMPROVEMENT: Use a gauspulse instead of Sine wave
    xn = xn + max_index(i,1).*sin(2*pi* (max_index(i,2) - max_index(1,2)) .* t);            % add harmonic
end

% Used for making impulse signals into gaussian shapes
gaus_mat = fspecial('Gaussian',40,6);

% Create Harmonic Mask
img_xn_x = repmat(xn,larger_size,1);
img_xn_y = img_xn_x';
img_xn = img_xn_y.*img_xn_x;
img_xn = img_xn(1:row_count, 1:col_count);

% Fourier of Harmonic Mask
F_a = abs(fftshift(fft2(img_xn)));
F_a = imfilter(F_a, gaus_mat,'conv');
F_a = (1-rescale(F_a,0,1));

% First Harmonic/ Center of the image
img_center = repmat(first_sig,larger_size,1);
img_center = img_center.*img_center';
img_center = img_center(1:row_count, 1:col_count);

% Fourier of First Harmonic
F_center = abs(fftshift(fft2(img_center)));
F_center = imfilter(F_center, gaus_mat,'conv');
F_center = (rescale(F_center,0,1));

% Remove First Harmonic from mask to preserve low frequencies
F_center(F_center <= mean(mean(F_center))) = 0;
F_a(F_center>0) = 1;

% Emphasize the masked values
% FUTURE IMPROVEMENT: Thresholding method must be changed
F_a(F_a < .99) = 0;
F_a(F_a >= .99) = rescale(F_a(F_a >= .99),0,1);

% Apply mask
F_A_model = F_t0;
F_A_model = F_A_model.*fftshift(F_a);

% Shift abs and log to see how the mask affected image
F_A_alt_model = log(abs(fftshift(F_A_model))+1);
F_A_alt_model = mat2gray(F_A_alt_model);

% Convert back to see the new image
IFF_A_model = ifft2(F_A_model);                 % take IFFT
FINAL_IM_A_model = uint8(real(IFF_A_model));      % Take real part and convert back to UINT8
FINAL_IM_A_model = uint8(255 * mat2gray(FINAL_IM_A_model));

FINAL_IM_A_model = medfilt2(FINAL_IM_A_model,[round(row_count/100),round(col_count/100)]);

% Convert to see mask
xn_fft = fftshift(abs(fft(xn)));
xn_fft = imfilter(xn_fft, gaus_mat,'conv');
xn_fft = rescale(xn_fft,0,max_index(1,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1);
hold on
for i = 2:harm_order
    plot(max_index(i,2),max_index(i,1),'r*');
end
plot(harm_mask_x);
plot(xn_fft);
title("Frequency Modeling")
legend("2nd Harmonic","3rd Harmonic","4th Harmonic","Sum of Column Frequencies","Generated Harmonic Model")
hold off

figure(2);
subplot(1,3,1)
imshow(F_a)
title("Mask in Freq")
subplot(1,3,2)
imshow(F_A_alt_model)
title("Freq of Restored Image")
subplot(1,3,3)
imshow(FINAL_IM_A_model);
title("Restored Image")
%imcontrast;