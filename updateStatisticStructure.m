function stat = updateStatisticStructure(stat, Finput)
% updateStatisticStructure - updates a structure of cumulative sequential pixel statistics
%
% Syntax:
%			>> stat = updateStatisticStructure(stat, Finput)
%




% GET/ENSURE DATA INPUT IS IN EXPECTED FORMAT (NUMERIC-ARRAY & FLOATING-POINT)
useFloat = true;
F = ignition.shared.formatVideoNumericArray(Finput, useFloat);

% GET DIMENSIONS OF INPUT
[numRows, numCols, numChannels, numFrames] = size(F);
frameDim = 4;


% INITIALIZE STAT STRUCTURE
if (nargin < 2)
	stat = struct.empty();
end
if isempty(stat)
	
	if isa(F, 'gpuArray')
		initZeroPixel = gpuArray.zeros(1,floatClassName);
		initZeroFrame = gpuArray.zeros(numRows, numCols, numChannels, floatClassName);
	else
		initZeroPixel = zeros(1,floatClassName);
		initZeroFrame = zeros(numRows, numCols, numChannels, floatClassName);
	end
	stat.N = initZeroPixel;
	stat.Min = min(F, [], frameDim);
	stat.Max = max(F, [], frameDim);
	stat.M1 = initZeroFrame;
	stat.M2 = initZeroFrame;
	stat.M3 = initZeroFrame;
	stat.M4 = initZeroFrame;
end


% STATISTIC CONTRIBUTION COUNTS/WEIGHTS
na = stat.N;
nb = numFrames;
n = na + nb;

% UPDATE MAX & MIN
stat.Min = min(min(F, [], frameDim), stat.Min);
stat.Max = max(max(F, [], frameDim), stat.Max);

% UPDATE CENTRAL MOMENTS
if nb == 1 % Run faster implementation if only updating with 1 frame
	d = cast(F, floatClassName) - stat.M1;
	dk = d./n;
	dk2 = dk.^2;
	s = d.*dk.*(n-1);
	stat.M1 = stat.M1 + dk;
	m2 = stat.M2;
	m3 = stat.M3;
	stat.M4 = stat.M4 + s.*dk2.*(n^2-3*n+3) + 6*dk2.*m2 - 4.*dk.*m3;
	stat.M3 = m3 + s.*dk.*(n-2) - 3.*dk.*m2;
	stat.M2 = m2 + s;
	
else	% Not optimized, but easier to follow
	m1a = stat.M1;
	m2a = stat.M2;
	m3a = stat.M3;
	m4a = stat.M4;
	
	m1b = cast(mean(F, frameDim, 'default'), floatClassName);
	m2b = moment(cast(F,floatClassName), 2, frameDim);
	m3b = moment(cast(F,floatClassName), 3, frameDim);
	m4b = moment(cast(F,floatClassName), 4, frameDim);
	
	d = bsxfun(@minus, m1b , m1a);
	stat.M1 = bsxfun(@plus, m1a , bsxfun(@times, d, (nb/n))); % 				dk = d.*(Nb/N);
	stat.M2 = m2a  +  m2b  +  (d.^2).*(na*nb/n); % dk2 = (d.^2).*Na.*Nb./N
	stat.M3 = m3a  +  m3b  +  (d.^3).*(na*nb*(na-nb)/(n^2))  ...
		+  3*(na.*m2b - nb.*m2a).*d./n;
	stat.M4 = m4a  +  m4b  +  (d.^4).*((na*nb*(na-nb)^2)/(n^3))  ...
		+  6*(m2b.*na^2 + m2a.*nb.^2).*((d.^2)./(n^2))  ...
		+  4*(m3b.*na  -  m3a.*nb).*(d./n);
end

% UPDATE SAMPLE COUNT
stat.N = n;




end
