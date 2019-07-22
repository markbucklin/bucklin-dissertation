function stat = updateStatisticsGPU(F, stat)

if isInitialized()
	[N,Fmin,Fmax,M1,M2,M3,M4] = fromStatStruct();
	if numFrames >= 1
		[Fmin,Fmax,M1,M2,M3,M4] = arrayfun(@statUpdateKernel,Fmin,Fmax,M1,M2,M3,M4,N, rowSubs,colSubs,chanSubs);
	end
else
	[N,Fmin,Fmax,M1,M2,M3,M4] = initializeStats();
end


	function [fmin,fmax,m1,m2,m3,m4] = statUpdateKernel(fmin,fmax,m1,m2,m3,m4,n, rowIdx,colIdx,chanIdx)

		k = int32(0);
		while k < numFrames

			% UPDATE SAMPLE INDICES (USUALLY TEMPORAL)
			k = k + 1;
			n = n + 1;

			% GET PIXEL SAMPLE
			f = single(F(rowIdx,colIdx,chanIdx,k));

			% PRECOMPUTE & CACHE SOME VALUES FOR SPEED
			d = f - m1;
			dk = d/n;
			dk2 = dk^2;
			s = d*dk*(n-1);

			% UPDATE CENTRAL MOMENTS
			m1 = m1 + dk;
			m4 = m4 + s*dk2*(n.^2-3*n+3) + 6*dk2*m2 - 4*dk*m3;
			m3 = m3 + s*dk*(n-2) - 3*dk*m2;
			m2 = m2 + s;

			% UPDATE MIN & MAX
			fmin = min(fmin, f);
			fmax = max(fmax, f);

		end

	end









end











