% export_disk_strels.m
% Exports exact strel('disk', r).Neighborhood boolean masks to .mat files,
% for use in EccoPy (Python) as bit-exact replacements for the naive
% Euclidean-circle _disk(r) approximation.
%
% Usage (see run_export_disk_strels.sh):
%   export RADII="3,5,15,25"
%   export OUTDIR="./disk_strels"
%   matlab -nodisplay -nosplash -r "run('export_disk_strels.m'); exit"
%
% Each output file disk_strel_r<R>.mat contains one variable:
%   nhood  -- logical (2R+1)x(2R+1) array, exactly strel('disk',R).Neighborhood

radiiStr = getenv('RADII');
if isempty(radiiStr)
    radiiStr = '3,5,15,25';  % default: matches enlarge_mixed=5, enlarge_conv=5
end
radii = str2double(strsplit(radiiStr, ','));

outdir = getenv('OUTDIR');
if isempty(outdir)
    outdir = '.';
end
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

for r = radii
    se = strel('disk', r);
    nhood = se.Neighborhood;

    fname = fullfile(outdir, sprintf('disk_strel_r%d.mat', r));
    save(fname, 'nhood');

    fprintf('radius %d: saved %s  (shape %dx%d, %d true pixels)\n', ...
        r, fname, size(nhood,1), size(nhood,2), sum(nhood(:)));
end

fprintf('Done. Exported %d structuring element(s) to %s\n', numel(radii), outdir);
