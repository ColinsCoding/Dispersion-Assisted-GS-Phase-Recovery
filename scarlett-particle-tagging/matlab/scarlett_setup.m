function scarlett_setup()
%SCARLETT_SETUP  Add the Scarlett MATLAB folders to the path.
%
%   Run once per session from anywhere:
%       addpath('<repository>/matlab'); scarlett_setup
%   Folders are located relative to this file, so no personal paths are needed.

    here = fileparts(mfilename('fullpath'));
    folders = {'physics', 'sensor', 'visualization', 'tests'};
    for i = 1:numel(folders)
        addpath(fullfile(here, folders{i}));
    end
end
