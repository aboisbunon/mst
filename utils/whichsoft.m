function [ soft ] = whichsoft()
% WHICHSOFT Finds whether the code is running under Matlab or GNU Octave

   octave_info = ver('octave') ;
   matlab_info = ver('matlab') ;
   
   if ~isempty(octave_info)
       soft = octave_info.Name ;
   end

   if ~isempty(matlab_info)
       soft = matlab_info.Name ;
   end
   
end

