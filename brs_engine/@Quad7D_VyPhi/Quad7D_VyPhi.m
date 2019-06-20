classdef Quad7D_VyPhi < DynSys

    properties
        
        T1Max
        T1Min
        T2Max
        T2Min

        % "Real" parameters
        m % mass

        transDrag %translational drag

        % Ficticious parameter for decomposition
        wpRange
    end

    methods
        function obj = Quad7D_VyPhi(x, T1Min, T1Max, T2Min, T2Max, wpRange, m, transDrag)
          % Dynamics:
          %    \dot v_y  = -transDrag*v_y/m - T1*sin(\phi)/m - T2*sin(\phi)/m
          %    \dot \phi = \omega_p
          %
          % Inputs:
          %   T1Max, T1Min, T2Max, T2Min - limits on T1 and T2 (controls
          %   m - mass
          %   grav - gravity
          %   transDrag - translational Drag
          
          if nargin < 2
            T1Min = 0;
          end
          
          if nargin < 3
            T1Max = 0.14;
          end
          
          if nargin < 4
            T2Min = 0;
          end
          
          if nargin < 5
            T2Max = 0.14;
          end
         
         if nargin < 6
            wpRange = [0 2*pi];
          end


          if nargin < 7
            m = 0.027;
          end      

          if nargin < 8
            transDrag = 7.93 * 10^-12 
          end      
          
          % Basic vehicle properties
          obj.nx = 2;
          obj.nu = 3;  
          
          obj.x = x;
          obj.xhist = obj.x;
          
          obj.T1Max = T1Max;
          obj.T1Min = T1Min;
          obj.T2Max = T2Max;
          obj.T2Min = T2Min;
          obj.m = m;
          obj.transDrag = transDrag;
          obj.wpRange = wpRange;
        end
        
    end % end methods
end % end classdef

