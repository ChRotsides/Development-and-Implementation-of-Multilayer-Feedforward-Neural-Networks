classdef activationLayer < Layer
    
    properties
        activation
        dactivation
        
    end

    methods
        function obj=init(obj,activation,dactivation)
            obj.activation=activation;
            obj.dactivation=dactivation;
        end
        function obj=forward(obj,X)
            obj.input=X;
            obj.output=obj.activation(X);

        end
        
        function obj=backwords(obj,output_error,lr)
            
            obj.input_error=obj.dactivation(obj.input).*output_error;
        end
    end

      methods(Static)

      function ds=dsigmoid(x)
            ds=(1./(exp(-x)+1)).*(1-(1./(exp(-x)+1)));
        end
        function s=sig(x)
            s=(1./(exp(-x)+1));
        end
        
        function r=tanhP(x)
            r=1-tanh(x).^2;
        end
        function r=signP(x)
            r=1;
        end


      end
     


end