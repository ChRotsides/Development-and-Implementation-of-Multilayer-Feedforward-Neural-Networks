classdef Layer
    properties
        W
        input
        output
        input_error
        output_error
        bias
    end

   methods
       function obj=init(obj,input_size,output_size)
            obj.W=(rand(input_size,output_size)*2)-1;
            obj.bias=(rand(1,output_size)*2)-1;
       end
       function obj=forward(obj,x)
                obj.input=x;
               obj.output=x*obj.W ;
               obj.output=obj.output+ obj.bias;
       end
       
       function obj=backwords(obj,output_error,lr)
            obj.input_error= output_error*obj.W';
            weights_error=obj.input'*output_error;

           obj.W = obj.W - lr.*weights_error;

           obj.bias=obj.bias-lr*output_error;
       end
   end


end