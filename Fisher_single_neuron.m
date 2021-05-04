function [I11,I12,I22] = Fisher_single_neuron(A,mu,Omega,stim)
%This function computes the fisher information matrix for a single neuron
%with standard definition.
I11 = 0;
I12 = 0;
I22 = 0;
term1 = (stim-mu)*Omega*(stim-mu)';
term2 = (stim-mu)*Omega(:,1);
term3 = (stim-mu)*Omega(:,2);
term4 = exp(-0.5*term1);
term5 = exp(-A*term4);

for r = 0:100
    term6 = exp(-0.5*r*term1);
    term7 = (A^r)/factorial(r);
    I11 = I11 + term7*term6*term5*((A*term4-r)*term2)^2;
    I12 = I12 + term7*term6*term5*((A*term4-r)*term2)*((A*term4-r)*term3);
    I22 = I22 + term7*term6*term5*((A*term4-r)*term3)^2;
    
end
end
