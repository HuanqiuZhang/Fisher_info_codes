function [I11,I12,I22] = Fisher_single_neuron_multiple_subfields(A,mu,Omega,stim)
%This function computes the fisher information matrix for a single neuron
%with starndard definition.
n_subfield = length(A);
I11 = 0;
I12 = 0;
I22 = 0;

term2 = zeros(1,n_subfield);
term3 = zeros(1,n_subfield);
term4 = zeros(1,n_subfield);
for k = 1:n_subfield
    term0 = (stim-mu(k,:))*squeeze(Omega(:,:,k))*(stim-mu(k,:))';
    term2(k) = A(k)*exp(-0.5*term0);
    term3(k) = (stim-mu(k,:))*squeeze(Omega(:,1,k));
    term4(k) = (stim-mu(k,:))*squeeze(Omega(:,2,k));
end
term1 = sum(term2);

for r = 0:100
    term5 = 1/factorial(r);
    I11 = I11 + term5*term1^(r-2)*exp(-term1)*(term1-r)^2*(sum(term2.*term3))^2;
    I12 = I12 + term5*term1^(r-2)*exp(-term1)*(term1-r)^2*(sum(term2.*term3))*(sum(term2.*term4));
    I22 = I22 + term5*term1^(r-2)*exp(-term1)*(term1-r)^2*(sum(term2.*term4))^2;

end
end
