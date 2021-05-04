%This code runs the simulation shown in Fig.5 of the paper

%Each cell has multiple subfields, modeled with Gamma distribution with
%statistics matched to the experimental data
model_mean = 0.9794;
model_std = 1.1016;

%compute scale and shape parameters of gamma distribution
model_scale = model_std^2/model_mean;
model_shape = model_mean/model_scale;

%simulation
for zeta = 1./[1:20]  %note this is actually zeta inverse
    for N = [50 100 200 400 800 1600 3200 6400 12800]
        dim = 2;
        rep = 10000;
        All_Fisher_info = zeros(dim,dim,rep);
        for j = 1:rep
            %sample # subfields
            n_subfields = gamrnd(model_shape,model_scale,[1,N]);
            n_subfields = round(n_subfields);
            total_subfields = sum(n_subfields);
            %sample neuron parameters
            A = 20*rand(total_subfields,1)+5;
            mu = 1.8*rand(total_subfields,dim);
            %exponential distribution of sigma
            sigma = exprnd(zeta,total_subfields,dim);
            
            
            if dim == 2  %randomly rotate the cov matrix
                theta = 2*pi*rand(total_subfields,1);
                Sigma = zeros(2,2,total_subfields);
                for i = 1:total_subfields
                    R = [cos(theta(i)) -sin(theta(i));sin(theta(i)) cos(theta(i))];
                    Sigma(:,:,i) = R*diag(sigma(i,:).^2)*R';   %covariance matrix
                end
                %information matrix
                Omega = zeros(2,2,total_subfields);
                for i = 1:total_subfields
                    Omega(:,:,i) = inv(Sigma(:,:,i));
                end
            end
            
            %compute Fisher information matrix
            stim = 1.8*rand(1,dim);
            Fisher_info = zeros(dim,dim);
            count = 1;
            for i = 1:N
                if n_subfields(i) ~=0
                    A_temp = A(count:count+n_subfields(i)-1);
                    mu_temp = mu(count:count+n_subfields(i)-1,:);
                    Omega_temp = Omega(:,:,count:count+n_subfields(i)-1);
                    
                    [I11,I12,I22] = Fisher_single_neuron_multiple_subfields(A_temp,mu_temp,Omega_temp,stim);
                    if ~isnan(I11) && ~isnan(I12) && ~isnan(I22) && isfinite(I11) && isfinite(I12) && isfinite(I22)
                        Fisher_info(1,1) = Fisher_info(1,1) + I11;
                        Fisher_info(1,2) = Fisher_info(1,2) + I12;
                        Fisher_info(2,1) = Fisher_info(2,1) + I12;
                        Fisher_info(2,2) = Fisher_info(2,2) + I22;
                    end
                    
                    count = count+n_subfields(i);
                end
            end
            All_Fisher_info(:,:,j) = Fisher_info;
        end
        save(strcat('FishInfo_subfields_exp_N',num2str(N),'_zeta',strrep(num2str(zeta),'.','_')),'All_Fisher_info')
    end
end