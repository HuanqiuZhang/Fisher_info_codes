%This code runs the simulation shown in Fig.4 of the paper

%simulation parameters
for zeta = 1./[1:20]  %note this is actually zeta inverse
    for N = [50 100 200 400 800 1600 3200 6400 12800]
        dim = 2;
        rep = 10000;
        All_Fisher_info = zeros(dim,dim,rep);
        summary = [];
        for j = 1:rep
            %sample neuron parameters
            A = 20*rand(N,1)+5;
            mu = rand(N,dim);
            %exponential distribution of sigma
            sigma = exprnd(zeta,N,dim);
%             %uniform distribution of sigma - same mean as exp distribution
%             sigma = 2*zeta*rand(N,dim);
%             %lognormal distribution of sigma - same mean and var as exp distribution
%             m = zeta; % mean
%             v = zeta^2; % variance
%             mu_lognormal = log((m^2)/sqrt(v+m^2));
%             sigma_lognormal = sqrt(log(v/(m^2)+1));
%             sigma = lognrnd(mu_lognormal,sigma_lognormal,[N,dim]);
            
            if dim == 2  %randomly rotate the cov matrix
                theta = 2*pi*rand(N,1);
                Sigma = zeros(2,2,N);
                for i = 1:N
                    R = [cos(theta(i)) -sin(theta(i));sin(theta(i)) cos(theta(i))];
                    Sigma(:,:,i) = R*diag(sigma(i,:).^2)*R';   %covariance matrix
                end
                %information matrix
                Omega = zeros(2,2,N);
                for i = 1:N
                    Omega(:,:,i) = inv(Sigma(:,:,i));
                end
            end
            
            %compute Fisher information matrix
            stim = rand(1,dim);
            Fisher_info = zeros(dim,dim);
            for i = 1:N
                [I11,I12,I22] = Fisher_single_neuron(A(i),mu(i,:),Omega(:,:,i),stim);
                Fisher_info(1,1) = Fisher_info(1,1) + I11;
                Fisher_info(1,2) = Fisher_info(1,2) + I12;
                Fisher_info(2,1) = Fisher_info(2,1) + I12;
                Fisher_info(2,2) = Fisher_info(2,2) + I22;
            end
            All_Fisher_info(:,:,j) = Fisher_info;
        end
        save(strcat('FishInfo_exp_N',num2str(N),'_zeta',strrep(num2str(zeta),'.','_')),'All_Fisher_info')
    end
end