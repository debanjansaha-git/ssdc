function jmo(Runs,fhd,problem_size,funcs,max_nfes,pop_size)

% Rand_Seeds=load('input_data/Rand_Seeds.txt');
rand('seed', sum(100 * clock));
Alg_Name='jMO_';

F =0.50*ones(pop_size,problem_size);
Cr=0.90*ones(pop_size,problem_size);

lu = [-100 * ones(1, problem_size); 100 * ones(1, problem_size)];
GMax = 0;
switch problem_size
    case 10
        GMax = 2163;
    case 30
        GMax = 2745;
    case 50
        GMax = 3022;
    case 100
        GMax = 3401;
end

fprintf('Running %s algorithm on D= %d\n',Alg_Name, problem_size)

% for n=0:15
%     RecordFEsFactor(n+1) = round(problem_size^((n/5)-3)*max_nfes);
% end
RecordFEsFactor = ...
    [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, ...
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

progress = numel(RecordFEsFactor);
val_2_reach = 10^(-8);

for func_no = funcs
    optimum=100*func_no;
    fprintf('\n-------------------------------------------------------\n')
    fprintf('Function = %d, Dimension size = %d\n', func_no, problem_size)
    allerrorvals = zeros(progress, Runs);
%     you can use parfor if you have MATLAB Parallel Computing Toolbox
%     parfor run_id = 1 : Runs
    for run_id = 1 : Runs
        g_iter=0;
        
        rand_ind = problem_size*func_no*Runs+run_id-Runs;
%         run_seed=Rand_Seeds(max(1, rand_ind));
        rng(rand_ind,'twister');
%         Run_RecordFEsFactor=RecordFEsFactor;
        run_funcvals = [];
        
        %%  parameter settings for L-SHADE
        p_best_rate = 0.11;
        arc_rate = 1.4;
        memory_size = 5;
        pop_size = 18 * problem_size;
        
        max_pop_size = pop_size;
        min_pop_size = 4.0;
        
        %% parameters introduced in iLSHADE
%         pmin = 0.1; pmax = 0.2;
        % parameters modified in jSO
        pmax = 0.25; pmin = pmax/2;
        
        %% Initialize the main population
        popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
        pop = popold; % the old population becomes the current population
        
        fitness = feval(fhd, pop', func_no);
        fitness = fitness';
        
        g_iter = g_iter + 1;
        nfes = 0;
        bsf_fit_var = 1e+30;
        bsf_solution = zeros(1, problem_size);
        
        %%%%%%%%%%%%%%%%%%%%%%%% for out
        for i = 1 : pop_size
            nfes = nfes + 1;
            
            if fitness(i) < bsf_fit_var
                bsf_fit_var = fitness(i);
                bsf_solution = pop(i, :);
            end
            
            if nfes > max_nfes
                break;
            end
        end
        
%         if(nfes>=Run_RecordFEsFactor(1))
%             run_funcvals = [run_funcvals; bsf_fit_var];
%             Run_RecordFEsFactor(1)=[];
%         end
        run_funcvals = [run_funcvals;ones(pop_size,1)*bsf_fit_var];
        
        %% ===== prob. of each DE operatorr =
        num_de = 2;
        count_S = zeros(1, num_de);
        probDE=1./num_de .* ones(1, num_de);
        
        memory_sf = 0.3 .* ones(memory_size, 1);  %% jSO changes to 0.3 from 0.5
        memory_cr = 0.8 .* ones(memory_size, 1);  %% iLSHADE changes to 0.8 from 0.5
        memory_pos = 1;

        archive.NP = round(arc_rate * pop_size); % the maximum size of the archive
        archive.pop = zeros(0, problem_size); % the solutions stored in te archive
        archive.funvalues = zeros(0, 1); % the function value of the archived solutions

        %% main loop
        while nfes < max_nfes
            pop = popold; % the old population becomes the current population
            [temp_fit, sorted_index] = sort(fitness, 'ascend');
            pop=pop(sorted_index,:);
            
            %% iLSHADE Changes
            rand_indx = randi(memory_size,1);
            if rand_indx == memory_size
                memory_sf(rand_indx) = 0.9;
                memory_cr(rand_indx) = 0.9;
            end
            
            %% LSHADE select crossover
            mem_rand_index = ceil(memory_size * rand(pop_size, 1));
            mu_sf = memory_sf(mem_rand_index);
            mu_cr = memory_cr(mem_rand_index);
            
            %% for generating crossover rate
            cr = normrnd(mu_cr, 0.1);
            
            %% iLSHADE Changes
            if g_iter < 0.25 * GMax
                cr = max(cr,0.7);
            elseif g_iter < 0.5 * GMax
                cr = max(cr,0.6);
            end  
            
            term_pos = find(mu_cr == -1);
            cr(term_pos) = 0;
            cr = min(cr, 1);
            cr = max(cr, 0);
            
            %% for generating scaling factor (Cauchy Distribution)
            sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
            pos = find(sf <= 0);
            
            while ~ isempty(pos)
                sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                pos = find(sf <= 0);
            end
            
            sf = min(sf, 1);
%             disp(sf)
            
%             %% iLSHADE Changes
%             if g_iter < 0.25 * GMax
%                 sf = min(sf,0.7);
%             elseif g_iter < 0.5 * GMax
%                 sf = min(sf,0.8);
%             elseif g_iter < 0.75 * GMax
%                 sf = min(sf,0.9)
%             end
            
            %% jSO Changes 
            if (g_iter < 0.6 * GMax)
                sf = min(sf,0.7);
            end

            r0 = 1 : pop_size;
            popAll = [pop; archive.pop];
            [r1, r2] = gnR1R2(pop_size, size(popAll, 1), r0);
            
            %% Mutation
            % p_best rate modification introduced in iLSHADE
            p_best_rate = ((pmax - pmin)/max_nfes)*nfes + pmin; 
            
            bb = rand(pop_size, 1);
            probiter = probDE(1,:);
            de_1 = bb <= probiter(1)*ones(pop_size, 1);
            de_2 = bb > probiter(1)*ones(pop_size, 1) & bb <= (ones(pop_size, 1));
            
            pNP = max(round(p_best_rate * pop_size), 2); %% choose at least two best solutions
            randindex = ceil(rand(1, pop_size) .* pNP); %% select from [1, 2, 3, ..., pNP]
            randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
            pbest = pop(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions
            
            % jSO weighted mutation
            if nfes < 0.2 * max_nfes
                sfw = 0.7 * sf;
            elseif nfes < 0.4 * max_nfes
                sfw = 0.8 * sf;
            else
                sfw = 1.2 * sf;
            end
            
            vi = pop;
            % DE/current-to-p-best/1
            temp = pop + sfw(:, ones(1, problem_size)) .* (pbest - pop(r1, :)) + ...
                sfw(:, ones(1, problem_size)).* (pop(r1, :) - popAll(r2, :));
            vi(de_1==1, :) = temp(de_1==1, :);
            % DE/rand-to-p-best/1
            temp = pop(r1, :) + sfw(:, ones(1, problem_size)) .* ((pbest - popAll(r2, :)));
            vi(de_2==1, :) = temp(de_2==1, :);
            
            vi = boundConstraint(vi, pop, lu);
            
            %% Cross-Over / Recombination
            mask = rand(pop_size, problem_size) > cr(:, ones(1, problem_size)); % mask is used to indicate which elements of ui comes from the parent
            rows = (1 : pop_size)'; cols = floor(rand(pop_size, 1) * problem_size)+1; % choose one position where the element of ui doesn't come from the parent
            jrand = sub2ind([pop_size problem_size], rows, cols); mask(jrand) = false;
            ui = vi; ui(mask) = pop(mask);
            
            %% Selection
            children_fitness = feval(fhd, ui', func_no);
            children_fitness = children_fitness';
            
            for i = 1 : pop_size
                nfes = nfes + 1;
                
                if children_fitness(i) < bsf_fit_var
                    bsf_fit_var = children_fitness(i);
                end
                
                if nfes > max_nfes
                    break;
                end
            end
            
            %% Update Results and generation
%             if(nfes>=Run_RecordFEsFactor(1))
%                 run_funcvals = [run_funcvals; bsf_fit_var];
%                 Run_RecordFEsFactor(1)=[];
%             end
            run_funcvals = [run_funcvals;ones(pop_size,1)*bsf_fit_var];
                
            g_iter = g_iter + 1;
            %%%%%%%%%%%%%%%%%%%%%%%% for out
            
            dif = abs(fitness - children_fitness);
            
            %% I == 1: the parent is better; I == 0: the offpring is better
            I = (fitness > children_fitness);
            goodCR = cr(I == 1);
            goodF = sf(I == 1);
            dif_val = dif(I == 1);
               
            archive = updateArchive(archive, popold(I == 1, :), fitness(I == 1));            
            %% ==================== update Prob. of each DE ===========================
            diff2 = max(0,(fitness - children_fitness))./abs(fitness);
            count_S(1)=max(0,mean(diff2(de_1==1)));
            count_S(2)=max(0,mean(diff2(de_2==1)));
            
            %% update probs.
            if count_S~=0
                  prob = max(0.1,min(0.9,count_S./(sum(count_S))));
            else
                  prob =1.0/2 * ones(1,2);
            end
            %% ==================== update population and fitness =======================
            [fitness, I] = min([fitness, children_fitness], [], 2);
            
            popold = pop;
            popold(I == 2, :) = ui(I == 2, :);
            
            num_success_params = numel(goodCR);
            
            if num_success_params > 0
                sum_dif = sum(dif_val);
                dif_val = dif_val / sum_dif;  
                
                %% for updating the memory of scaling factor 
                memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
                
                %% for updating the memory of crossover rate
                if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
                    memory_cr(memory_pos)  = -1;
                else
                    memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
                end
                
                memory_pos = memory_pos + 1;
                
                if memory_pos > memory_size
                    memory_pos = 1;
                end
            end
            
            %% for resizing the population size
            plan_pop_size = round((((min_pop_size - max_pop_size) / max_nfes) * nfes) + max_pop_size);
            
            if pop_size > plan_pop_size
                reduction_ind_num = pop_size - plan_pop_size;
                if pop_size - reduction_ind_num <  min_pop_size
                    reduction_ind_num = pop_size - min_pop_size;
                end
                
                pop_size = pop_size - reduction_ind_num;
                
                for r = 1 : reduction_ind_num
                    [valBest, indBest] = sort(fitness, 'ascend');
                    worst_ind = indBest(end);
                    popold(worst_ind,:) = [];
                    pop(worst_ind,:) = [];
                    fitness(worst_ind,:) = [];
                end
                
                archive.NP = round(arc_rate * pop_size);
                
                if size(archive.pop, 1) > archive.NP
                    rndpos = randperm(size(archive.pop, 1));
                    rndpos = rndpos(1 : archive.NP);
                    archive.pop = archive.pop(rndpos, :);       
                    archive.funvalues = archive.funvalues(rndpos, :); 
                end
            end
        end
        
        fprintf('%d th run, best-so-far error value = %1.8e\n', run_id , run_funcvals(end)-optimum)
        
        errorVals= [];
        for w = 1 : progress
            bestold = run_funcvals(RecordFEsFactor(w) * max_nfes) - optimum;
            errorVals(w)= abs(bestold);
        end
        allerrorvals(:, run_id) = errorVals;
        
    end %% end 1 run
    
    allerrorvals(allerrorvals < val_2_reach) = 0;
    
    [~, sorted_index] = sort(allerrorvals(end,:), 'ascend');
    allerrorvals = allerrorvals(:, sorted_index);
    
    fprintf('min_funvals:\t%e\n',min(allerrorvals(end,:)));
    fprintf('median_funvals:\t%e\n',median(allerrorvals(end,:)));
    fprintf('mean_funvals:\t%e\n',mean(allerrorvals(end,:)));
    fprintf('max_funvals:\t%e\n',max(allerrorvals(end,:)));
    fprintf('std_funvals:\t%e\n',std(allerrorvals(end,:)));
    
    res_file = [min(allerrorvals(end,:)), max(allerrorvals(end,:)), ...
        median(allerrorvals(end,:)), mean(allerrorvals(end,:)), std(allerrorvals(end,:))];
    
    file_name=sprintf('Results/%s_%s_%s.txt',Alg_Name,int2str(func_no),int2str(problem_size));
    save(file_name, 'allerrorvals', '-ascii');
    file_name2=sprintf('TableRes/%s_%s_%s.txt',Alg_Name,int2str(func_no),int2str(problem_size));
    save(file_name2, 'res_file', '-ascii');
%     fprintf("\n\n********** Total Generations from one Algorithm Run: %s ************\n\n",num2str(GMax));
end %% end 1 function run

end