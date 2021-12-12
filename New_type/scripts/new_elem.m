clc;
set(0,'DefaultTextInterpreter', 'latex');
set(0,'DefaultAxesFontSize', 24);
set(0,'DefaultTextFontSize', 24);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Read custom settings  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Считываем настроичные данные
set_file = fopen('../Output Data/Data for Matlab/settings_matlab.txt', 'r');
count_iter = fread(set_file, 1, 'int') + 1;      %The Number of iterations
num_ch_new = fread(set_file, 1, 'int');          %Number of Poisson Process Triggers
count_solve_step = fread(set_file, 1, 'int');    %The number of iterations, which decided ODE
sizeA_finish = fread(set_file, 1, 'int');        %Number of elements at the last moment
count_step = fread(set_file, 1, 'int') + 1;      %The Number of points in the grid to solve ODE
fclose(set_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% The results of the Poisson process %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
poisson_file = fopen('../Output Data/Data for Matlab/poisson_matlab.txt', 'r');
poisson_val_new = [];
poisson_iter_new = [];
poisson_iter = [];
poisson_val = [];
poisson_all = zeros(count_iter, 1);

for i = 1 : count_iter
    buf = fread(poisson_file, 1, 'int');
    poisson_all(i, 1) = buf;
    
    if buf == 1
        poisson_val_new = [poisson_val_new buf];
        poisson_iter_new = [poisson_iter_new i - 1];
    end
    
    if buf > 0 
        poisson_val = [poisson_val buf];
        poisson_iter = [poisson_iter i - 1];
    end
end
fclose(poisson_file);



num_view_file = fopen('../Output Data/Data for Matlab/num_view_matlab.txt', 'r');
num_view = zeros(num_ch_new, 1);

for i = 1 : num_ch_new
    num_view(i, 1) = fread(num_view_file, 1, 'int');
end



figure();
hold on;

if size(poisson_val_new, 2) > 0
    plot(poisson_iter_new, poisson_val_new, 'gs', 'LineWidth', 3);
    l = 'Adding a new view';
end

xlim([0 1.01 * count_iter]);
ylim([0 3]);
xlabel('$\tau$','FontSize', 32);
ylabel('Poisson process', 'FontSize', 32);
title('The results of the Poisson process');

if size(poisson_val_new, 2) > 0
    legend(l);
end

beta_file = fopen('../Output Data/Data for Matlab/beta_matlab.txt', 'r');
beta = zeros(1, size(poisson_iter_new, 2));
j = 1;
for i = 1 : count_iter
    bbeta = fread(beta_file, 1, 'double');
    if bbeta > 0
        beta(1, j) = round(bbeta, 3);
        j = j + 1;
    end
end
fclose(beta_file);

j = 1;
for i = 1 : num_ch_new
    
    if isempty(find(poisson_iter_new == poisson_iter(1, i))) == 0
        l = ['(', num2str(poisson_iter(1, i)), ', ', num2str(num_view(i, 1)), ', ', num2str(beta(1, j)), ')'];
        j = j + 1;
    else
        l = ['(', num2str(poisson_iter(1, i)), ', ', num2str(num_view(i, 1)), ')']; 
    end 
    
    text(poisson_iter(1, i), poisson_val(1, i) + 0.1, l, 'Rotation', 90, 'FontSize', 15)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% fitness %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fitness_file = fopen('../Output Data/Data for Matlab/fitness_matlab.txt', 'r');
fitness = zeros(count_iter, 1);
for i = 1 : count_iter
    fitness(i, 1) = fread(fitness_file, 1, 'double'); 
end
fclose(fitness_file);

time = 0 : (count_iter - 1);

figure();
hold on;
plot(time, fitness, 'k--', 'LineWidth', 4.5);

xlabel('$\tau$','FontSize', 32);
ylabel('$\bar f$', 'FontSize', 32);
title('Fitness');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fixed point %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
count_view_file = fopen('../Output Data/Data for Matlab/count_view_matlab.txt', 'r');
count_view = zeros(count_iter, 1);
for i = 1 : count_iter
    count_view(i, 1) = fread(count_view_file, 1, 'int'); 
end
fclose(count_view_file);

freq = zeros(sizeA_finish, count_iter);
time_vec = zeros(sizeA_finish, count_iter);
freq_file = fopen('../Output Data/Data for Matlab/freqType_matlab.txt', 'r');
for i = 1 : count_iter
    for j = 1 : count_view(i, 1)
        freq(j, i) = fread(freq_file, 1, 'double');
        time_vec(j, i) = i;
    end
    
    for k = j + 1 : sizeA_finish
        freq(k, i) = -1; 
        time_vec(k, i) = -1;
    end
end
fclose(freq_file);

figure();
hold on;

for i = 1 : sizeA_finish
    freq_ = [];
    time_vec_ = [];
    for j = 1 : count_iter
        if freq(i, j) > -1
            freq_ = [freq_ freq(i, j)];
            time_vec_ = [time_vec_ time_vec(i, j)];
        end
    end
    plot(time_vec_, freq_, 'LineWidth', 3);
end
xlabel('$\tau$','FontSize', 32);
ylabel('$\bar u$', 'FontSize', 32);
title('Fixed point');

leg = 'u1  ';
for i = 2 : sizeA_finish
    if i < 10
        leg = [leg; strcat('u', num2str(i), 32, 32)];
    else if 10 <= i & i < 100
            leg = [leg; strcat('u', num2str(i), 32)];
        else
            leg = [leg; strcat('u', num2str(i))];
        end
    end
end
legend(leg);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
solve_odu_file = fopen('../Output Data/Data for Matlab/solve_odu_matlab.txt','r');
solve_odu = zeros(count_iter, 1);

time2 = [];
for i = 1 : count_iter
    a = fread(solve_odu_file, 1, 'int');
    solve_odu(i, 1) = a;
end
fclose(solve_odu_file);

freq_cont_file = fopen('../Output Data/Data for Matlab/freqType_continuos_matlab.txt', 'r');
freq_cell = {};

cnt = 1;
for i = 1 : count_iter
    if solve_odu(i, 1) == 1
        freq = zeros(count_view(i, 1), count_step);
        for j = 1 : count_step
            for k = 1 : count_view(i, 1)
               freq(k, j) = fread(freq_cont_file, 1, 'double'); 
            end
        end
        freq_cell{cnt} = freq;
        cnt = cnt + 1;
        freq = [];
    end
end
fclose(freq_cont_file);

time_file = fopen('../Output Data/Data for Matlab/time_matlab.txt', 'r');
time_cont = zeros(count_step, 1);
for i = 1 : count_step
    time_cont(i, 1) = fread(time_file, 1, 'double');
end    
fclose(time_file);

for i = 2 : count_iter
    if poisson_all(i, 1) > 0
        poisson_all(i - 1, 1) = 1;
    end
end

cnt = 1;
for i = 1 : count_iter
    if solve_odu(i, 1) == 1 
            freq = freq_cell{cnt};
            figure();
            hold on;
            plot(time_cont, freq, 'LineWidth', 3);
            xlabel('t','FontSize', 32);
            ylabel('$\bar u$', 'FontSize', 32);
            l = ['ODE solution on ', num2str(i - 1), ' iteration'];
            title(l);
            
            leg = 'u1  ';
            for j = 2 : count_view(i, 1)
                if j < 10
                    leg = [leg; strcat('u', num2str(j), 32, 32)];
                else if 10 <= j & j < 100
                        leg = [leg; strcat('u', num2str(j), 32)];
                    else
                        leg = [leg; strcat('u', num2str(j))];
                    end
                end
            end
            legend(leg);
        cnt = cnt + 1;
    end
end




