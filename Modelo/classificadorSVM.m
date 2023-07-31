clc
clear
close all

%% -------------------- Carregando Sinal ----------------------------------
% os audios usados estão na pasta nomeados de um.mp3, dois.mp3 e tres.mp3
% o pré processamento dos audios foi feito
% em outro script e consistiu em segmentar o audio pegando as amostras no
% das palavras um, dois e tres. No fim, as amostras foram salvas no arquivo
% sinais.dat.
load sinais.dat
load rotulos.dat
sound(sinais(1,:),48000);
% sound(sinais(40,:),48000);
% sound(sinais(100,:),48000);

%% -------------------- Extração de atributos -----------------------------

tamanho_amostras = size(sinais);
quant_features = 11;

dataset = zeros(tamanho_amostras(1),quant_features);

for i = 1:tamanho_amostras(1)
    % Atributo média da densidade spectral
    dataset(i,1) =  mode(pwelch(sinais(i,:)));
    % Atributo média do módulo da transformada de fourier
    dataset(i,2) =  mode(abs(fft(sinais(i,:))));
    % Atributo assimetria assimetria
    dataset(i,3) =  skewness(sinais(i,:));
    % Atributo curtose curtose
    dataset(i,4) =  kurtosis(sinais(i,:));
    % Atributo entropia entropia
    dataset(i,5) =  entropy(sinais(i,:));
    % Atributo desvio padrão desvio padrao
    dataset(i,6) =  std(sinais(i,:));
    % Atributo amplitude 
    dataset(i,7) =  max(sinais(i,:)) - min(sinais(i,:));
    % Atributo Energia do sinal
    dataset(i,8) = sum(abs(sinais(i,:)).^2);
    % Atributo Taxa de variação do espectro
    dataset(i,9) = mean(abs(diff(abs(fft(sinais(i,:))))));
    % Atributo RMS
    dataset(i,10) = rms(sinais(i,:));
    % Atributo numero cepstral
    dataset(i,11) = mode(rceps(sinais(i,:)));
end

% Normalizando a base de dados
dataset = normalizando_dataset(dataset);


clear i quant_features tamanho_amostras


%% -------------------- Grid Search ---------------------------------------

parametros = {{'KernelFunction','linear'},{'KernelFunction', 'rbf'},{'KernelFunction','polynomial', 'PolynomialOrder', 1},{'KernelFunction','polynomial', 'PolynomialOrder', 2},0,0;
               {'BoxConstraint', 10e-3},{'BoxConstraint', 10e-2},{'BoxConstraint', 10e-1},{'BoxConstraint', 1},{'BoxConstraint', 10},{'BoxConstraint', 100};
               {'KernelScale', 10e-3},{'KernelScale', 10e-2},{'KernelScale', 10e-1},{'KernelScale', 1},{'KernelScale', 10},{'KernelScale', 100}};


%% -------------------- Teste e validação ---------------------------------

% Numero de folds
k = 3;
% dividindo a base de dados em treino e teste por meio do indices do k-fold
[index_teste,index_treino] = K_fold(dataset, k);
media_acuracia = zeros(k,1);

for i=1:1:k
    % Pegando os dados de treino e teste dos k-folds
    [dataset_teste, rotulos_teste] = fold_rodada(index_teste(i,:),dataset,rotulos);
    [dataset_treino, rotulos_treino] = fold_rodada(index_treino(i,:),dataset,rotulos); 
    
    % Aplicar o grid_search para encontrar os melhores parametros.
    [model_parametro,~] = Grid_search(parametros,dataset_treino, dataset_teste,rotulos_treino,rotulos_teste);
    
    % Treinando o modelo SVM com os dados 
    model = SVM_multiclass(dataset_treino, rotulos_treino, model_parametro);

    media_acuracia(i,1)  = Acuracia_SVM(model,rotulos_teste,dataset_teste);
end

disp("Acurácia do classificador SVM: " + mean(media_acuracia));


clear i k index_treino index_teste dataset_teste rotulos_teste dataset_treino rotulos_treino
clear media_acuracia model_parametro
%% -------------------- Funções implementadas -----------------------------

% Função que normaliza os dados
function dados_normalizados = normalizando_dataset(features)
    % Função que normaliza os atributos usando o metodo zscore, assim os
    % atributos devem ter média zero e desvio padrão um.
    [n_amostras, n_atributos] = size(features);
    dados_normalizados = zeros(n_amostras,n_atributos);
    for i=1:1:n_atributos
        dados_normalizados(:,i) = zscore(features(:,i));
    end
end

% Função Grid search que escolhe os melhores hiperparametros do modelo SVM
function [model,acuracia_grid] = Grid_search(parametros,dados_treinamento, dados_teste,rotulos_treinamento,rotulos_teste)
    % Cria um vetor com todos os arranjos possíveis dos parâmetros
    [arr_1, arr_2, arr_3] = ndgrid(parametros(1,1:end-2),parametros(2,:),parametros(3,:));
    arranjo_parametros = [arr_1(:) arr_2(:) arr_3(:)];
    
    % Tamanho das combinações dos parâmetros
    tam_arranjo = size(arranjo_parametros);
    % Define a matriz dos rótulos votados pelos modelos 1vs1
    rotulos_votados = zeros(length(rotulos_teste),length(unique(rotulos_treinamento)));
    % Define os vetores dos rótulos  preditos e a acurácia do modelo
    rotulos_predito = zeros(length(rotulos_teste),1);
    taxa_acuracia = zeros(length(rotulos_teste),1);
    
    % Analisa a acurácia de cada arranjo dos parâmetros
    for j=1:1:tam_arranjo(1)
        % Cria os modelos one vs one para o arranjo de parâmatro j
        modelo = SVM_multiclass(dados_treinamento, rotulos_treinamento ,arranjo_parametros(j,:));
        
        % Calcula os votos preditos dos modelos 1vs1
        for i=1:1:length(modelo)
            rotulos_votados(:,i) = predict(modelo{i}, dados_teste) ;
        end
        
        % Método para compoutar a decisão dos votos dos modelos
        for k=1:1:length(rotulos_votados) 
            % Todos os modelos votarem em diferentes classes (Caso de empate) 
            if all(size(unique(rotulos_votados(k,:))) == size(rotulos_votados(k,:)))
                % Escolhe de forma aleatória a classe
                indice = randi(numel(rotulos_votados(k,:)));
                rotulos_predito(k,1) = rotulos_votados(k,indice);
            else
                % caso contrário é escolhido a classes mais votadas
                rotulos_predito(k,1) = mode(rotulos_votados(k,:));
            end
        end
        % Calcula a acurácia do modelo com os arranjos de parâmetros j
        taxa_acuracia(j,1) = acuracia(rotulos_teste, rotulos_predito);
    end
    % Pega o modelo e os parãmetros que resultaram em uma maior acurácia do modelo
    [acuracia_grid, indice] = max(taxa_acuracia);

    model = arranjo_parametros(indice,:);

end

% Função que calcula a acurácia do modelo SVM
function taxa_acuracia = Acuracia_SVM(modelo,rotulos_teste,dados_teste)
    % Define a matriz dos rótulos votados pelos modelos 1vs1
    rotulos_votados = zeros(length(rotulos_teste),length(modelo));
    % Define o vetor dos rótulos  preditos
    rotulos_predito = zeros(length(rotulos_teste),1);
    
    % Calcula os votos preditos dos modelos 1vs1
    for i=1:1:length(modelo)
        rotulos_votados(:,i) = predict(modelo{i}, dados_teste) ;
    end

    % Método para compoutar a decisão dos votos dos modelos
    for k=1:1:length(rotulos_votados) 
        % Todos os modelos votarem em diferentes classes (Caso de empate) 
        if all(size(unique(rotulos_votados(k,:))) == size(rotulos_votados(k,:)))
                % Escolhe de forma aleatória a classe
                indice = randi(numel(rotulos_votados(k,:)));
                rotulos_predito(k,1) = rotulos_votados(k,indice);
        else
                % Caso contrário é escolhido a classes mais votadas
                rotulos_predito(k,1) = mode(rotulos_votados(k,:));
        end
    end
    % Retorna a acurácia do modelo
    taxa_acuracia = acuracia(rotulos_teste, rotulos_predito);

end

% Função que cria o Modelo SVM multiclasses
function model = SVM_multiclass(dataset, rotulos,parametros)
    % Cria as combinações possíveis de infretamento entre as classes
    combinacao_classes = nchoosek(unique(rotulos), 2);
    % Cria os modelos multiclasses
    for i=1:1:length(combinacao_classes)
        [dataset_novo,rotulos_novo] = Dataset_Dividido(dataset,rotulos,combinacao_classes(i,:));
        model{i} = fitcsvm(dataset_novo, rotulos_novo,parametros{1}{:},parametros{2}{:},parametros{3}{:});
        dataset_novo = [];
        rotulos_novo = [];
    end

end

% Função que transforma a a base multiclasse em uma base de classe binária
function [dataset_novo,rotulos_novo] = Dataset_Dividido(dataset,rotulos,classes)
    % Variável auxilar para o indice
    index = 1;
    % Divide a base multiclasse em uma base de classe binária
    for i=1:1:length(rotulos)
        if(rotulos(i,1)==classes(1,1) || rotulos(i,1)==classes(1,2))
            dataset_novo(index,:) = dataset(i,:);
            rotulos_novo(index,1) = rotulos(i,1);
            index = index + 1;
        end
    end
end


% Função que calcula a Porcetagem de acertos usando acuracia
function taxa_acerto = acuracia(rotulos_reais,rotulos_preditas)
    % Tanto rotulos_reais quanto rotulos_preditas são vetores onde os
    % que contem as classes definidas nas linhas e uma unica coluna.
    n_amostras = length(rotulos_reais);
    acertos = 0;

    for i=1:1:n_amostras
        if rotulos_reais(i,1) == rotulos_preditas(i,1)
            acertos = acertos + 1;
        end
    end
    taxa_acerto = (acertos/n_amostras)*100;
end

% Função que dividide a base de dados entre treino e validação usando o metodo k-fold
function [teste, treino] = K_fold(data_set,K)
    % O método de cruzamento k-fold consiste em dividir o conjunto total
    % de dados em k subconjuntos mutualmente exclusivos do mesmo tamanho.
    
    % Para implementar o método, a ideia é criar um vetor que indique através 
    % dos indices quem vai ser os objetos de treino e quem vai ser os objetos 
    % de teste, dessa forma possibilitar dividir a base de dados entre
    % treino e teste.
    
    % A função K_fold retorna os indices de treino e os indices de teste.

    N_objetos = size(data_set); %Descobre quem é a quantidade de atributos 'M' e a quantidade de objetos 'N' 
    resto = mod(N_objetos,K);
    dataset_indx = randperm(N_objetos(1)); %cria um vetor aleatório com a permutação 1 atea quantidade de objetos sem repeti-los.
     if resto(1) ~= 0
        for i=1:1:resto
           dataset_indx(:,i) = [];
        end
     end
    quant_grupo = floor(N_objetos(1)/K); %Define a quantidade de elementos em cada subconjunto K-fold, arredondado 
    vetor_index_teste = zeros(K,quant_grupo);
    vetor_index_treino = zeros(K,length(dataset_indx)-quant_grupo);
    ind_inicial = 1; %variavel auxiliar que aponta para o primeiro indice do subconjunto k-fold
    ind_final = quant_grupo; %variavel auxilar que aponta para o ultimo indice do subconjunto k-fold

    for i=1:K

        %Constroi o vetor de indice para os testes
        vetor_index_teste(i,:) = dataset_indx(1,ind_inicial:ind_final);
        
        %Constroi o vetor de indices para os treinos
        if i == 1
            vetor_index_treino(i,:) = dataset_indx(1,(ind_final+1):end);
        elseif i == K
            vetor_index_treino(i,:) = dataset_indx(1,1:(ind_inicial-1));
        else
            vetor_index_treino(i,:) = [dataset_indx(1,1:(ind_inicial-1)), dataset_indx(1,(ind_final+1):end)];
        end
        ind_inicial = ind_final+1; %Indice inicial recebe o ultimo indice mais 1
        ind_final = ind_final+ quant_grupo; %Indice final recebe o final mais a quantidade de elementos do k-fold, já que o periodo se repete em cada k-fold
    end
    teste = vetor_index_teste;
    treino = vetor_index_treino;
end

% Função que seleciona a base de dados
function [atributos, classes]= fold_rodada(indices,features,rotulos)
    % A função dataset_rodada() retorna os atributos da base de dados conforme 
    % os index definidos pelo método k-fold. 
    % A função retorna um vetor contendo os atributos nas colunas e as 
    % amostras nas linhas.

    n = length(indices);
    tamanho_features = size(features);
    features_selecionados = zeros(n,tamanho_features(2)); %Cria a base de dados dos atributos selecionados
    rotulos_selecionados = zeros(n,1); %Cria a base de dados dos rotulos selecionados
    %Preenche os dados de acordo com a divisão de subgrupos
    for i=1:n
        features_selecionados(i,:) = features(indices(1,i),:);
        rotulos_selecionados(i,1) = rotulos(indices(1,i),1);
    end
    atributos = features_selecionados;
    classes = rotulos_selecionados;
end


