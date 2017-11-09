% Fonction calculant l'estimateur LAR
% En entree : y = vecteur de taille R_n ; X = matrice de taille n*P ; c = constante ou vecteur ;
% u = vecteur de taille R_(n-p) (z,u) = forme canonique de y = X*b + e.
% Renvoie un vecteur de meme taille que z ou une matrice de taille
% length(z)*length(c).
%
% Si u est précisé, on suppose la variance inconnue.


% ORTHOGONAL CASE : A = Ip 

function [beta_LAR,beta_LSI,mu] = estimLAR(X, y)

    [n,P] = size(X) ; 
    beta_LS = X\y ;

    mu = [sort(abs(beta_LS),'descend');0]' ;
    Z = repmat(beta_LS,1,P+1);     % idem en colonne : Z et C sont de meme taille    

    % Calcul du lasso sous forme canonique pour une matrice de design orthogonale
    beta_LAR = sign(Z).*max(abs(Z)-repmat(mu,P,1),0);
        
%        keyboard 
    beta_LSI = zeros(P,P+1);
    for j=2:1:P+1
        beta_LSI((beta_LAR(:,j)~=0),j) = beta_LS(beta_LAR(:,j)~=0) ;
    end

        
end




% function [beta_LAR] = estimLAR(X, y)
% 
% [z, u] = canonical(X, y) ;
% P = length(z) ; 
% 
% switch known_var
%    case 1
%         C = repmat([sort(abs(z),'descend');0]',P,1); % Matrice de replication en lignes pour la soustraction
%         Z = repmat(z,1,P+1);     % idem en colonne : Z et C sont de meme taille    
% 
%     % Calcul du lasso sous forme canonique pour une matrice de design orthogonale
%         phi_LAR = sign(Z).*max(abs(Z)-C,0);
%         beta_LAR= A\phi_LAR;
%         
%     case 0
%         S = sum(u.^2) ;
%         C = repmat([sort(abs(z)/S,'descend');0]',P,1); % Matrice de replication en lignes pour la soustraction
%         Z = repmat(z,1,P+1);     % idem en colonne : Z et C sont de meme taille    
% 
%     % Calcul du lasso sous forme canonique pour une matrice de design orthogonale
%         phi_LAR = sign(Z).*max(abs(Z)-C*S,0);
%         beta_LAR= A\phi_LAR;
% 
% end
%         
% end




% % Fonction calculant l'estimateur LAR
% % En entree : z = vecteur de taille R_p ; c = constante ou vecteur ;
% % u = vecteur de taille R_(n-p) (z,u) = forme canonique de y = X*b + e.
% % Renvoie un vecteur de meme taille que z ou une matrice de taille
% % length(z)*length(c).
% %
% % Si u est précisé, on suppose la variance inconnue.
% 
%function [phi_LAR] = estimLAR(z,c,u)
%         C=repmat(c(:)',length(z),1); % Matrice de replication en lignes pour la soustraction
%         Z=repmat(z,1,length(c));     % idem en colonne : Z et C sont de meme taille    
% 
%     % Calcul du lasso sous forme canonique pour une matrice de design orthogonale
%         phi_LAR=sign(Z).*max(abs(Z)-C,0);
% 
% %        phi_LAR = (phi_LAR ~= 0).*Z ; % Subset selection
% end