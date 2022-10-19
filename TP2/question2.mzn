% Bathylle de la Grandière - 2166208
% Augustin Barruol - 2161214

include "globals.mzn"; 

int: n = 5;
% the given results
array[1..4] of int: res = [1,2,5,7];

% number of points for each player
array[1..n] of var 0..3*n: x;

% (i, j) is the result of player i against player j
array[1..n, 1..n] of var {0,1,3}: matches;

% Contraintes
constraint forall(k in 1..n)(x[k] == sum(i in 1..n)(matches[k,i])); % On donne le score de chaque joueur
constraint sum(k in 1..n)(matches[k,k]) == 0; % Les valeurs sur la diagonale doivent être nulles
constraint forall(k in 1..4)(sum(i in 1..n)(matches[k,i]) == res[k]); % somme de la ligne i de matches == ligne i de res (pour i allant de 1 à 4)
constraint forall(i in 1..n)(forall(j in 1..n)(if matches[i,j] == 0 /\ i != j then matches[j,i] = 3 endif)); % Contrainte de symétrie : pour i différent de j, si i gagne contre j, alors forcément j a perdu contre i
constraint forall(i in 1..n)(forall(j in 1..n)(if matches[i,j] == 1 then matches[j,i] = 1 endif)); % Contrainte de symétrie : pour i différent de j, si i a fait une partie nulle contre j, alors forcément j a fait une partie nulle contre i


solve satisfy;

output 
[
  "Scores\n"
]
++
[
  show(x)
]
++	
[
  "\nMatches\n"
]
++
[ show(matches[i,j]) ++ 
         if j == n then "\n" else " " endif |
         i in 1..n, j in 1..n
]
++
[
"Le joueur 5 a obtenu un score de "
]
++
[
show(sum(i in 1..n)(matches[5,i]))
]
;