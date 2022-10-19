% Bathylle de la Grandière - 2166208
% Augustin Barruol - 2161214


include "globals.mzn"; 
include "./data/question1-1.dzn";
%include "./data/question1-2.dzn";
%include "./data/question1-3.dzn";


% Variables 
int: r; % number of rows 
int: c; % number of columns
array[1..r] of int: rowinsects; % total of insects by row
array[1..c] of int: colinsects; % total of insects by col
array[1..r, 1..c] of 0..r*c: knownpositions;  % known number of insects, 0 appears when unknown

array[1..r, 1..c] of var 1..r*c: x; % the final garden grid (decision variable)

% Contraintes
constraint forall(k in 1..r)(sum (i in 1..c)(x[k,i]) == rowinsects[k]); % somme des éléments de la ligne k de x == rowinsects[k]
constraint forall(k in 1..c)(sum (i in 1..r)(x[i,k]) == colinsects[k]); % somme des éléments de la colonne k de x == colinsects[k]
constraint forall(i in 1..r)(forall(j in 1..c)(if knownpositions[i,j] != 0 then x[i,j] == knownpositions[i,j] endif)); % si une position est connue, elle doit être utilisée
constraint forall(i in 1..r)(alldifferent([ x[i,j] | j in 1..c ])); % tous les éléments d'une même ligne sont différents
constraint forall(i in 1..c)(alldifferent([ x[j,i] | j in 1..r ]));% tous les éléments d'une même colonne sont différents

solve satisfy;

% Résultats
output [
  "["
]
++
[
  if j = 1 then "\n" else "," endif ++
    if fix(x[i,j]) < 10 then "  " else " "  endif ++
    show(x[i,j]) ++
  if j = c then 
    ", f(" ++ show(rowinsects[i]) ++ "),"
  else 
    ""
  endif
  | i in 1..r, j in 1..c
] 
++
["\n"]++
[
  "f(" ++ show(colinsects[j]) ++ "),"
  | j in 1..c
]

++ ["]\n"]
;