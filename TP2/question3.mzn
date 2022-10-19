% Bathylle de la Grandière - 2166208
% Augustin Barruol - 2161214

include "globals.mzn"; 
include "./data/question3-1.dzn";
%include "./data/question3-2.dzn";

int: n; % num rows and columns

int: L = 1; % Lamp
int: P = 2; % Plant

% 1 for a lamp, 2 for a plant and 0 for nothing 
array[1..n,1..n] of 0..2: garden;

% number of plants by row/col
array[1..n] of int: rows;
array[1..n] of int: cols;

% decision variables
array[1..n,1..n] of var 0..P: x;

% éléments déjà placés dans le jardin
constraint forall(i,j in 1..n where garden[i,j] != 0)(
  x[i,j] = garden[i,j]
);

% nombre de plantes sur chaque ligne fixé par rows[i]
constraint forall(i in 1..n)(
  sum(j in 1..n where x[i,j] = 2)(1) == rows[i]
);

% nombre de plantes sur chaque colonne fixé par cols[j]
constraint forall(j in 1..n)(
  sum(i in 1..n where x[i,j] = 2)(1) == cols[j]
);

%%% les plantes ne doivent pas se toucher
% plantes au milieu du jardin
constraint forall(i,j in 2..n-1 where x[i,j] == 2)(
  forall(k in i-1..i+1)(
    forall(l in j-1..j+1)(
      if not(k = i /\ l = j) then x[k,l] != 2 
      endif
    )
  )
);
% plantes sur le bord gauche
constraint forall(i in 2..n-1 where x[i,1] == 2)(
  forall(k in i-1..i+1)(
    forall(l in 1..2)(
      if not(k = i /\ l = 1) then x[k,l] != 2 
      endif
    )
  )
);
% plantes sur le bord droit
constraint forall(i in 2..n-1 where x[i,n] == 2)(
  forall(k in i-1..i+1)(
    forall(l in n-1..n)(
      if not(k = i /\ l = n) then x[k,l] != 2 
      endif
    )
  )
);
% plantes sur le bord haut
constraint forall(j in 2..n-1 where x[1,j] == 2)(
  forall(k in 1..2)(
    forall(l in j-1..j+1)(
      if not(k = 1 /\ l = j) then x[k,l] != 2 
      endif
    )
  )
);
% plantes sur le bord bas
constraint forall(j in 2..n-1 where x[n,j] == 2)(
  forall(k in n-1..n)(
    forall(l in j-1..j+1)(
      if not(k = n /\ l = j) then x[k,l] != 2 
      endif
    )
  )
);
% plantes dans les coins
constraint if x[1,1] == 2 then x[1,2] != 2 /\ x[2,1] != 2 /\ x[2,2] != 2 endif;
constraint if x[1,n] == 2 then x[1,n-1] != 2 /\ x[2,n] != 2 /\ x[2,n-1] != 2 endif;
constraint if x[n,1] == 2 then x[n,2] != 2 /\ x[n-1,1] != 2 /\ x[n-1,2] != 2 endif;
constraint if x[n,n] == 2 then x[n,n-1] != 2 /\ x[n-1,n] != 2 /\ x[n-1,n-1] != 2 endif;

%%% plantes à côté d'une lampe
% au milieu du jardin
constraint forall(i,j in 2..n-1 where x[i,j] == 2)(
  x[i-1,j] == 1 \/ x[i+1,j] == 1 \/ x[i,j-1] == 1 \/ x[i,j+1] == 1
);
% sur le bord gauche
constraint forall(i in 2..n-1 where x[i,1] == 2)(
  x[i-1,1] == 1 \/ x[i+1,1] == 1 \/ x[i,2] == 1
);
% sur le bord droit
constraint forall(i in 2..n-1 where x[i,n] == 2)(
  x[i-1,n] == 1 \/ x[i+1,n] == 1 \/ x[i,n-1] == 1
);
% sur le bord haut
constraint forall(j in 2..n-1 where x[1,j] == 2)(
  x[1,j-1] == 1 \/ x[1,j+1] == 1 \/ x[2,j] == 1
);
% sur le bord bas
constraint forall(j in 2..n-1 where x[n,j] == 2)(
  x[n,j-1] == 1 \/ x[n,j+1] == 1 \/ x[n-1,j] == 1
);
% dans les coins
constraint if x[1,1] == 2 then x[1,2] == 1 \/ x[2,1] == 1 endif;
constraint if x[1,n] == 2 then x[1,n-1] == 1 \/ x[2,n] == 1 endif;
constraint if x[n,1] == 2 then x[n,2] == 1 /\ x[n-1,1] == 1 endif;
constraint if x[n,n] == 2 then x[n,n-1] == 1 /\ x[n-1,n] == 1 endif;


%%% lampes à côté d'une plante
% au milieu du jardin
constraint forall(i,j in 2..n-1 where x[i,j] == 1)(
  x[i-1,j] == 2 \/ x[i+1,j] == 2 \/ x[i,j-1] == 2 \/ x[i,j+1] == 2
);
% sur le bord gauche
constraint forall(i in 2..n-1 where x[i,1] == 1)(
  x[i-1,1] == 2 \/ x[i+1,1] == 2 \/ x[i,2] == 2
);
% sur le bord droit
constraint forall(i in 2..n-1 where x[i,n] == 1)(
  x[i-1,n] == 2 \/ x[i+1,n] == 2 \/ x[i,n-1] == 2
);
% sur le bord haut
constraint forall(j in 2..n-1 where x[1,j] == 1)(
  x[1,j-1] == 2 \/ x[1,j+1] == 2 \/ x[2,j] == 2
);
% sur le bord bas
constraint forall(j in 2..n-1 where x[n,j] == 1)(
  x[n,j-1] == 2 \/ x[n,j+1] == 2 \/ x[n-1,j] == 2
);
% dans les coins
constraint if x[1,1] == 1 then x[1,2] == 2 \/ x[2,1] == 2 endif;
constraint if x[1,n] == 1 then x[1,n-1] == 2 \/ x[2,n] == 2 endif;
constraint if x[n,1] == 1 then x[n,2] == 2 /\ x[n-1,1] == 2 endif;
constraint if x[n,n] == 1 then x[n,n-1] == 2 /\ x[n-1,n] == 2 endif;

output 
[
 if j = 1 then "\n" ++ show(rows[i]) ++ " " else " " endif ++
   if fix(x[i,j]) = L then "L" elseif fix(x[i,j]) = P then "P" else "_" endif
| i,j in 1..n
] ++ ["\n  "] ++ [show(cols[i]) ++ " " | i in 1..n];
