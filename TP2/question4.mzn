% Bathylle de la Grandière - 2166208
% Augustin Barruol - 2161214

include "globals.mzn";
include "./data/question4-1.dzn";
%include "./data/question4-2.dzn";
%include "./data/question4-3.dzn";

int: n_nurse; % Number of nurses
int: n_week; % Number of weeks in our horizon
int: n_day; % Number of days in our horizon (7 * n_week). Monday is the first day of the week
int: n_work_period; % Number of work period. It equals 4 (MATIN, SOIREE, NUIT, REPOS)

% Indexing the work periods
int: MATIN = 1;
int: SOIREE = 2;
int: NUIT = 3;
int: REPOS = 4;

int: dur_work_period; % Duration of the work periods (6h)

array[1..n_day,1..n_work_period] of int: demande_min; % Number of nurses required for each day and each work period (= 0 for REPOS)
array[1..n_nurse,1..n_day,1..n_work_period] of int: preference; % Work preference for each nurse

% Decision variables
array[1..n_nurse,1..n_day,1..n_work_period] of var 0..1: x; % work period [MATIN, SOIREE, NUIT, REPOS]
var int: sum_preference;

% demande minimale journalière de l'hopital
constraint forall(day in 1..n_day)(
  forall(period in 1..n_work_period)(
    sum (nurse in 1..n_nurse) (x[nurse,day,period]) >= demande_min[day,period]
  )
);

% une infirmière ne peut travailler qu'une seule période par jour
constraint forall(nurse in 1..n_nurse)(
  forall(day in 1..n_day)(
    sum (period in 1..n_work_period) (x[nurse,day,period]) == 1
  )
);

% travail <= 36h par semaine
constraint forall(nurse in 1..n_nurse)( % pour chaque infirmier
  forall(week in 0..n_week-1)( % pour chaque semaine
    % on somme sur la semaine le nb de jours de travail multipliés par le nombre d'heures de travail
    sum (day in 7*week+1..7*week+7) ( 
      dur_work_period * sum(period in 1..n_work_period-1)(x[nurse,day,period]) ) <= 36
));

% minimum 4 jours dans la semaine
constraint forall(nurse in 1..n_nurse)(
  forall(week in 0..n_week-1)(
    % on somme sur la semaine le nb de jours de travail
    sum (day in 7*week+1..7*week+7) (sum(period in 1..n_work_period-1)(x[nurse,day,period])) >= 4
  )
);

% travail de nuit jour j oblige repos jour j+1
constraint forall(nurse in 1..n_nurse)(
  forall(day in 1..n_day-1 where x[nurse,day,NUIT] == 1) (x[nurse,day+1,REPOS] == 1)
);

% pas de jour de travail isolé
constraint forall(nurse in 1..n_nurse)(
  forall(day in 2..n_day-1 where x[nurse,day,REPOS] == 0)( % pour tout jour de travail
    (x[nurse,day-1,REPOS] != 1 \/ x[nurse,day+1,REPOS] != 1) % le jour précédent ou le jour suivant n'est pas un repos
  )
);

% travail au plus 3j consécutifs
constraint forall(nurse in 1..n_nurse)(
  forall(day in 1..n_day-3 where x[nurse,day,REPOS] == 0)(
    if (x[nurse,day+1,REPOS] == 0 /\ x[nurse,day+2,REPOS] == 0) then x[nurse,day+3,REPOS] == 1
    endif
  )
);

constraint sum(nurse in 1..n_nurse)( sum(day in 1..n_day)( sum(period in 1..n_work_period where x[nurse,day,period] == 1)(preference[nurse,day,period]) ) ) == sum_preference;

solve maximize sum_preference;

output
[
  "total: " ++ show(sum_preference) ++ "\n" ++
  "solution:\n\t    "
]
++
["Day \(j)\t" | j in 1..n_day]
++ 
[ 
  if j = 1 then "\n NURSE \(i): " else "\t" endif ++
    if fix(x[i,j,4]) == 1 then "--" elseif fix(x[i,j,3]) == 1  then "NUIT" elseif fix(x[i,j,2]) == 1  then "SOIREE" elseif fix(x[i,j,1]) == 1 then "MATIN" else "NONE" endif
  | i in 1..n_nurse, j in 1..n_day
];

